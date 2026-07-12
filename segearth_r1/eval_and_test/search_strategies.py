"""
Search strategies for SegEarth-R1 evaluation.

Provides:
  - generate_candidates: LLM text generation for candidate answers.
  - score_candidates: Scores candidate answers via eval_seg.
  - best_of_n_search: Best-of-N sampling (generate N, pick highest score).
  - self_consistency_search: IoU-based consensus clustering over N samples
    (adapted from Wang et al. 2022, arXiv:2203.11171).
"""

import torch
from typing import List, Optional, Dict, Callable

from segearth_r1.constants import IMAGE_TOKEN_INDEX
from segearth_r1.mm_utils import tokenizer_image_token
from segearth_r1 import conversation as conversation_lib

# Prefix strings matching the dataset classes exactly.
PREFIX_INST_REASONING = (
    "This is an image <image>, Please doing Reasoning Segmentation "
    "according to the following instruction:"
)
PREFIX_INST_REFERRING = (
    "This is an image <image>, Please doing Referring Segmentation "
    "according to the following instruction:"
)

# Default for backward compat (ReasonSegDataset / EarthReason)
PREFIX_INST = PREFIX_INST_REASONING


def decode_question_text(tokenizer, token_refer_id: torch.Tensor) -> str:
    """Recover raw referring/question text from an already-tokenized
    ``token_refer_id`` (built by ``preprocess_referring_instruction``,
    which is ``tokenizer.encode(instruction) + [SEG_token_id]``).
    We drop the trailing appended token before decoding."""
    ids = token_refer_id[:-1] if token_refer_id.numel() > 1 else token_refer_id
    return tokenizer.decode(ids.cpu().tolist(), skip_special_tokens=True).strip()


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
@torch.no_grad()
def generate_candidates(
    model,
    tokenizer,
    image_tensor: torch.Tensor,   # [1, 3, H, W]
    question_text: str,
    device,
    num_return: int = 1,
    num_beams: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 64,
    conv_version: str = "llava_phi",
    prefix_inst: str = PREFIX_INST,
    seed: Optional[int] = None,
) -> List[str]:
    """Single ``model.generate()`` call producing ``num_return`` candidate
    answer strings for one image."""
    if seed is not None:
        torch.manual_seed(seed)

    conv = conversation_lib.conv_templates[conv_version].copy()
    human_turn = f"{prefix_inst} {question_text}"
    conv.append_message(conv.roles[0], human_turn)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    # num_return_sequences > 1 requires do_sample=True or num_beams > 1
    if num_return > 1 and num_beams <= 1 and not do_sample:
        do_sample = True

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=image_tensor.to(device).float(),
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return,
        # use_cache=False: the model's KV-cache-based attention_mask rebuilding
        # (prepare_inputs_for_generation / the input_ids.shape[1]==1 branch in
        # prepare_inputs_labels_for_multimodal) computes mask length from raw
        # token/cache counts and doesn't account for image/refer-token expansion,
        # causing a width mismatch after the first decode step. Recomputing the
        # full sequence each step avoids that bug (slower, but correct here since
        # max_new_tokens is small).
        use_cache=False,
    )
    if num_beams > 1:
        gen_kwargs.update(num_beams=num_beams, do_sample=False, early_stopping=True)
    elif do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    print("image_tensor shape:", image_tensor.shape, "dtype:", image_tensor.dtype,
          "min:", image_tensor.min().item(), "max:", image_tensor.max().item())
    out = model.generate(**gen_kwargs)

    prompt_len = input_ids.shape[1]
    texts = []
    for row in out:
        new_tokens = row[prompt_len:]
        texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return texts


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
@torch.no_grad()
def score_candidates(
    model,
    tokenizer,
    item_tensors: Dict[str, torch.Tensor],
    cand_texts: List[str],
    device,
    preprocess_referring_instruction: Callable,
) -> List[dict]:
    """Score all candidate answer strings for ONE image via ``eval_seg``.

    Reuses the item's existing ``input_ids``/``labels``/embedding indices/
    ``token_refer_id`` and only rebuilds ``token_answer_id`` per candidate.
    """
    n = len(cand_texts)
    input_ids = item_tensors["input_ids"].unsqueeze(0).repeat(n, 1).to(device)
    labels = item_tensors["labels"].unsqueeze(0).repeat(n, 1).to(device)
    attention_mask = torch.ones_like(input_ids)
    images = item_tensors["images"].unsqueeze(0).repeat(n, 1, 1, 1).to(device)
    refer_embedding_indices = (
        item_tensors["refer_embedding_indices"].unsqueeze(0).repeat(n, 1).to(device)
    )
    # answer_embedding_indices may not exist for referring datasets
    if "answer_embedding_indices" in item_tensors:
        answer_embedding_indices = (
            item_tensors["answer_embedding_indices"].unsqueeze(0).repeat(n, 1).to(device)
        )
    else:
        answer_embedding_indices = None
    token_refer_id = [item_tensors["token_refer_id"].to(device) for _ in range(n)]
    token_answer_id = [
        preprocess_referring_instruction(t, tokenizer).to(device) for t in cand_texts
    ]

    outputs = model.eval_seg(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images.float(),
        masks=None,
        token_refer_id=token_refer_id,
        refer_embedding_indices=refer_embedding_indices,
        labels=labels,
        token_answer_id=token_answer_id,
        answer_embedding_indices=answer_embedding_indices,
    )

    results = []
    for text, out in zip(cand_texts, outputs):
        pred_masks = out["pred_masks"]
        # SEG_instance_inference drops the leading query dim when there's only
        # one mask query (mask_pred.shape[0] == 1), returning [H, W] instead of
        # [1, H, W]. Restore it so the [:1]/[best_q:best_q+1] indexing below
        # always operates on a 3D [Q, H, W] tensor.
        if pred_masks.dim() == 2:
            pred_masks = pred_masks.unsqueeze(0)
        scores = out.get("scores")
        # scores is either a Tensor or None (from SEG_instance_inference)
        if scores is not None:
            scores_t = scores if isinstance(scores, torch.Tensor) else torch.tensor(scores)
            if scores_t.numel() > 0:
                best_q = int(torch.argmax(scores_t))
                mask = pred_masks[best_q : best_q + 1]
                score = float(scores_t[best_q])
            else:
                mask = pred_masks[:1] if pred_masks.dim() >= 2 else pred_masks.unsqueeze(0)
                score = 0.0
        else:
            mask = pred_masks[:1] if pred_masks.dim() >= 2 else pred_masks.unsqueeze(0)
            # pred_masks from eval_seg are already binarized (> 0).float(),
            # so .mean() gives the foreground ratio as a confidence proxy.
            score = float(mask.float().mean())
        results.append({"text": text, "mask": mask.detach().cpu(), "score": score})
    return results


# --------------------------------------------------------------------------
# Search strategies
# --------------------------------------------------------------------------
@torch.no_grad()
def best_of_n_search(
    model, tokenizer, item_tensors, question_text, device,
    preprocess_referring_instruction: Callable,
    n: int = 8, temperature: float = 1.0, top_p: float = 0.95,
    max_new_tokens: int = 64, conv_version: str = "llava_phi",
    prefix_inst: str = PREFIX_INST,
) -> dict:
    """Generate N candidate answers, score them, return the best one."""
    image = item_tensors["images"].unsqueeze(0)
    cand_texts = generate_candidates(
        model, tokenizer, image, question_text, device,
        num_return=n, do_sample=True, temperature=temperature, top_p=top_p,
        max_new_tokens=max_new_tokens, conv_version=conv_version,
        prefix_inst=prefix_inst,
    )
    scored = score_candidates(
        model, tokenizer, item_tensors, cand_texts, device,
        preprocess_referring_instruction,
    )
    return max(scored, key=lambda r: r["score"])


@torch.no_grad()
def self_consistency_search(
    model, tokenizer, item_tensors, question_text, device,
    preprocess_referring_instruction: Callable,
    n: int = 8,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 64,
    conv_version: str = "llava_phi",
    prefix_inst: str = PREFIX_INST,
    iou_thresh: float = 0.5,
    mask_prob_thresh: float = 0.5,
) -> dict:
    """Self-consistency via IoU-based consensus clustering (Wang et al. 2022).

    Samples N reasoning chains, clusters the resulting masks by IoU,
    and returns the per-pixel majority-vote mask from the largest cluster.
    """
    image = item_tensors["images"].unsqueeze(0)
    cand_texts = generate_candidates(
        model, tokenizer, image, question_text, device,
        num_return=n, do_sample=True, temperature=temperature, top_p=top_p,
        max_new_tokens=max_new_tokens, conv_version=conv_version,
        prefix_inst=prefix_inst,
    )
    scored = score_candidates(
        model, tokenizer, item_tensors, cand_texts, device,
        preprocess_referring_instruction,
    )

    # Masks from eval_seg are already binarized via (logits > 0).float(),
    # so just threshold at 0.5 (effectively treating 1.0 as positive).
    bin_masks = [(r["mask"] > mask_prob_thresh).bool() for r in scored]

    def iou(a: torch.Tensor, b: torch.Tensor) -> float:
        inter = (a & b).sum().item()
        union = (a | b).sum().item()
        return inter / union if union > 0 else 0.0

    # Greedy consensus clustering.
    clusters: List[List[int]] = []
    for i, m in enumerate(bin_masks):
        placed = False
        for cluster in clusters:
            if iou(m, bin_masks[cluster[0]]) >= iou_thresh:
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    best_cluster = max(clusters, key=len)
    stacked = torch.stack([bin_masks[i] for i in best_cluster]).float()
    consensus_mask = (stacked.mean(dim=0) > 0.5)  # per-pixel majority vote

    return {
        "text": scored[best_cluster[0]]["text"],
        "mask": consensus_mask,
        "cluster_size": len(best_cluster),
        "n_clusters": len(clusters),
        "agreement_ratio": len(best_cluster) / n,
    }


# --------------------------------------------------------------------------
# TTA (Test-Time Augmentation) for referring segmentation
# --------------------------------------------------------------------------
def _flip_image(img: torch.Tensor, mode: str) -> torch.Tensor:
    """Flip a [C, H, W] or [B, C, H, W] image tensor."""
    if mode == "h":
        return torch.flip(img, [-1])
    elif mode == "v":
        return torch.flip(img, [-2])
    elif mode == "hv":
        return torch.flip(img, [-2, -1])
    return img


def _flip_mask(mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Inverse-flip a predicted mask back to original orientation."""
    # Same ops as forward flip since flip is self-inverse.
    return _flip_image(mask, mode)


@torch.no_grad()
def tta_eval(
    model,
    inputs: Dict[str, torch.Tensor],
    device,
    augmentations: Optional[List[str]] = None,
) -> List[dict]:
    """Test-Time Augmentation for referring segmentation.

    Runs ``eval_seg`` on the original image plus geometric augmentations
    (horizontal flip, vertical flip, both), averages the sigmoid mask
    logits across all views, and re-binarizes.

    Args:
        model: The segearth_r1 model in eval mode.
        inputs: A single batch dict (already on device) with keys:
            input_ids, attention_mask, images, masks, token_refer_id,
            refer_embedding_indices, labels, and optionally
            token_answer_id / answer_embedding_indices.
        device: Target device.
        augmentations: List of flip modes to apply in addition to the
            original. Defaults to ``["h", "v", "hv"]``.

    Returns:
        List of result dicts (same format as ``eval_seg``), one per
        batch item, with TTA-averaged masks.
    """
    if augmentations is None:
        augmentations = ["h", "v", "hv"]

    has_answer = "token_answer_id" in inputs and inputs["token_answer_id"] is not None

    def _run_eval_seg(images_tensor):
        kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            images=images_tensor.float(),
            masks=inputs["masks"],
            token_refer_id=inputs["token_refer_id"],
            refer_embedding_indices=inputs["refer_embedding_indices"],
            labels=inputs["labels"],
            token_answer_id=inputs.get("token_answer_id") if has_answer else None,
            answer_embedding_indices=inputs.get("answer_embedding_indices") if has_answer else None,
        )
        return model.eval_seg(**kwargs)

    # Original pass
    all_runs = [_run_eval_seg(inputs["images"])]

    # Augmented passes
    for aug in augmentations:
        aug_images = _flip_image(inputs["images"], aug)
        all_runs.append(_run_eval_seg(aug_images))

    # Merge: average sigmoid masks across all augmentation views
    batch_size = len(all_runs[0])
    merged = []
    for b in range(batch_size):
        # Collect mask logits from every augmentation for this batch item
        masks_to_avg = []
        for run_idx, run in enumerate(all_runs):
            pred = run[b]["pred_masks"].float()
            # Pick the best mask if scores are available
            scores = run[b].get("scores")
            if scores is not None:
                scores_t = scores if isinstance(scores, torch.Tensor) else torch.tensor(scores)
                if scores_t.numel() > 0:
                    best_q = int(torch.argmax(scores_t))
                    pred = pred[best_q:best_q + 1]
                else:
                    pred = pred[:1] if pred.dim() >= 2 else pred.unsqueeze(0)
            else:
                pred = pred[:1] if pred.dim() >= 2 else pred.unsqueeze(0)

            # Inverse-flip the mask back to original orientation
            aug_mode = "" if run_idx == 0 else augmentations[run_idx - 1]
            if aug_mode:
                pred = _flip_mask(pred, aug_mode)
            # pred_masks from eval_seg are already binarized (0/1 float),
            # so we average them directly (no sigmoid needed).
            masks_to_avg.append(pred)

        avg_mask = torch.stack(masks_to_avg).mean(dim=0)
        binary_mask = (avg_mask > 0.5).float()
        merged.append({
            "pred_masks": binary_mask,
            "scores": [],  # already merged, no per-query scores
        })

    return merged