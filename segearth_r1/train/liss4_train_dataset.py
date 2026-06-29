import os
import re
import cv2
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from segearth_r1.constants import IGNORE_INDEX, REFER_TOKEN_INDEX, ANSWER_TOKEN_INDEX
from segearth_r1.train.train_dataset import (
    preprocess_image,
    preprocess_mask,
    preprocess_llama2,
    preprocess_referring_instruction
)

def get_split_patches(meta_assignment_path, split):
    assignments = []
    if not os.path.exists(meta_assignment_path):
        raise FileNotFoundError(f"Meta assignment file not found at: {meta_assignment_path}")

    with open(meta_assignment_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 3:
                mask_id, class_name, class_id = parts
                if mask_id.lower() in ('patch_id', 'mask_id'):
                    continue
                # Determine image/patch ID
                parts_to_remove = class_name.count('_') + 1
                patch_id = mask_id.rsplit('_', parts_to_remove)[0]
                assignments.append((patch_id, mask_id, class_name, class_id))
                
    # Extract unique patch IDs, sort them to ensure deterministic split
    unique_patches = sorted(list(set(item[0] for item in assignments)))
    
    n = len(unique_patches)
    if n <= 3:
        train_idx = max(1, int(n * 0.6))
        val_idx = min(n - 1, train_idx + 1)
    else:
        train_idx = int(n * 0.7)
        val_idx = int(n * 0.85)
        if val_idx == train_idx:
            val_idx = train_idx + 1
            
    train_patches = unique_patches[:train_idx]
    val_patches = unique_patches[train_idx:val_idx]
    test_patches = unique_patches[val_idx:]
    
    if split == 'train':
        selected = set(train_patches)
    elif split == 'val':
        selected = set(val_patches)
    elif split == 'test':
        selected = set(test_patches)
    else:
        selected = set(unique_patches)
        
    filtered = []
    for patch_id, mask_id, class_name, class_id in assignments:
        if patch_id in selected:
            filtered.append((patch_id, mask_id, class_name, class_id))
    return filtered


class Liss4ReasonSegDataset(Dataset):
    def __init__(self, base_data_path, tokenizer, split='train', image_size=1024):
        super(Liss4ReasonSegDataset, self).__init__()
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.base_data_path = base_data_path
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.split = split

        # Configure paths based on the requested dataset structure
        # Assume base_data_path points to D:\Liss4Patches_processed
        self.images_root = os.path.join(base_data_path, "images")
        self.labels_root = os.path.join(base_data_path, "labels_liss4")
        self.qas_root = os.path.join(base_data_path, "QAs_liss4")
        self.meta_assignment = os.path.join(base_data_path, "meta_assignment_sentinal2.txt")

        # Load split assignments
        self.assignments = get_split_patches(self.meta_assignment, split)
        print(f"Loaded {len(self.assignments)} samples for split '{split}' from {self.meta_assignment}")

    def __len__(self):
        return len(self.assignments)

    def __getitem__(self, idx):
        patch_id, mask_id, class_name, class_id = self.assignments[idx]
        data_dict = {}

        # 1. Load FCC Image
        image_path = os.path.join(self.images_root, f"{patch_id}.jpg")
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = preprocess_image(image, self.image_size)
        processed_image = (torch.tensor(processed_image) - self.pixel_mean) / self.pixel_std
        data_dict['image'] = processed_image
        data_dict['image_name'] = mask_id

        # 2. Load Label Mask
        label_path = os.path.join(self.labels_root, f"{mask_id}.png")
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Label mask not found: {label_path}")
        mask[mask != 0] = 1
        processed_mask = preprocess_mask(mask, self.image_size)
        data_dict['mask'] = processed_mask

        # 3. Load QAs
        qa_path = os.path.join(self.qas_root, f"{mask_id}_qa.json")
        if not os.path.exists(qa_path):
            raise FileNotFoundError(f"QA JSON not found: {qa_path}. Please run question_answe_1.py to generate it first.")
        with open(qa_path, "r", encoding="utf-8") as file:
            qa_data = json.load(file)

        question = qa_data["questions"][0]
        answer_num = len(qa_data["answer"])
        if answer_num == 0:
            answer = "There is no target object in the scene."
        else:
            answer = qa_data["answer"][0]

        prefix_inst = 'This is an image <image>, Please doing Reasoning Segmentation according to the following instruction:'
        instruction = ' {}'.format(question)
        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': 'Sure, It is <seg>. \n<answer>.'}]]
        
        text_dict = preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'reason_seg'
        
        token_refer_id = preprocess_referring_instruction(instruction, self.tokenizer)
        token_answer_id = preprocess_referring_instruction(answer, self.tokenizer)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
        answer_embedding_indices = torch.zeros_like(input_ids)
        answer_embedding_indices[input_ids == ANSWER_TOKEN_INDEX] = 1
        
        data_dict['token_refer_id'] = token_refer_id
        data_dict['token_answer_id'] = token_answer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        data_dict['answer_embedding_indices'] = answer_embedding_indices

        return data_dict


if __name__ == "__main__":
    # Test block
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = Liss4ReasonSegDataset(
        base_data_path="D:\\Liss4Patches_processed",
        tokenizer=tokenizer,
        split='all'
    )
    print(f"Total samples: {len(dataset)}")
    try:
        sample = dataset[0]
        print("Keys in sample:", sample.keys())
        print("Image shape:", sample['image'].shape)
        print("Mask shape:", sample['mask'].shape)
    except Exception as e:
        print("Could not load sample (expected if QAs not generated yet):", e)
