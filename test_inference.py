import os
import sys
import torch

# Ensure SegEarth-R1 is in python path
sys.path.insert(0, '/kaggle/working/SegEarth-R1')

from segearth_r1.model.language_model.llava_phi import segearth_r1, LlavaConfig
from segearth_r1.train.train_dataset import get_mask_config
from segearth_r1.constants import IMAGE_TOKEN_INDEX

def test_inference():
    print("1. Creating dummy model configuration...")
    config = LlavaConfig()
    
    # Configure dummy model sizes
    config.vocab_size = 51200
    config.hidden_size = 2048
    config.num_attention_heads = 32
    config.num_hidden_layers = 12
    config.swin_type = "base"
    config.mm_vision_tower = "dummy_swin"
    config.mask_decode_train = True
    config.mm_projector_type = "SparseConv_1"
    config.mm_hidden_size = 256
    config.projector_outdim = 2048
    config.with_norm = True
    config.with_layernorm = False

    print("2. Loading mask decoder config...")
    # Load mask config from yaml
    mask_cfg = get_mask_config('/kaggle/working/SegEarth-R1/segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml')
    mask_cfg.MODEL.MASK_FORMER.SEG_TASK = 'referring'

    print("3. Instantiating model with random weights on GPU...")
    # Instantiate model
    model = segearth_r1(config, mask_decoder_cfg=mask_cfg, use_seg_query=False).cuda()
    model.initial_mask_module()
    model.eval()
    print("✅ Model initialized successfully!")

    print("4. Preparing dummy inputs...")
    batch_size = 1
    seq_len = 10
    
    # Create input ids (dummy text tokens + 1 image token)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
    input_ids[0, 3] = IMAGE_TOKEN_INDEX  # Place image token index
    
    attention_mask = torch.ones((batch_size, seq_len)).cuda()
    
    # Create random image tensor [1, 3, 1024, 1024]
    images = torch.randn(batch_size, 3, 1024, 1024).cuda()
    
    # Define dummy masks for the target region
    masks = [torch.randn(1, 1024, 1024).cuda() > 0]
    
    # Define reference token indices
    refer_embedding_indices = [torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.long).cuda()]

    print("5. Running model forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            masks=masks,
            refer_embedding_indices=refer_embedding_indices,
            dataset_type=['refer_seg']
        )
    
    print("\n✅ Forward pass succeeded without errors!")
    print(f"Outputs keys: {outputs.keys()}")

if __name__ == "__main__":
    test_inference()
