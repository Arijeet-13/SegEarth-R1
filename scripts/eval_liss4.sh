export CUDA_VISIBLE_DEVICES=0
python segearth_r1/eval_and_test/eval.py \
  --model_path ./checkpoint/SegEarth-R1_Liss4Reason \
  --base_data_path "D:/Liss4Patches_processed" \
  --data_split "val" \
  --version "llava_phi" \
  --mask_config "segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml" \
  --dataset_type "Liss4Reason"
