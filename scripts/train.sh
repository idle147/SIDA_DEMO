export CUDA_VISIBLE_DEVICES=0,1,2
deepspeed --num_gpus=3 --master_port=24999 train_ds.py \
  --version="liuhaotian/LLaVA-Lightning-7B-delta-v1-1" \
  --dataset_dir='./dataset' \
  --vision_pretrained="/home/yuyangxin/data/pretrain_models/sam_vit_h_4b8939.pth" \
  --dataset="magic_brush" \
  --sample_rates="9,3,3,1" \
  --exp_name="lisa-7b"