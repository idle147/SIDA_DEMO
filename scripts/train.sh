/home/yuyangxin/anaconda3/envs/llm/bin/deepspeed --num_nodes=1 --num_gpus=3 train_ds.py \
  --version="liuhaotian/LLaVA-Lightning-7B-delta-v1-1" \
  --dataset_dir='./dataset' \
  --vision_pretrained="/home/yuyangxin/data/pretrain_models/sam_vit_h_4b8939.pth" \
  --dataset="reason_seg" \
  --sample_rates="9,3,3,1" \
  --exp_name="lisa-7b"
