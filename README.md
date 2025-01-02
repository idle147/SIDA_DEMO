# 论文复现
https://github.com/hzlsaber/SIDA

# 环境安装
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# 数据集下载
https://huggingface.co/datasets/osunlp/MagicBrush/viewer


# TODO:
1. 没有文本模态的说明, 只复现了 mask分类 和 detection检测的部分

# Train

`
sh ./scripts/train.py

# When training is finished, to get the full model weight:
cd ./runs/lisa-7b/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
`

# Merge LoRA Weight
Merge the LoRA weights of pytorch_model.bin, save the resulting model into your desired path in the Hugging Face format:
`
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_LLaVA" \
  --weight="PATH_TO_pytorch_model.bin" \
  --save_path="PATH_TO_SAVED_MODEL"

`
`
For example:
CUDA_VISIBLE_DEVICES="" python3 merge_lora_weights_and_save_hf_model.py \
  --version="./LLaVA/LLaVA-Lightning-7B-v1-1" \
  --weight="lisa-7b/pytorch_model.bin" \
  --save_path="./LISA-7B"
`

# Validation
deepspeed --master_port=24999 train_ds.py \
  --version="PATH_TO_LISA_HF_Model_Directory" \
  --dataset_dir='./dataset' \
  --vision_pretrained="PATH_TO_SAM" \
  --exp_name="lisa-7b" \
  --eval_only