import glob
import json
import os
from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from models.llava import conversation as conversation_lib
from models.segment_anything.utils.transforms import ResizeLongestSide

from .forensic_utils import ANSWER_LIST, LONG_QUESTION_LIST


class MagicBrushDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 500,
        exclude_val=False,
        explanatory=0.1,
        *args,
        **kwargs,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory

        self.base_image_dir = Path(base_image_dir) / "MagicBrush"
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        edit_sessions = self.base_image_dir / "edit_sessions.json"
        with open(edit_sessions) as f:
            content = json.load(f)

        real_images, edited_images = [], []
        for name, infos in content.items():
            for info in infos:
                real_image_path = self.base_image_dir / "images" / name / info["input"]
                real_images.append([real_image_path, "Negative"])
                edited_img_path = self.base_image_dir / "images" / name / info["output"]
                mask_image_path = self.base_image_dir / "images" / name / info["mask"]
                edited_images.append([edited_img_path, mask_image_path])

        print("len(real_images): ", len(real_images))
        print("len(edited_images): ", len(edited_images))
        self.images = real_images + edited_images
        random.shuffle(self.images)

        # if explanatory != -1:
        #     self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
        #     self.img_to_explanation = {}
        #     with open(
        #         os.path.join(
        #             base_image_dir,
        #             reason_seg_data,
        #             "explanatory",
        #             "train.json",
        #         )
        #     ) as f:
        #         content = json.load(f)
        #     for item in content:
        #         img_name = item["image"]
        #         self.img_to_explanation[img_name] = {
        #             "query": item["query"],
        #             "outputs": item["outputs"],
        #         }

        #     print("len(self.img_to_explanation): ", len(self.img_to_explanation))

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        image_path, mask_path = self.images[idx]

        # 读取并转换图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 预处理图像以适配clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # 预处理图像以适配sam
        image = self.transform.apply_image(image)
        resize = tuple(image.shape[:2])

        # 随机选择问题和答案
        question_template = random.choice(self.long_question_list)
        questions = [question_template]
        answers = [random.choice(self.answer_list)]

        # 构建对话
        conversations = []
        for question, answer in zip(questions, answers):
            conv = conversation_lib.default_conversation.copy()
            conv.messages = []
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
            conversations.append(conv.get_prompt())

        # 预处理图像
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        image_shape = tuple(image.shape)

        # 处理掩码
        if mask_path == "Negative":
            # 创建与图像大小相同的全黑掩码
            mask = np.zeros((image_shape[1], image_shape[2]), dtype=np.uint8)
            detection_label = "real"
        else:
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, (image_shape[2], image_shape[1]), interpolation=cv2.INTER_NEAREST)

            # 定义要提取的颜色
            colors_to_extract = [(31, 31, 31), (20, 20, 20)]

            # 初始化二值掩码
            binary_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

            # 提取指定颜色的像素点
            for color in colors_to_extract:
                color_mask = cv2.inRange(mask, np.array(color), np.array(color))
                binary_mask = cv2.bitwise_or(binary_mask, color_mask)

            # 将掩码值转换为0和1
            mask = np.where(binary_mask > 0, 1, 0).astype(np.uint8)
            detection_label = "fake"

        # 将掩码转换为张量
        masks = torch.from_numpy(np.expand_dims(mask, axis=0))
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            detection_label,
            False,
        )
