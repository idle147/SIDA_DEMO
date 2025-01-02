import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from .llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaModel
from .segment_anything import build_sam_vit_h


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,
    eps=1e-6,
):
    """
    计算 DICE 损失，类似于 masks 的 generalized IOU。

    参数:
        inputs: 任意形状的浮点张量。每个示例的预测。
        targets: 与 inputs 形状相同的浮点张量。存储输入中每个元素的二分类标签。
                (0 表示负类，1 表示正类)。
        num_masks: float，表示 mask 的数量。
        scale: float，缩放因子。
        eps: float，避免除零的微小值。

    返回:
        损失张量。
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    计算交叉熵损失。

    参数:
        inputs: 任意形状的浮点张量。每个示例的预测。
        targets: 与 inputs 形状相同的浮点张量。存储输入中每个元素的二分类标签。
                (0 表示负类，1 表示正类)。

    返回:
        损失张量。
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel(LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    """
    LlavaLlamaForCausalLM 是一种基于 LLaMA（Large Language Model Meta AI）模型的变体，专门用于因果语言建模任务。
    因果语言建模（Causal Language Modeling）是一种生成式任务，模型根据前面的文本生成后续的文本。
    """

    def __init__(self, config, **kwargs):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get("vision_tower", "openai/clip-vit-large-patch14")
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", 1.0)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", 1.0)
        else:
            config.mm_vision_tower = config.vision_tower

        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.det_token_idx = kwargs.pop("det_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 使用多头注意力替代卷积模块
        self.attention_module = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.fc_1 = nn.Linear(256, 256)
        self.fc_detection = nn.Linear(256, 1)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.initialize_weights()
        self.post_init()

    def initialize_weights(self):
        # 初始化 attention_module
        for name, param in self.attention_module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            param.requires_grad = True

        # 初始化其他新添加的层，如 fc_1 和 fc_detection
        for name, param in self.fc_1.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            # 确保这些参数需要梯度
            param.requires_grad = True

        for name, param in self.fc_detection.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            # 确保这些参数需要梯度
            param.requires_grad = True

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(pixel_values[i].unsqueeze(0))
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def create_token_embeddings(
        self,
        input_ids: torch.LongTensor,
        token_idx: int,
        offset: torch.LongTensor,
        last_hidden_state: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        提取与指定标志位（如seg_token或det_token）相关的特征。

        参数:
            input_ids: 输入的 token ID。
            token_idx: 要提取特征的标志位的索引（如 seg_token_idx 或 det_token_idx）。
            offset: 用于分割的偏移量，通常与批次相关。
            last_hidden_state: 模型最后一层的隐藏状态。

        返回:
            pred_embeddings: 提取的嵌入特征列表。
        """
        # 确定输入序列中哪些位置是目标标志位，并跳过第一个[CLS]标记
        token_mask = input_ids[:, 1:] == token_idx
        token_mask = torch.cat(
            [
                token_mask,
                torch.zeros((token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        token_mask = torch.cat([torch.zeros((token_mask.shape[0], 255)).bool().cuda(), token_mask], dim=1)

        pred_embeddings = last_hidden_state[token_mask]
        seg_token_counts = token_mask.int().sum(-1)  # [bs, ]

        token_offset = seg_token_counts.cumsum(-1)
        token_offset = torch.cat([torch.zeros(1).long().cuda(), token_offset], dim=0)

        token_offset = token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(token_offset) - 1):
            start_i, end_i = token_offset[i], token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        return pred_embeddings_

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        # 基于SAM的Image encoder以获取图像视觉层面的嵌入
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        if inference:
            # 模型进入推理模式
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1

            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            # 模型进行一次向前传播, 保存下来
            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            # 维度对齐
            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None
        else:
            images_clip_list = []

            # 分块以优化内存
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = images_clip[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            # 检查输入数据是否有 NaN 或 Inf
            assert not torch.isnan(input_ids).any(), "Input IDs contain NaN"
            assert not torch.isinf(input_ids).any(), "Input IDs contain Inf"

            # 大模型输出结果
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        # 计算 last_hidden_state
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        # 使用 create_token_embeddings 提取 seg_token 的嵌入
        pred_embeddings_seg = self.create_token_embeddings(
            input_ids=input_ids,
            token_idx=self.seg_token_idx,
            offset=offset,
            last_hidden_state=last_hidden_state,
        )

        # 使用 create_token_embeddings 提取 det_token 的嵌入
        pred_embeddings_det = self.create_token_embeddings(
            input_ids=input_ids,
            token_idx=self.det_token_idx,
            offset=offset,
            last_hidden_state=last_hidden_state,
        )

        pred_masks, pred_det = [], []
        for i in range(len(pred_embeddings_seg)):
            # 获取预测的 det
            res = self.fc_detection(pred_embeddings_det[i])
            pred_det.append(res)

            # 调整形状
            h_det_til = self.fc_1(pred_embeddings_det[i])  # [1, 256]
            h_seg = pred_embeddings_seg[i]  # [1, 256]
            attention_output, _ = self.attention_module(query=h_det_til, key=h_seg, value=h_seg)
            # 断言 attention_output不是空向量
            assert not torch.isnan(attention_output).any(), "Attention output contains NaN"
            assert not torch.isinf(attention_output).any(), "Attention output contains Inf"
            h_seg_til = h_seg + attention_output

            # 将文本特征输入到SAM的prompt_encoder内获取稀疏和密集 embeddings
            sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings_seg[i].unsqueeze(1),
            )

            sparse_embeddings = sparse_embeddings.to(pred_embeddings_seg[i].dtype)

            # SAM的encoder: 使用一个transformer将image embedding和prompt embedding做双向的cross-attention
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i],
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings + h_seg_til,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # 生成预测的结果
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        # 计算损失
        output = model_output.logits
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight

        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask, pred_mask = gt_masks[batch_idx], pred_masks[batch_idx]
            assert gt_mask.shape[0] == pred_mask.shape[0], "gt_mask.shape: {}, pred_mask.shape: {}".format(gt_mask.shape, pred_mask.shape)
            mask_bce_loss += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            mask_dice_loss += dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            num_masks += gt_mask.shape[0]
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        det_loss = 0
        # 获取分类损失
        for i in range(len(pred_det)):
            device_tmp = pred_det[i][0].device
            gt_det_i = (
                torch.tensor([0], dtype=torch.float).to(device_tmp)
                if kwargs["sampled_classes_list"][i] == "real"
                else torch.tensor([1], dtype=torch.float).to(device_tmp)
            )
            loss_i = self.loss_fn(pred_det[i][0], gt_det_i)
            det_loss += loss_i

        loss = ce_loss + mask_loss + det_loss

        # 确保最终损失没有 NaN 或 Inf
        assert not torch.isnan(loss).any(), "Final loss contains NaN"
        assert not torch.isinf(loss).any(), "Final loss contains Inf"

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "det_loss": det_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        """
        模型执行评估
        """
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i].shape,
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
