import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_conv1d():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)

    # 定义输入张量 [batch_size=1, channels=2, seq_length=256]
    h_det_til =  torch.load('h_det_til.pth')
    h_seg = torch.load('h_seg.pth')

    # 归一化输入
    h_det_til = (h_det_til - h_det_til.mean()) / (h_det_til.std() + 1e-8)
    h_seg = (h_seg - h_seg.mean()) / (h_seg.std() + 1e-8)

    # 检查输入是否有 NaN 或 Inf
    assert not torch.isnan(h_det_til).any(), "h_det_til contains NaN"
    assert not torch.isinf(h_det_til).any(), "h_det_til contains Inf"
    assert not torch.isnan(h_seg).any(), "h_seg contains NaN"
    assert not torch.isinf(h_seg).any(), "h_seg contains Inf"

    # 构建 conv_input
    conv_input = torch.stack([h_det_til, h_seg], dim=1)  # [1, 2, 256]

    # 检查 conv_input
    assert not torch.isnan(conv_input).any(), "conv_input contains NaN"
    assert not torch.isinf(conv_input).any(), "conv_input contains Inf"

    # 定义单层卷积模块
    conv_module = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=256, kernel_size=3, padding=1), nn.ReLU())

    conv_module.to(device)
    conv_input = conv_input.to(device)
    # 使用固定的小权重进行初始化
    with torch.no_grad():
        for layer in conv_module:
            if isinstance(layer, nn.Conv1d):
                nn.init.constant_(layer.weight, 0.01)  # 固定权重为0.01
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)  # 固定偏置为0.0

    # 检查卷积模块的权重和偏置
    for name, param in conv_module.named_parameters():
        assert not torch.isnan(param).any(), f"{name} contains NaN after initialization"
        assert not torch.isinf(param).any(), f"{name} contains Inf after initialization"

    # 执行卷积操作
    conv_output = conv_module(conv_input)

    # 检查 conv_output 是否有 NaN 或 Inf
    if torch.isnan(conv_output).any():
        print("conv_output contains NaN")
    else:
        print("conv_output does not contain NaN")

    if torch.isinf(conv_output).any():
        print("conv_output contains Inf")
    else:
        print("conv_output does not contain Inf")

    # 打印 conv_output 的统计信息
    print(
        f"conv_output - mean: {conv_output.mean().item():.6f}, std: {conv_output.std().item():.6f}, min: {conv_output.min().item():.6f}, max: {conv_output.max().item():.6f}"
    )


if __name__ == "__main__":
    test_conv1d()
