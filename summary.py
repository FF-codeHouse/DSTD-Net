# --------------------------------------------#
#   该部分代码用于查看当前多帧网络结构与计算量
# --------------------------------------------#
import torch
from thop import clever_format, profile
from torchinfo import summary

from nets.yolo import YoloBody


if __name__ == "__main__":
    input_shape = [640, 640]
    num_classes = 1
    num_frame = 5
    phi = "s"

    # 需要使用 device 来指定网络在 GPU 还是 CPU 运行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloBody(
        num_classes=num_classes,
        phi=phi,
        num_frame=num_frame,
    ).to(device).eval()

    # 当前模型输入格式：batch, channel, frame, height, width
    dummy_input = torch.randn(
        1,
        3,
        num_frame,
        input_shape[0],
        input_shape[1],
        device=device,
    )

    # 查看网络结构。depth 可调大或调小，以控制显示的层级。
    summary(
        model,
        input_data=dummy_input,
        depth=4,
        col_names=("input_size", "output_size", "num_params", "trainable"),
        device=device.type,
    )

    # THOP 返回的第一个值实际表示 MACs。
    macs, forward_params = profile(
        model,
        inputs=(dummy_input,),
        verbose=False,
    )

    # --------------------------------------------------------#
    #   一次 MAC 包含一次乘法和一次加法。
    #   有些论文使用 MACs，即不乘 2；
    #   有些论文将乘法、加法分别计数，此时 FLOPs = MACs * 2。
    # --------------------------------------------------------#
    flops = macs * 2

    # THOP 的参数量是本次前向中参与计算的参数量。
    # model.parameters() 统计模型中注册的全部参数。
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    macs, flops, forward_params, total_params, trainable_params = clever_format(
        [macs, flops, forward_params, total_params, trainable_params],
        "%.3f",
    )

    print("\nModel complexity")
    print("-" * 50)
    print("Input shape:            (1, 3, %d, %d, %d)" % (
        num_frame, input_shape[0], input_shape[1]
    ))
    print("Total MACs:             %s" % macs)
    print("Total FLOPs (MACs x 2): %s" % flops)
    print("Forward params (THOP):  %s" % forward_params)
    print("Registered params:      %s" % total_params)
    print("Trainable params:       %s" % trainable_params)
