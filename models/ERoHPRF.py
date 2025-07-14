import torch
import torch.nn as nn


def getConv2D(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=None, groups=1, bias=True):
    if padding is None:
        padding = 0
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=bias)


def getBN(channels, eps=1e-5, momentum=0.01, affine=True):
    return nn.BatchNorm2d(num_features=channels, eps=eps, momentum=momentum, affine=affine)


def mergeBN(convLayer, BNLayer):
    std = (BNLayer.running_var + BNLayer.eps).sqrt()
    t = (BNLayer.weight / std).reshape(-1, 1, 1, 1)
    return convLayer.weight * t, BNLayer.bias - BNLayer.running_mean * BNLayer.weight / std


def kernelFuse(target, sec):
    sec_h = sec.size(2)
    sec_w = sec.size(3)
    target_h = target.size(2)
    target_w = target.size(3)
    target[:, :, target_h // 2 - sec_h // 2: target_h // 2 - sec_h // 2 + sec_h,
    target_w // 2 - sec_w // 2: target_w // 2 - sec_w // 2 + sec_w] += sec


class AsymmConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, depthwise=True):
        super(AsymmConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.depthwise = depthwise

        if depthwise:
            self.convVe1 = getConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                     stride=stride, padding=[padding, 0], groups=in_channels, bias=False)
            self.bnVe1 = getBN(out_channels)
            self.convHo1 = getConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                     stride=stride, padding=[0, padding], groups=in_channels, bias=False)
            self.bnHo1 = getBN(out_channels)
            # ----------------------------------------------------------------
            self.convVek = getConv2D(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(kernel_size, kernel_size - 2),
                                     stride=stride, padding=[padding, padding - 1], groups=in_channels, bias=False)
            self.bnVek = getBN(out_channels)
            self.convHok = getConv2D(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(kernel_size - 2, kernel_size),
                                     stride=stride, padding=[padding - 1, padding], groups=in_channels, bias=False)
            self.bnHok = getBN(out_channels)
            # ----------------------------------------------------------------
            self.convSq = getConv2D(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=(kernel_size, kernel_size), stride=stride, padding=[padding, padding],
                                    groups=in_channels, bias=False)
            self.bnSq = getBN(out_channels)
        else:
            self.convVe1 = getConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                     stride=stride, padding=[padding, 0], bias=False)
            self.bnVe1 = getBN(out_channels)
            self.convHo1 = getConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                     stride=stride, padding=[0, padding], bias=False)
            self.bnHo1 = getBN(out_channels)
            # ----------------------------------------------------------------
            self.convVek = getConv2D(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(kernel_size, kernel_size - 2),
                                     stride=stride, padding=[padding, padding - 1], bias=False)
            self.bnVek = getBN(out_channels)
            self.convHok = getConv2D(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(kernel_size - 2, kernel_size),
                                     stride=stride, padding=[padding - 1, padding], bias=False)
            self.bnHok = getBN(out_channels)
            # ----------------------------------------------------------------
            self.convSq = getConv2D(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=(kernel_size, kernel_size), stride=stride, padding=[padding, padding],
                                    bias=False)
            self.bnSq = getBN(out_channels)

        self.mergedFlag = False
        self.mergedConv = None

    def forward(self, x):
        if not self.mergedFlag:
            output = self.bnVe1(self.convVe1(x)) + self.bnVek(self.convVek(x)) + self.bnHo1(
                self.convHo1(x)) + self.bnHok(self.convHok(x)) + self.bnSq(self.convSq(x))
        else:
            output = self.mergedConv(x)

        return output

    def mergeAsyKernels(self):
        convVe1_weight, convVe1_bias = mergeBN(self.convVe1, self.bnVe1)
        convHo1_weight, convHo1_bias = mergeBN(self.convHo1, self.bnHo1)
        convSq_weight, convSq_bias = mergeBN(self.convSq, self.bnSq)

        kernelFuse(convSq_weight, convVe1_weight)
        kernelFuse(convSq_weight, convHo1_weight)
        convSq_bias = convSq_bias + convVe1_bias + convHo1_bias

        convVek_weight, convVek_bias = mergeBN(self.convVek, self.bnVek)
        convHok_weight, convHok_bias = mergeBN(self.convHok, self.bnHok)
        kernelFuse(convSq_weight, convVek_weight)
        kernelFuse(convSq_weight, convHok_weight)
        convSq_bias = convSq_bias + convVek_bias + convHok_bias

        if self.depthwise:
            self.mergedConv = getConv2D(in_channels=self.in_channels, out_channels=self.out_channels,
                                        kernel_size=(self.kernel_size, self.kernel_size), stride=self.stride,
                                        padding=[self.padding, self.padding], groups=self.in_channels, bias=True)
        else:
            self.mergedConv = getConv2D(in_channels=self.in_channels, out_channels=self.out_channels,
                                        kernel_size=(self.kernel_size, self.kernel_size), stride=self.stride,
                                        padding=[self.padding, self.padding], bias=True)

        self.mergedConv.weight.data = convSq_weight
        self.mergedConv.bias.data = convSq_bias
        self.mergedFlag = True

        self.__delattr__('convVe1')
        self.__delattr__('bnVe1')
        self.__delattr__('convHo1')
        self.__delattr__('bnHo1')
        self.__delattr__('convVek')
        self.__delattr__('bnVek')
        self.__delattr__('convHok')
        self.__delattr__('bnHok')
        self.__delattr__('convSq')
        self.__delattr__('bnSq')


class MultiConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride, depthwise=True):
        super(MultiConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.stride = stride
        self.depthwise = depthwise

        self.conv_list = nn.ModuleList()

        for kernel in kernel_sizes:
            self.conv_list.append(AsymmConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                depthwise=depthwise
            ))

        self.mergedFlag = False
        self.mergedConv = None

    def forward(self, x):
        y = None
        if not self.mergedFlag:
            for conv in self.conv_list:
                if y is None:
                    y = conv(x)
                else:
                    y = y + conv(x)
        else:
            y = self.mergedConv(x)
        return y

    def mergeMulKernels(self):
        self.conv_list[-1].mergeAsyKernels()
        conv_weight = self.conv_list[-1].mergedConv.weight.data
        conv_bias = self.conv_list[-1].mergedConv.bias.data
        for index in range(len(self.conv_list) - 1):
            self.conv_list[index].mergeAsyKernels()
            kernelFuse(conv_weight, self.conv_list[index].mergedConv.weight)
            conv_bias += self.conv_list[index].mergedConv.bias

        if self.depthwise:
            self.mergedConv = getConv2D(in_channels=self.in_channels, out_channels=self.out_channels,
                                        kernel_size=(self.kernel_sizes[-1], self.kernel_sizes[-1]), stride=self.stride,
                                        padding=[self.kernel_sizes[-1] // 2, self.kernel_sizes[-1] // 2],
                                        groups=self.in_channels, bias=True)
        else:
            self.mergedConv = getConv2D(in_channels=self.in_channels, out_channels=self.out_channels,
                                        kernel_size=(self.kernel_sizes[-1], self.kernel_sizes[-1]), stride=self.stride,
                                        padding=[self.kernel_sizes[-1] // 2, self.kernel_sizes[-1] // 2],
                                        bias=True)

        self.mergedConv.weight.data = conv_weight
        self.mergedConv.bias.data = conv_bias
        self.mergedFlag = True

        self.__delattr__('conv_list')


if __name__ == '__main__':
    temp = torch.randn((1, 1, 7, 7))
    net = MultiConvBlock(in_channels=1, out_channels=2, kernel_sizes=[3, 5, 7], stride=1)
    net.eval()
    a = net(temp)
    print(a)
    print(net)
    print("=====================")
    # print(net)
    net.mergeMulKernels()
    net.eval()
    b = net(temp)
    print(b)
    print(a - b)
    # print(net)
