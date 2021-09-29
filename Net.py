import torch
from torch import nn

# from torchvision.models.resnet import

channels_layer1 = [3, 192, ]
channels_layer2 = [192, 256, ]
channels_layer3 = [256, 128, 256, 256, 512]
channels_layer4 = [512, 256] * 4 + [512, 512, 1024]
channels_layer5 = [1024, 512] * 2 + [1024] * 3
channels_layer6 = [1024] * 3

bbox_num = 2
bbox_size = 5
class_num = 20
S = 7


class yolo1Net(nn.Module):
    def __init__(self,
                 input_ch=448,
                 dilation: int = 1,
                 groups: int = 1):  # 默认为448，如果不是可以进行填充
        super(yolo1Net, self).__init__()
        self.dilation = dilation
        self.groups = groups
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = nn.Conv2d
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d
        self.dropout = nn.Dropout2d(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear

        layer = []

        layer.append(self.conv(
            in_channels=channels_layer1[0],
            out_channels=channels_layer1[0 + 1],
            kernel_size=7,
            stride=2,
            padding=3,
            dilation=self.dilation,
            groups=self.groups, )
        )
        # 归一化和激活层
        layer.append(self.batch_norm(channels_layer1[0 + 1]))
        layer.append(self.leaky_relu)

        layer.append(self.max_pooling)

        layer.append(self.conv(
            in_channels=channels_layer2[0],
            out_channels=channels_layer2[0 + 1],
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=self.dilation,
            groups=self.groups, )
        )
        # 归一化和激活层
        layer.append(self.batch_norm(channels_layer2[0 + 1]))
        layer.append(self.leaky_relu)

        layer.append(self.max_pooling)

        for i in range(0, len(channels_layer3) - 1, 2):
            layer.append(
                self.conv(
                    in_channels=channels_layer3[i],
                    out_channels=channels_layer3[i + 1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )

            layer.append(self.batch_norm(channels_layer3[i + 1]))

            layer.append(
                self.conv(
                    in_channels=channels_layer3[i + 1],
                    out_channels=channels_layer3[i + 2],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )
            layer.append(self.batch_norm(channels_layer3[i + 2]))
            layer.append(self.leaky_relu)

        layer.append(self.max_pooling)

        for i in range(0, len(channels_layer4) - 1, 2):
            layer.append(
                self.conv(
                    in_channels=channels_layer4[i],
                    out_channels=channels_layer4[i + 1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )

            layer.append(self.batch_norm(channels_layer4[i + 1]))
            layer.append(
                self.conv(
                    in_channels=channels_layer4[i + 1],
                    out_channels=channels_layer4[i + 2],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )
            layer.append(self.batch_norm(channels_layer4[i + 2]))
            layer.append(self.leaky_relu)

        layer.append(self.max_pooling)

        for i in range(0, len(channels_layer5) - 1, 2):
            if i != 4:
                layer.append(
                    self.conv(
                        in_channels=channels_layer5[i],
                        out_channels=channels_layer5[i + 1],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        dilation=self.dilation,
                        groups=self.groups,
                    )
                )
                layer.append(self.batch_norm(channels_layer5[i + 1]))

            # 请注意这里之所以可以直接写i+1和i+2是因为最后三个卷积的输入和输出都是1024！！！
            layer.append(
                self.conv(
                    in_channels=channels_layer5[i + 1],
                    out_channels=channels_layer5[i + 2],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )
            layer.append(self.batch_norm(channels_layer5[i + 2]))

            if i == 4:
                layer.append(
                    self.conv(
                        in_channels=channels_layer5[i + 1],
                        out_channels=channels_layer5[i + 2],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        dilation=self.dilation,
                        groups=self.groups,
                    )
                )
                layer.append(self.batch_norm(channels_layer5[i + 2]))
                break
            layer.append(self.leaky_relu)

        for i in range(len(channels_layer6) - 1):
            layer.append(
                self.conv(
                    in_channels=channels_layer6[i],
                    out_channels=channels_layer6[i + 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )
            layer.append(self.batch_norm(channels_layer6[i + 1]))
        layer.append(self.leaky_relu)

        self.connect_1 = self.linear(
            in_features=7 * 7 * 1024,
            out_features=4096,
        )
        self.connect_2 = self.linear(
            in_features=4096,
            out_features=7 * 7 * 30,
        )
        self.layer_list = nn.Sequential(*layer)

    def forward(self, inputs):
        x = self.layer_list(inputs)
        x = x.view(-1, 7 * 7 * 1024)
        x = self.connect_1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.connect_2(x)
        x = self.sigmoid(x)
        return x.view(-1, S, S, (class_num + bbox_num * bbox_size))


# if __name__ == '__main__':
#     model = yolo1Net()
#     test_tensor = torch.randn((10, 3, 448, 448))
#     print(model(test_tensor).shape)
