
import torch
import torch.nn.functional as F
from torch import nn
from backbone_resnet50 import resnet50
from  helper import initialize_weights
BatchNorm2d = nn.BatchNorm2d


class _PSPModule(nn.Module):
    "PSPNet的decoder部分，负责将backbone的输出进行池化，然后拼接"
    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        # 通过这个列表表达式将backboned输出进行平均池化
        # bin_sizes是池化的大小，在这里等于[1,2,3,6]
        # 每次取出这个列表中的一个值，将他传给make_stages函数，得到一个全局平均池化后的结果
        # 通过for循环将四种尺寸的全局平局池化的结果保存到列表中 传入nn.MoudleList 生成网络
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer)
                                      for b_s in bin_sizes])
        # 对四种尺寸的全局平均池化拼接后的结果调整通道数
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
           align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class PSPNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, pretrained=False, use_aux=True,   ):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        model = resnet50(pretrained)  # 导入resnet主干网络
        m_out_sz = model.fc.in_features # 2048

        self.use_aux = use_aux
        # 由于使用的加深的resnet50,所以这里取10层,即resnet50 layer前部分
        self.initial = nn.Sequential(*list(model.children())[:4])
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # 将decoder部分的输出再次降维到分类类别数
        self.master_branch = nn.Sequential(
            _PSPModule(m_out_sz, bin_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
        )
        # 由于PSPNet有辅助损失函数，这里需要搭建一个辅助分支
        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(m_out_sz // 2, m_out_sz // 4, kernel_size=3, padding=1, bias=False),
            norm_layer(m_out_sz // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
        )
        # 对主干网络和辅助分支进行权值初始化
        initialize_weights(self.master_branch, self.auxiliary_branch)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x = self.initial(x)
        #print('x shape ',x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x) # 将主干网layer3的结果作为辅助分支
        x = self.layer4(x_aux)

        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear',align_corners=True)
        output = output[:, :, :input_size[0], :input_size[1]]
        # 判断是否需要辅助分支,如果在训练模式下就使用，验证模式下不使用辅助分支
        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear',align_corners=True)
            aux = aux[:, :, :input_size[0], :input_size[1]]
            return [output, aux]
        return output



if __name__ == '__mian__':
    input = torch.randn((2,3,224,224))
    psp = PSPNet(num_classes=21)
    o = psp(input)
    print(o)




















