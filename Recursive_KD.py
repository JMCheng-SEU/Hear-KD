import torch
from torch import nn
import torch.nn.functional as F


class InforComu_Ht(nn.Module):
    def __init__(self, mid_channel, shape):
        super(InforComu_Ht, self).__init__()
        self.comu_conv_ht = nn.Conv2d(1, mid_channel * 2, kernel_size=(1, 1))
        self.comu_linear_ht = nn.Linear(257, shape)

    def forward(self, src, tgt):
        outputs = torch.tanh(self.comu_conv_ht(self.comu_linear_ht(tgt))) * src
        return outputs


class HL_ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, shape, fuse):
        super(HL_ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.InstanceNorm2d(mid_channel, affine=True),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel,
                out_channel,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=False,
            ),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(),
        )
        self.Info_com = InforComu_Ht(mid_channel, shape)

        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None

    def forward(self, ht, x, y=None):
        n, _, h, w = x.shape
        x = self.conv1(x)

        if self.att_conv is not None:
            z = torch.cat([x, y], dim=1)
            z = self.Info_com(z, ht)
            z = self.att_conv(z)
            x = (
                x * z[:, 0].view(n, 1, h, w).contiguous()
                + y * z[:, 1].view(n, 1, h, w).contiguous()
            )

        y = self.conv2(x)
        return y, x


class HL_ABF_Res(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, shape, fuse):
        super(HL_ABF_Res, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.InstanceNorm2d(mid_channel, affine=True),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel,
                out_channel,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=False,
            ),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(),
        )
        self.Info_com = InforComu_Ht(mid_channel, shape)

        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None

    def forward(self, ht, x, y=None, shape=None):
        n, _, h, w = x.shape
        x = self.conv1(x)

        if self.att_conv is not None:
            y = F.interpolate(y, (h, shape), mode="nearest")
            z = torch.cat([x, y], dim=1)
            z = self.Info_com(z, ht)
            z = self.att_conv(z)
            x = (
                x * z[:, 0].view(n, 1, h, w).contiguous()
                + y * z[:, 1].view(n, 1, h, w).contiguous()
            )

        y = self.conv2(x)
        return y, x


class HL_RecursiveKD(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channel, shapes):
        super(HL_RecursiveKD, self).__init__()
        self.shapes = shapes
        reverse_shapes = shapes[::-1]

        abfs = nn.ModuleList()
        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                HL_ABF_Res(
                    in_channel,
                    mid_channel,
                    out_channels[idx],
                    reverse_shapes[idx],
                    idx < len(in_channels) - 1,
                )
            )

        self.abfs = abfs[::-1]

    def forward(self, student_features, ht):
        x = student_features[::-1]
        results = []
        out_features, res_features = self.abfs[0](ht, x[0])
        results.append(out_features)

        for features, abf, shape in zip(x[1:], self.abfs[1:], self.shapes[1:]):
            out_features, res_features = abf(ht, features, res_features, shape)
            results.insert(0, out_features)

        return results


class Mid_Ht_Intra_Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channel, shapes, detach=False):
        super(Mid_Ht_Intra_Fusion, self).__init__()
        reverse_shapes = shapes[::-1]
        self.detach = detach

        abfs = nn.ModuleList()
        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                HL_ABF(
                    in_channel,
                    mid_channel,
                    out_channels[idx],
                    reverse_shapes[idx],
                    idx < len(in_channels) - 1,
                )
            )

        self.abfs = abfs[::-1]

    def forward(self, features_set, ht):
        if self.detach:
            for i in range(len(features_set)):
                features_set[i] = features_set[i].detach()

        x = features_set[::-1]
        out_features, res_features = self.abfs[0](ht, x[0])
        for features, abf in zip(x[1:], self.abfs[1:]):
            out_features, res_features = abf(ht, features, res_features)

        return out_features
