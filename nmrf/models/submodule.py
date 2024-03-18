import torch
from torch import nn
import torch.nn.functional as F


class ConvexUpsampler(nn.Module):
    def __init__(self, guide_dim, factor):
        super().__init__()

        self.factor = factor

        self.conv1 = nn.Conv2d(1, 64, 7, padding=3)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(guide_dim + 64, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, factor**2 * 9, 1, 1, 0)
        nn.init.constant_(self.conv5.weight, 0.)
        nn.init.constant_(self.conv5.bias, 0.)

    def forward(self, x, guide):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        f = F.relu(self.conv3(torch.cat((y, guide), dim=1)))
        f = F.relu(self.conv4(f))
        mask = 0.25 * self.conv5(f)

        bs, _, ht, wd = x.shape
        mask = mask.view(bs, 9, self.factor, self.factor, ht, wd)
        mask = F.softmax(mask, dim=1)

        x = F.unfold(x * self.factor, kernel_size=3, stride=1, padding=1)
        x = x.view(bs, 9, 1, 1, ht, wd)

        up_x = torch.sum(mask * x, dim=1, keepdim=False)  # [B, K, K, H, W]
        up_x = up_x.permute(0, 3, 1, 4, 2)  # [B, H, K, W, K]
        up_x = up_x.reshape(bs, self.factor*ht, self.factor*wd)

        return up_x


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class RefineNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.inplanes = 128
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 128, 3, 1, 1, dilation=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, dilation=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, padding=4, dilation=4), nn.ReLU(inplace=True))
        self.conv5 = self._make_layer(BasicBlock, 96, 1, 1, 1, 8)
        self.conv6 = self._make_layer(BasicBlock, 64, 1, 1, 1, 16)
        self.conv7 = self._make_layer(BasicBlock, 32, 1, 1, 1, 1)

        self.conv8 = nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)

        self.apply(self._init_weights)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, disp):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)

        disp = disp + conv8

        return F.relu(disp)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, planes, 3, stride, padding=dilation if dilation > 1 else pad, dilation=dilation), nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, padding=dilation if dilation > 1 else pad, dilation=dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


def softmax_with_extra_logit(x, dim=-1):
    """Softmax function with an additional virtual logit equal to zero..

    For compatibility with some previously trained models.

    This is equivalent to adding one to the denominator.
    In the context of attention, it allows you to attend to nothing.

    Args:
        x: input to softmax
        axis: the axis or axes along which the softmax should be computed. Either an
            integer or a tuple of integers.

    Returns:
        A tensor with the same shape as x.
    """
    m = torch.max(x, dim, keepdim=True)[0]
    m = torch.clamp(m.detach(), min=0)
    unnormalized = torch.exp(x - m)
    # after shift, extra logit is -m. Add exp(-m) to denominator
    denom = torch.sum(unnormalized, dim, keepdim=True) + torch.exp(-m)
    return unnormalized / denom