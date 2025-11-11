import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.pvtv2_new import  pvt_v2_b2

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bn=True, relu=True):
        super().__init__()
        if bn:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=True)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MLF(nn.Module):
    def __init__(self, in_channels):
        super(MLF, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = lambda x, target: F.interpolate(
            x, size=target.shape[2:], mode='bilinear', align_corners=False
        )

        self.scale1_to2 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.scale1_to3 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.scale2_to3 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.scale1_to4 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.scale2_to4 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.scale3_to4 = BasicConv2d(in_channels, in_channels, 3, padding=1)

        self.fuse_scale1_to2 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.fuse_2scale = BasicConv2d(2 * in_channels, 2 * in_channels, 3, padding=1)
        self.fuse_3scale = BasicConv2d(3 * in_channels, 3 * in_channels, 3, padding=1)

        self.concat_fuse2 = BasicConv2d(2 * in_channels, 2 * in_channels, 3, padding=1)
        self.concat_fuse3 = BasicConv2d(3 * in_channels, 3 * in_channels, 3, padding=1)
        self.concat_fuse4 = BasicConv2d(4 * in_channels, 4 * in_channels, 3, padding=1)

        self.final_fuse = BasicConv2d(4 * in_channels, 4 * in_channels, 3, padding=1)
        self.channel_compress = nn.Conv2d(4 * in_channels, in_channels, 1)

    def forward(self, x1, x2, x3, x4):
        x1_fuse = x1
        x2_fuse = self.scale1_to2(self.upsample(x1, x2)) * x2
        x3_fuse = self.scale1_to3(self.upsample(x1, x3)) \
                  * self.scale2_to3(self.upsample(x2, x3)) * x3
        x4_fuse = self.scale1_to4(self.upsample(x1, x4)) \
                  * self.scale2_to4(self.upsample(x2, x4)) \
                  * self.scale3_to4(self.upsample(x3, x4)) * x4

        x2_concat = torch.cat([x2_fuse, self.fuse_scale1_to2(self.upsample(x1_fuse, x2_fuse))], dim=1)
        x2_concat = self.concat_fuse2(x2_concat)

        x3_concat = torch.cat([x3_fuse, self.fuse_2scale(self.upsample(x2_concat, x3_fuse))], dim=1)
        x3_concat = self.concat_fuse3(x3_concat)

        x4_concat = torch.cat([x4_fuse, self.fuse_3scale(self.upsample(x3_concat, x4_fuse))], dim=1)
        x4_concat = self.concat_fuse4(x4_concat)

        x = self.final_fuse(x4_concat)
        x = self.channel_compress(x)

        return x

class MSF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = lambda x, target: F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, 3, padding=1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            BasicConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv_cat = nn.Sequential(
            BasicConv2d(4 * out_channels, out_channels, 3, padding=1),
            BasicConv2d(out_channels, out_channels, 3, padding=1))
        self.conv_res = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 3, padding=1),
            BasicConv2d(out_channels, out_channels, 3, padding=1))
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.soft = torch.nn.Softmax(dim=1)
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x3 = self.upsample(x3, x)
        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_c = self.conv_res(x)
        x_a = self.avgpool(x_cat)
        x_out = x_c * self.soft(self.conv_cat(x_a)) + x_c
        return x_out
class SA(nn.Module):
    def __init__(self, in_dim):
        super(SA, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.key_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.scale = 1.0 / (in_dim ** 0.5)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.conv6 = BasicConv2d(in_dim, in_dim, kernel_size=3,  padding=1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, W * H)
        x_w = torch.bmm(proj_query, proj_key) * self.scale

        out_max = torch.max(x_w, dim=-1)[0]
        out_avg = torch.mean(x_w, dim=-1)

        out_co = out_max + out_avg

        x_co = out_co.view(B, -1)
        x_co = F.softmax(x_co, dim=-1)
        x_co = x_co.view(B, 1, H, W)
        out = x * x_co
        out = self.conv6(out)

        return out
class LC(nn.Module):
    def __init__(self, channel,ratio=4):
        super(LC, self).__init__()
        self.conv_xi = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channel // ratio, out_features=channel, bias=False),
        )
        self.sa = SA(channel)
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x, x_i, diff):
        #aggregation
        x_i = self.conv_xi(x_i)
        x_l_u = x + x_i
        b, c, _, _ = x_l_u.size()
        x_l_uv = self.avg_pooling(x_l_u) + self.max_pooling(x_l_u)
        x_l_uv = x_l_uv.view(b, c)
        x_v = self.fc_layers(x_l_uv)
        x_v = torch.sigmoid(x_v)
        x_v = x_v.view(b, c, 1, 1)
        x_l_u_out = x_l_u * x_v.expand_as(x_l_u) + x_l_u
        #calibration
        x_b = x_l_u_out * diff + x_l_u_out
        out = self.alpha * self.sa(x_b) + x_b
        return out

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.backbone1 = pvt_v2_b2()
        path = '../pre_trained/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone1.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone1.load_state_dict(model_dict)
        self.upsample = lambda x, target: F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)

        self.predict_layer_1 = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=True)
        )
        self.predtrans2 = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)
        )
        self.predtrans3 = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)
        )
        self.predtrans4 = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)
        )
        self.pred_coarse = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)
        )
        self.MLF = MLF(64)
        self.MSF_4 = MSF(512, 64)
        self.MSF_3 = MSF(320, 64)
        self.MSF_2 = MSF(128, 64)
        self.MSF_1 = MSF(64, 64)
        self.LC_4 = LC(64)
        self.LC_3 = LC(64)
        self.LC_2 = LC(64)
        self.LC_1 = LC(64)
        self.kernel = np.ones((5, 5), np.uint8)
    def forward(self, image):
        x_list = self.backbone1(image)
        x4 = self.MSF_4(x_list[3])
        x3 = self.MSF_3(x_list[2])
        x2 = self.MSF_2(x_list[1])
        x1 = self.MSF_1(x_list[0])

        x_c_p = self.MLF(x4, x3, x2, x1)
        D_c = self.pred_coarse(x_c_p)
        U_4 = 1 - 2*torch.abs(torch.sigmoid(D_c) - 0.5)
        U_4 = self.upsample(U_4, x4)
        x_c_p = self.upsample(x_c_p, x4)
        x4_e = self.LC_4(x4, x_c_p, U_4)
        D_4 = self.predtrans4(x4_e)

        U_3 = torch.abs(torch.sigmoid(self.upsample(D_4, x3)) - torch.sigmoid(self.upsample(D_c, x3)))
        x4_s = self.upsample(x4_e, x3)
        x3_e = self.LC_3(x3, x4_s, U_3)
        D_3 = self.predtrans3(x3_e)

        U_2 = torch.abs(torch.sigmoid(self.upsample(D_3, x2)) - torch.sigmoid(self.upsample(D_4, x2)))
        x3_s = self.upsample(x3_e, x2)
        x2_e = self.LC_2(x2, x3_s, U_2)
        D_2 = self.predtrans2(x2_e)

        U_1 = torch.abs(torch.sigmoid(self.upsample(D_2, x1)) - torch.sigmoid(self.upsample(D_3, x1)))
        x2_s = self.upsample(x2_e, x1)
        x1_e = self.LC_1(x1, x2_s, U_1)
        D_1 = self.predict_layer_1(x1_e)
        return D_1, D_2, D_3, D_4, D_c

if __name__ == '__main__':
    from thop import profile
    model = MyNet().cuda()
    x = torch.randn(1, 3, 352, 352).cuda()
    flops, params = profile(model, inputs=(x,))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))