
#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .resnet import Resnet18
from torch.nn import BatchNorm2d
from my_utils.utils import *

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.Upsample(scale_factor=up_factor,
                mode='bilinear', align_corners=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        #  self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        #  atten = self.sigmoid_atten(atten)
        atten = atten.sigmoid()
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()

        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.up32 = nn.Upsample(scale_factor=2.)
        self.up16 = nn.Upsample(scale_factor=2.)

        self.init_weight() 

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)

        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = self.up32(feat32_sum)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up # x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        ## use conv-bn instead of 2 layer mlp, so that tensorrt 7.2.3.4 can work for fp16
        self.conv = nn.Conv2d(out_chan,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)

        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv(atten)
        atten = self.bn(atten)

        atten = atten.sigmoid()
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params







class BiSeNetV1_encoder(nn.Module):
    def __init__(self):
        super(BiSeNetV1_encoder, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.init_weight()
    def forward(self,x):
        H, W = x.size()[2:]
        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x)
        return [feat_sp,feat_cp8]


    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params



class BiSeNetV1_decoder(nn.Module):
    def __init__(self):
        super(BiSeNetV1_decoder, self).__init__()
        self.ffm = FeatureFusionModule(256, 256)
        
    def forward(self,*x):
        feat_fuse = self.ffm(x[0],x[1])

        return feat_fuse


class BiSeNetV1_header(nn.Module):
    def __init__(self,n_classes = 1):
        super(BiSeNetV1_header, self).__init__()
        self.conv_out = BiSeNetOutput(256, 256, n_classes, up_factor=8)

    def forward(self, x):
        feat_out = self.conv_out(x)
        feat_out = F.sigmoid(feat_out)
        return feat_out
    


class BiSeNetV1(nn.Module):

    def __init__(self, n_classes, aux_mode='train', *args, **kwargs):
        super(BiSeNetV1, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes, up_factor=8)
        self.aux_mode = aux_mode
        if self.aux_mode == 'train':
            self.conv_out16 = BiSeNetOutput(128, 64, n_classes, up_factor=8)
            self.conv_out32 = BiSeNetOutput(128, 64, n_classes, up_factor=16)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        if self.aux_mode == 'train':
            feat_out16 = self.conv_out16(feat_cp8)
            feat_out32 = self.conv_out32(feat_cp16)
            return feat_out, feat_out16, feat_out32
        elif self.aux_mode == 'eval':
            return feat_out,
        elif self.aux_mode == 'pred':
            feat_out = feat_out.argmax(dim=1)
            return feat_out
        else:
            raise NotImplementedError

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


class CBAM(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel // reduction, out_features=in_channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):

        maxout = self.max_pool(x)
        maxout = self.mlp(maxout.view(maxout.size(0), -1))
        avgout = self.avg_pool(x)
        avgout = self.mlp(avgout.view(avgout.size(0), -1))
        channel_out = self.sigmoid(maxout + avgout)
        channel_out = channel_out.view(x.size(0), x.size(1), 1, 1)
        channel_out = channel_out * x
        max_out, _ = torch.max(channel_out, dim=1, keepdim=True)
        mean_out = torch.mean(channel_out, dim=1, keepdim=True)
        out = torch.cat((max_out, mean_out), dim=1)
        out = self.sigmoid(self.conv(out))
        out = out * channel_out
        return out
    
    
class selfattention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  
        attn_matrix = self.softmax(attn_matrix) 
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1)) 
        out = out.view(*input.shape)

        return self.gamma * out + input




class Enhance_module(nn.Module):
    def __init__(self,scale, channels = 256):
        super(Enhance_module, self).__init__()
        self.attention = selfattention(channels )
        self.scale = scale

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    def _threshold(self, x, threshold=None):
        if threshold is not None:
            return (x > threshold).type(x.dtype)
        else:
            return x
    def forward(self, x,*y):
        pred_mask = x
        # pr_mask = np.argmax(pr_mask, axis=0)
        # pred_mask = pred_mask.cpu().detach().numpy().squeeze(1)
        # gray_mask = np.uint8(pred_mask * 255)
        # threshold_mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY)[1]


        for i in range(len(pred_mask)):
            pred_mask_single = pred_mask[i].cpu().detach().numpy().squeeze()
            
            pred_mask_single = np.argmax(pred_mask_single, axis=0).astype(np.uint8)
           
            
            pred_mask_single =  pred_mask_single.squeeze()
            pred_mask_copy = pred_mask_single
            gray_mask_single = np.uint8(pred_mask_single * 255)
            threshold_mask = cv2.threshold(gray_mask_single, 127, 255, cv2.THRESH_BINARY)[1]

            countours = find_contour(threshold_mask)

            threshold_mask_copy = np.zeros_like(gray_mask_single, dtype=np.uint8)
            areas = []
            # min border box 
            if len(countours) > 0:
                for idx in range(len(countours)):
                    areas.append(cv2.contourArea(countours[idx]))
                max_id = areas.index(max(areas))
                max_rect = cv2.minAreaRect(countours[max_id])
                max_rect= np.int0(cv2.boxPoints(max_rect))
                x_left = np.min(max_rect[:,0])
                x_right = np.max(max_rect[:, 0])
                y_up = np.min(max_rect[:, 1])
                y_bottom = np.max(max_rect[:, 1])
                threshold_mask_copy[y_up: y_bottom, x_left: x_right] = 1
            
            else:
                y_up,y_bottom, x_left, x_right = 0,0,224,224

            threshold_mask_copy[y_up: y_bottom, x_left: x_right] = 1
            
            threshold_mask_now = np.expand_dims(threshold_mask_copy, 0)
            threshold_mask_now = np.expand_dims(threshold_mask_now, 0)
            pred_mask_now = np.expand_dims(pred_mask_copy, 0)
            pred_mask_now = np.expand_dims(pred_mask_now, 0)

            if i ==0 :
                threshold_mask_copy_arr =  threshold_mask_now
                pred_mask_copy_arr = pred_mask_now
            else:
                threshold_mask_copy_arr =  np.concatenate((threshold_mask_copy_arr,threshold_mask_now),axis=0)
                pred_mask_copy_arr = np.concatenate((pred_mask_copy_arr, pred_mask_now),axis=0)


        mask_box_tensor = torch.tensor(torch.from_numpy(threshold_mask_copy_arr), dtype=torch.float32, device="cuda:0",
                                        requires_grad=True)
        pred_mask_tensor = torch.tensor(torch.from_numpy(pred_mask_copy_arr), dtype=torch.float32, device="cuda:0",
                                       requires_grad=True)
       

        #
        pred_mask_tensor_resize = F.interpolate(pred_mask_tensor, scale_factor=self.scale, mode="bilinear",align_corners=True)
        mask_box_tensor_resize = F.interpolate(pred_mask_tensor, scale_factor=self.scale, mode="bilinear", align_corners=True)


        # back_pred_mask_tensor_resize = mask_box_tensor_resize - pred_mask_tensor_resize

        back_box_tensor_resize= 1- mask_box_tensor_resize
      
        result_pred = y[0]*mask_box_tensor_resize
        result_back_pred = y[0]* back_box_tensor_resize

        result_pred1 = y[1]*mask_box_tensor_resize
        result_back_pred1= y[1]* back_box_tensor_resize

        result = torch.cat([result_pred,result_back_pred,result_pred1,result_back_pred1],dim=1)
        result = self.attention(result)

        
        return result