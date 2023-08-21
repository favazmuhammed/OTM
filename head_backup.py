from torch.nn import Module, Parameter
import math
import torch
import torch.nn as nn
from torch import Tensor

def build_head(head_type,
               embedding_size,
               class_num,
               m,
               z_alpha,
               t_alpha,
               h,
               s,
               delta,
               include_pose,
               include_quality,
               ):
    if head_type == 'adaface':
        head = AdaFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       h=h,
                       s=s,
                       t_alpha=t_alpha,
                       )

    elif head_type == 'arcface':
        head = ArcFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       s=s,
                       )
    elif head_type == 'cosface':
        head = CosFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       s=s,
                       )

    elif head_type == 'otm':
        head = OTMFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       h=h,
                       s=s,
                       t_alpha=t_alpha,
                       include_pose=include_pose,
                       include_quality=include_quality
                       )

    else:
        raise ValueError('not a correct head type', head_type)
    return head


def l2_norm(input, axis=1):
    norm = torch.norm(input,2, axis,True)
    output = torch.div(input, norm)
    return output

# def l2_norm(x, dim=1):
#     norm = torch.norm(x, 2, dim, keepdim=True)
#     output = torch.div(x, norm)
#     return output


class AdaFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('\nAdaFace with the following property')
        print('m', self.m)
        print('h', self.h)
        print('s', self.s)
        print('t_alpha', self.t_alpha)

    def forward(self, embbedings, norms, label, z_pose=None):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m

class CosFace(nn.Module):

    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.4):
        super(CosFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.4
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.eps = 1e-4

        print('init CosFace with ')
        print('m', self.m)
        print('s', self.s)

    def forward(self, embbedings, norms, label, z_pose=None):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        cosine = cosine - m_hot
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m


class ArcFace(Module):

    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.5):
        super(ArcFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.eps = 1e-4

        print('\nArcFace with the following property')
        print('m', self.m)
        print('s', self.s)

    def forward(self, embbedings, norms, label, z_pose=None):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        theta = cosine.acos()

        theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi-self.eps)
        cosine_m = theta_m.cos()
        scaled_cosine_m = cosine_m * self.s

        return scaled_cosine_m

class OTMFace(Module):
    def __init__(self, embedding_size=512, classnum=70722, m=0.4, h=0.333, s=64, t_alpha=0.01, include_pose=False, include_quality=False) -> None:
        super().__init__()
        self.m = m
        self.h = h
        self.s = s
        self.t_alpha = t_alpha
        self.eps = 1e-3
        self.include_pose = include_pose
        self.include_quality = include_quality

        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.register_buffer('batch_mean', torch.ones(1)*20)
        self.register_buffer('batch_std', torch.ones(1)*100)
        # self.norm_layer = nn.BatchNorm1d(1, eps=self.eps, momentum=self.t_alpha, affine=False)

        print('\nOTM with the following property')
        print('m', self.m)
        print('h', self.h)
        print('s', self.s)
        print('t_alpha', self.t_alpha)

    def forward(self, feats: Tensor, norms: Tensor, label: Tensor, z_pose:Tensor) -> Tensor:
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(feats, kernel_norm).clamp(-1+self.eps, 1-self.eps)
        safe_norms = torch.clip(norms, min=0.001, max=100)

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler_quality = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)    # 66% between -1, 1
        # margin_scaler = self.norm_layer(safe_norms)
        margin_scaler_quality = margin_scaler_quality * self.h                       # 68% between -0.333, 0.333 when h:0.333
        margin_scaler_quality = torch.clip(margin_scaler_quality, -1, 1)
        
        # print(margin_scaler_quality.shape)
        # eg. m=0.5, h:0.333
        # range
        #   (66% range)
        #   -1  -0.333  0.333   1   (margin_scaler)
        # -0.5  -0.166  0.166 0.5   (m * margin_scaler)
        
        # update the margin
        margin_scaler_pose = z_pose.reshape(-1,1)

        if self.include_quality and self.include_pose:
            # otm
            margin_scaler = torch.min(margin_scaler_quality, margin_scaler_pose)
        elif self.include_pose:
            # otm-A
            margin_scaler = margin_scaler_pose 

        elif self.include_quality:
            # otm - B
            margin_scaler = margin_scaler_quality
        else: 
            # no adaption on margin
            margin_scaler = torch.ones_like(margin_scaler_pose)


        # g_angular
        m_arc = torch.zeros(label.shape[0], cosine.shape[1], device=cosine.device).scatter_(1, label.view(-1, 1), 1.0)
        g_angular = self.m * margin_scaler


        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, self.eps, math.pi - self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.shape[0], cosine.shape[1], device=cosine.device).scatter_(1, label.view(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m
