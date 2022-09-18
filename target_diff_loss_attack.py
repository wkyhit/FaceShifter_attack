import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure

import torch
import torch.nn as nn

class IFGSMAttack(object):
    def __init__(self, model=None, arcface=None,device=None,mask=None, epsilon=0.04, k=50, a=0.01):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
        mask: mask for the attack_area
        """
        self.model = model
        self.arcface = arcface
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.loss_fn2 = nn.L1Loss().to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to("cuda")
        self.fid = FrechetInceptionDistance(feature=64).to("cuda")
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure().to("cuda")
        self.device = device
        self.mask = mask

        # PGD(True) or I-FGSM(False)?
        self.rand = True

        #attack on specific channel?
        self.channel = False
    
    def _batch_multiply_tensor_by_vector(self,vector, batch_tensor):
        """Equivalent to the following
        for ii in range(len(vector)):
            batch_tensor.data[ii] *= vector[ii]
        return batch_tensor
        """
        return (
            batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()

    def batch_multiply(self,float_or_vector, tensor):
        if isinstance(float_or_vector, torch.Tensor):
            assert len(float_or_vector) == len(tensor)
            tensor = self._batch_multiply_tensor_by_vector(float_or_vector, tensor)
        elif isinstance(float_or_vector, float):
            tensor *= float_or_vector
        else:
            raise TypeError("Value has to be float or torch.Tensor")
        return tensor

    def perturb(self, X_nat, y,target_img):
        """
        Vanilla Attack.
        """
        origin_img_src = X_nat.clone().detach_()#保留原始的img_src
        origin_img_src = origin_img_src.to(self.device)
        y = y.to(self.device)
        target_img = target_img.to(self.device)


        if self.rand: # PGD attack
            X_nat = X_nat.to(self.device)
            random = torch.rand_like(origin_img_src).uniform_(-self.epsilon,self.epsilon).to(self.device)
            x_tmp = X_nat + random
            X_nat = x_tmp.clone().detach_()

        X_nat = X_nat.to(self.device)        
        # self.model.set_input(X_nat)#设置model的数据

        for i in range(self.k):
            X_nat.requires_grad = True 
            embeds = self.arcface(F.interpolate(X_nat[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))
            output,_ = self.model(target_img,embeds)

            # img_id_downsample = F.interpolate(X_nat, size=(112,112))
            # latend_id = self.model.netArc(img_id_downsample) #攻击对象：latend_id
            # latend_id = F.normalize(latend_id, p=2, dim=1)
            # output = self.model(None, target_img, latend_id, None, True)[0] #[3,224,224]
            

            self.model.zero_grad() #梯度清零?

            #对单通道的梯度mask
            if self.channel:
                channel_idx = 2 #通道2噪声不明显
                grad_channel_mask = torch.zeros_like(X_nat)
                grad_channel_mask[:,channel_idx,:,:] = 1
                grad_channel_mask = grad_channel_mask.to(self.device)

            if self.mask is not None:
                mask = self.mask.clone().detach_()
                mask = mask.to(self.device)

                loss = self.loss_fn(output*mask, y*mask) #损失函数
            else:

                # Minus in the loss means "towards" and plus means "away from"
                # use mse loss
                # loss = self.loss_fn(output, y)

                # loss = ((output - y)**2).sum() #self_defined loss
                # loss = loss.mean()

                #use l1 loss
                # loss = self.loss_fn2(output, y)

                # nullfying attack with mse loss
                # loss = -1*((output-origin_img_src)**2).sum()
                # loss = -1*((output-target_img)**2).sum()
                # loss = loss.mean()

                # use lpips loss: a low lpips loss means the two images are perceptual similar
                # loss = self.lpips(output, y)

                #use fid loss: a low fid loss means low quality
                # self.fid.update(output, real=True)
                # self.fid.update(y, real=False)
                # loss = self.fid.compute()

                #use ms_ssim loss: a hight ms_ssim loss means the two images are structure similar
                loss = -1*self.ms_ssim(output, y)

            loss.requires_grad_(True) #!!解决无grad bug
            loss.backward()
            grad = X_nat.grad.data

            # ******基于 L infinity*******
            if self.channel:
                grad = grad * grad_channel_mask
            img_src_adv = X_nat + self.a * torch.sign(grad)

            #*****基于L2 attack*******
            # batch_size = grad.size(0)
            # p = 2 # L2
            # #防止梯度为0的情况
            # samll_constant = 1e-6 
            # norm = grad.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)
            # norm = torch.max(norm, torch.ones_like(norm)*samll_constant)
            # grad = self.batch_multiply(1./norm, grad)
            # if self.channel:
            #     grad = grad * grad_channel_mask
            # img_src_adv = X_nat + grad

            # eta = torch.clamp(img_src_adv - origin_img_src, min=-self.epsilon, max=self.epsilon)#加入的噪声
            # 对batch 做 mean
            eta = torch.mean(torch.clamp(img_src_adv - origin_img_src, min=-self.epsilon, max=self.epsilon).detach_(),dim=0)#加入的噪声
            #注意tensor取值-1~1
            X_nat = torch.clamp(origin_img_src + eta, min=-1, max=1).detach_()#攻击后的img_src结果

            # Debug
            # X_adv, loss, grad, output_att, output_img = None, None, None, None, None
        #返回攻击后的img_src和noise
        # return X_nat, eta 
        return X_nat, X_nat-origin_img_src

