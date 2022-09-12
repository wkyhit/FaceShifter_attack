import os
import sys
import torch
sys.path.append('./face_modules/')
import torchvision.transforms as transforms
import torch.nn.functional as F
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from network.AEI_Net import *
from face_modules.mtcnn import *
import cv2
import PIL.Image as Image
import numpy as np
import configparser

import target_attack

config=configparser.ConfigParser()
config.read("config.txt")
Xs_imgs_path = config.get("image_inference","source_image_path")
Xt_imgs_path = config.get("image_inference","target_image_path")
save_path = config.get("image_inference","result_image_save_path")
adv_save_path = config.get("image_inference","adv_result_image_save_path")
g_weights_path = config.get("pretrained_weights","g_latest_path")
arcface_weights_path = config.get("pretrained_weights","arcface_weights_path")

detector = MTCNN()
device = torch.device('cuda')
G = AEI_Net(c_id=512)
G.eval()
# G.load_state_dict(torch.load('./saved_models/G_latest.pth', map_location=torch.device('cpu')))
G.load_state_dict(torch.load(g_weights_path, map_location=torch.device('cpu')))
G = G.cuda()

arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
# arcface.load_state_dict(torch.load('./face_modules/model_ir_se50.pth', map_location=device), strict=False)
arcface.load_state_dict(torch.load(arcface_weights_path, map_location=device), strict=False)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

source_list = []
target_list = []

for root, dirs, files in os.walk(Xs_imgs_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            source_list.append(os.path.join(root, file))

for root, dirs, files in os.walk(Xt_imgs_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            target_list.append(os.path.join(root, file))

source_list.sort()
target_list.sort()

idx = 1
for i in range(len(source_list)):


    Xs = cv2.imread(source_list[i])
    Xs = cv2.resize(Xs,(256,256))
    Xs = test_transform(Xs)
    Xs = Xs.unsqueeze(0).cuda()

    with torch.no_grad():
        embeds = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))
        # embeds = arcface(F.interpolate(Xs, (112, 112), mode='bilinear', align_corners=True))

    for j in range(len(target_list)):
        Xt_raw = cv2.imread(target_list[j])
        try:
            Xt, trans_inv = detector.align(Image.fromarray(Xt_raw[:, :, ::-1]), crop_size=(256, 256), return_trans_inv=True)
        except Exception as e:
            print('the target image is wrong, please change the image')
        
        Xt_raw = Xt_raw.astype(np.float)/255.0
        Xt = test_transform(Xt)
        Xt = Xt.unsqueeze(0).cuda()

        mask = np.zeros([256, 256], dtype=np.float)
        for i in range(256):
            for j in range(256):
                dist = np.sqrt((i-128)**2 + (j-128)**2)/128
                dist = np.minimum(dist, 1)
                mask[i, j] = 1-dist
        mask = cv2.dilate(mask, None, iterations=20)

        # with torch.no_grad():
        
        #********original result********
        Yt, _ = G(Xt, embeds) 

        """
        result_img attack
        """
        # Initialize Metrics
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        y = Yt.cpu().detach()
        
        attack = target_attack.IFGSMAttack(model=G, arcface=arcface,device=device)
        #传入img_id作为原始X, y作为目标Y，返回攻击后的adv_img_id
        Xs_adv,perturb = attack.perturb(Xs.clone().detach_(), y,Xt.clone().detach_())

        #*********adversarial result********
        adv_Yt, _ = G(Xs_adv, embeds)

        # save the adversarial result
        adv_Yt = adv_Yt.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5
        adv_Yt = adv_Yt[:, :, ::-1]
        adv_Yt_trans_inv = cv2.warpAffine(adv_Yt, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
        mask_ = cv2.warpAffine(mask,trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
        mask_ = np.expand_dims(mask_, 2)
        adv_Yt_trans_inv = mask_*adv_Yt_trans_inv + (1-mask_)*Xt_raw
        cv2.imwrite("{}/{}.jpg".format(adv_save_path,idx),adv_Yt_trans_inv*255)

        #save the original result
        Yt = Yt.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5
        Yt = Yt[:, :, ::-1]
        Yt_trans_inv = cv2.warpAffine(Yt, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
        mask_ = cv2.warpAffine(mask,trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
        mask_ = np.expand_dims(mask_, 2)
        Yt_trans_inv = mask_*Yt_trans_inv + (1-mask_)*Xt_raw
        # Yt_trans_inv = cv2.resize(Yt_trans_inv,(512,512))
        cv2.imwrite("{}/{}.jpg".format(save_path,idx),Yt_trans_inv*255)
        # cv2.imshow('image',Yt)
        # cv2.imwrite(save_path,Yt*255)
        # cv2.waitKey(0)





        
        idx += 1