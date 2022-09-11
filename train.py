from network.AEI_Net import *
from network.MultiscaleDiscriminator import *
from utils.Dataset import FaceEmbed, With_Identity
from torch.utils.data import DataLoader
import torch.optim as optim
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch.nn.functional as F
import torch
import time
import torchvision
import cv2
from apex import amp
import configparser

config=configparser.ConfigParser()
config.read("config.txt")
dataset_path = config.get("train","dataset_path")

batch_size = 4
lr_G = 4e-4
lr_D = 4e-4
max_epoch = 2000
show_step = 10
save_epoch = 1
model_save_path = './saved_models/'
optim_level = 'O1'

fine_tune_with_identity = False

device = torch.device('cuda')
# torch.set_num_threads(12)

G = AEI_Net(c_id=512).to(device)
D = MultiscaleDiscriminator(input_nc=3, n_layers=5, norm_layer=torch.nn.InstanceNorm2d).to(device)
G.train()
D.train()

arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load('./face_modules/model_ir_se50.pth', map_location=device), strict=False)

opt_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0, 0.999), weight_decay=1e-4)
opt_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0, 0.999), weight_decay=1e-4)

G, opt_G = amp.initialize(G, opt_G, opt_level=optim_level)
D, opt_D = amp.initialize(D, opt_D, opt_level=optim_level)

try:
    G.load_state_dict(torch.load('./saved_models/G_latest.pth', map_location=torch.device('cpu')), strict=False)
    D.load_state_dict(torch.load('./saved_models/D_latest.pth', map_location=torch.device('cpu')), strict=False)
except Exception as e:
    print(e)


dataset = FaceEmbed([dataset_path], same_prob=0.5)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()


def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1-X)
    else:
        return torch.relu(X+1)


def get_numpy_image(X):
    X = X[:8]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]).numpy() * 0.5 + 0.5
    X = X.transpose([1,2,0])*255
    np.clip(X, 0, 255).astype(np.uint8)
    return X


def make_image(Xs, Xt, Y):
    Xs = get_numpy_image(Xs)
    Xt = get_numpy_image(Xt)
    Y = get_numpy_image(Y)
    return np.concatenate((Xs, Xt, Y), axis=0).transpose([2, 0, 1])


print(torch.backends.cudnn.benchmark)
#torch.backends.cudnn.benchmark = True
for epoch in range(0, max_epoch):
    # torch.cuda.empty_cache()
    for iteration, data in enumerate(dataloader):
        start_time = time.time()
        Xs, Xt, same_person = data
        Xs = Xs.to(device)
        Xt = Xt.to(device)
        # embed = embed.to(device)
        with torch.no_grad():
            embed = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
        same_person = same_person.to(device)
        #diff_person = (1 - same_person)
        diff_person = torch.ones_like(same_person)
        # test
        same_person = diff_person

        # train G
        opt_G.zero_grad()
        Y, Xt_attr = G(Xt, embed)

        Di = D(Y)#[-1][0]
        L_adv = 0

        for di in Di:
            L_adv += hinge_loss(di[0], True).mean(dim=[1, 2,3])
        L_adv = torch.sum(L_adv * diff_person) / (diff_person.sum() + 1e-4)

        Y_aligned = Y[:, :, 19:237, 19:237]
        forehead = Y_aligned[:, :, :50, :].detach()
        down = Y_aligned[:, :, 50:, :]
        Y_aligned = torch.cat((forehead, down), dim=2)
        ZY = arcface(F.interpolate(Y_aligned, [112, 112], mode='bilinear', align_corners=True))
        L_id =(1 - torch.cosine_similarity(embed, ZY, dim=1)).mean()

        Y_attr = G.get_attr(Y)
        L_attr = 0
        for i in range(len(Xt_attr)):
            L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(batch_size, -1), dim=1).mean()
        L_attr /= 2.0

        L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

        lossG = 1*L_adv + 10*L_attr + 20*L_id + 7*L_rec
        # lossG = 1*L_adv + 10*L_attr + 5*L_id + 10*L_rec
        with amp.scale_loss(lossG, opt_G) as scaled_loss:
            scaled_loss.backward()

        # lossG.backward()
        opt_G.step()

        # train D
        opt_D.zero_grad()
        # with torch.no_grad():
        #     Y, _ = G(Xt, embed)
        fake_D = D(Y.detach())
        loss_fake = 0
        for di in fake_D:
            loss_fake += torch.sum(hinge_loss(di[0], False).mean(dim=[1, 2,3]) * diff_person) / (diff_person.sum() + 1e-4)

        true_D = D(Xs)
        loss_true = 0
        for di in true_D:
            loss_true += torch.sum(hinge_loss(di[0], True).mean(dim=[1, 2,3]) * diff_person) / (diff_person.sum() + 1e-4)
        # true_score2 = D(Xt)[-1][0]

        lossD = 0.5*(loss_true.mean() + loss_fake.mean())

        with amp.scale_loss(lossD, opt_D) as scaled_loss:
            scaled_loss.backward()
        # lossD.backward()
        opt_D.step()
        batch_time = time.time() - start_time
        if iteration % show_step == 0:
            image = make_image(Xs, Xt, Y)
            cv2.imwrite('./latest.jpg', image.transpose([1,2,0])[:,:,::-1])
        print(f'epoch: {epoch}    {iteration} / {len(dataloader)}')
        print(f'lossD: {lossD.item()}    lossG: {lossG.item()} batch_time: {batch_time}s')
        print(f'L_adv: {L_adv.item()} L_id: {L_id.item()} L_attr: {L_attr.item()} L_rec: {L_rec.item()}')
        if iteration % 2000 == 0:
            torch.save(G.state_dict(), './saved_models/G_latest.pth')
            torch.save(D.state_dict(), './saved_models/D_latest.pth')


