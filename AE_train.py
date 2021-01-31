import torch
import numpy as np
from torch.nn import functional as F
from torch import nn, optim, autograd
from torch.autograd import Variable
from ae_dataloader import HS
from torch.utils.data import DataLoader
import os
import scipy.io as sio
from CC import CC
from SAM import SAM
from AE_model import AE3D,AE2D

torch.manual_seed(22)
np.random.seed(22)

#Here is the Hyperparameter
device = torch.device('cuda')
epoches = 300
batchsz = 7
learning_rate_1 = 0.001
learning_rate_2 = 0.001
alpha_sam = 0.001

#Here is the path to load and save data
version = 'v1'
root = '/home/xd132/xzc/project1/data'
weights_path_1 = '/home/xd132/xzc/project1/weights/' + version + '/AE3D/data/ae1'
weights_path_2 = '/home/xd132/xzc/project1/weights/' + version + '/AE2D/data/ae1'
code_path = '/home/xd132/xzc/project1/code/data/' + version
concat_str = 'concat_1'

#Generate iterators that load data
ae_db = HS(root)
ae_loader = DataLoader(ae_db, batch_size=batchsz, shuffle=True,num_workers=4,drop_last=False)
ae_loader_test = DataLoader(ae_db, batch_size=1,num_workers=4,drop_last=False)


def main():
    ae3d = AE3D()
    ae2d = AE2D()
    ae3d.to(device)
    ae2d.to(device)

    if os.path.exists(weights_path_1 + '/ae3d_weights.pth'):
        print('start load last time ae3d weights')
        ae3d.load_state_dict(torch.load(weights_path_1 + '/ae3d_weights.pth'))

    if os.path.exists(weights_path_2 + '/ae2d_weights.pth'):
        print('start load last time ae2d weights')
        ae2d.load_state_dict(torch.load(weights_path_2 + '/ae2d_weights.pth'))

    print(ae3d)
    print(ae2d)

    optim_ae3d = optim.Adam(ae3d.parameters(), lr=learning_rate_1)
    optim_ae2d = optim.Adam(ae2d.parameters(), lr=learning_rate_2)

    best_ae3d_loss = 1
    best_ae2d_loss = 1
    best_ae3d_epoch = -1
    best_ae2d_epoch = -1
    loss_fn_l1 = nn.L1Loss()
    loss_fn_cc = CC()
    loss_fn_sam = SAM()

    for epoch in range(epoches):

        for step, (LRHS, PAN) in enumerate(ae_loader):
            #LRHS = Variable(F.interpolate(LRHS, scale_factor=2, mode='nearest', align_corners=None),requires_grad=True)
            #LRHS = Variable(F.interpolate(LRHS, scale_factor=4, mode='nearest', align_corners=None),requires_grad=True)
            b,c,h,w = LRHS.shape
            LRHS = LRHS.to(device)
            LRHS = LRHS.reshape(b,1,c,h,w)
            #PAN = PAN[:, :, ::2, ::2]
            PAN = PAN[:, :, ::4, ::4]
            PAN = PAN.to(device)

            optim_ae3d.zero_grad()
            _,out_ae3d = ae3d(LRHS)
            loss_ae3d_l1 = loss_fn_l1(out_ae3d,LRHS)
            loss_ae3d_sam = alpha_sam*loss_fn_sam(torch.squeeze(out_ae3d),torch.squeeze(LRHS))
            loss_ae3d = loss_ae3d_l1 + loss_ae3d_sam
            loss_ae3d.backward()
            optim_ae3d.step()

            if loss_ae3d < best_ae3d_loss:
                best_ae3d_loss = loss_ae3d
                best_ae3d_epoch = epoch
                torch.save(ae3d.state_dict(), weights_path_1 + '/ae3d_weights.pth')

            optim_ae2d.zero_grad()
            _,out_ae2d = ae2d(PAN)
            loss_ae2d_l1 = loss_fn_l1(out_ae2d, PAN)
            loss_ae2d_cc = 1 - loss_fn_cc(out_ae2d, PAN)
            loss_ae2d = loss_ae2d_l1 + loss_ae2d_cc
            loss_ae2d.backward()
            optim_ae2d.step()

            if loss_ae2d < best_ae2d_loss:
                best_ae2d_loss = loss_ae2d
                best_ae2d_epoch = epoch
                torch.save(ae2d.state_dict(), weights_path_2 + '/ae2d_weights.pth')

        print("epoch:", epoch, "loss_ae3d:", loss_ae3d.item(),"loss_ae2d:",loss_ae2d.item())

        if (epoch + 1) % epoches == 0:
            index = 1
            with torch.no_grad():

                for step, (LRHS, PAN) in enumerate(ae_loader_test):
                    #LRHS = F.interpolate(LRHS, scale_factor=2, mode='nearest', align_corners=None)
                    #LRHS = F.interpolate(LRHS, scale_factor=4, mode='nearest', align_corners=None)
                    b, c, h, w = LRHS.shape
                    LRHS = LRHS.to(device)
                    LRHS = LRHS.reshape(b,1,c, h, w)
                    #PAN = PAN[:, :, ::2, ::2]
                    PAN = PAN[:, :, ::4, ::4]
                    PAN = PAN.to(device)

                    ae3d.load_state_dict(torch.load(weights_path_1 + '/ae3d_weights.pth'))
                    ae2d.load_state_dict(torch.load(weights_path_2 + '/ae2d_weights.pth'))
                    ae3d.eval()
                    ae2d.eval()
                    out1,_ = ae3d(LRHS)
                    out1 = out1.reshape(b,c*out1.shape[1],h,w)
                    out2,_ = ae2d(PAN)
                    HS_fusion = torch.cat([out1,out2],1)

                    HS_fusion = np.array((HS_fusion).cpu())
                    path_str = os.path.join(code_path,concat_str,'epoch' + str(epoch + 1))
                    path = os.path.join(path_str, 'encode_' + str(index) + '.mat')
                    index = index + 1
                    if not os.path.exists(path_str):
                        os.mkdir(path_str)

                    sio.savemat(path, {'b': HS_fusion.squeeze()})

            print("best_ae3d_epoch:", best_ae3d_epoch, "best_ae3d_loss:", best_ae3d_loss.item(),
                  "best_ae2d_epoch:", best_ae2d_epoch, "best_ae2d_loss:", best_ae2d_loss.item())


if __name__ == '__main__':
    main()


