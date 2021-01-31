import torch
import numpy as np
from torch import nn, optim
from dataloader import HS
from torch.utils.data import DataLoader
from torch.autograd import Variable
import scipy.io as sio
import os
from ResNet_model import ResNet


torch.manual_seed(22)
torch.cuda.manual_seed(22)
np.random.seed(22)

#Here is the Hyperparameter
device = torch.device('cuda')
epoches = 300
batchsz = 7
learning_rate = 0.001
data_in_channel = 220
data_out_channel = 102
divid_epoches = 50

#Here is the path to load and save data
version = 'v1'
data_name = 'data'
root = '/home/xd132/xzc/project1/code'
root_gtHS = '/home/xd132/xzc/project1/data_' + data_name
weights_path = '/home/xd132/xzc/project1/weights/' + version + '/' + data_name
fusion_path = '/home/xd132/xzc/project1/fusion/' + version + '/' + data_name
#Generate iterators that load data
trian_db = HS(root_gtHS,root,data_name,version,'train')
test_db = HS(root_gtHS,root,data_name,version,'test')
train_loader = DataLoader(trian_db, batch_size=batchsz, shuffle=True,num_workers=8,drop_last=False)
test_loader = DataLoader(test_db, batch_size=1,num_workers=4,drop_last=False)


def main():

    model = ResNet(data_in_channel,data_out_channel)
    model.to(device)

    if os.path.exists(weights_path + '/resnet_weights.pth'):
        print('start load last time resnet weights')
        model.load_state_dict(torch.load(weights_path + '/resnet_weights.pth'))

    print(model)

    optim_model = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_model, milestones=[150,300], gamma=0.1, last_epoch=-1)
    best_loss = 1
    best_epoch = -1
    loss_fn_l1 = nn.L1Loss()

    for epoch in range(epoches):

        for step, (input1,input2,input3,gtHS) in enumerate(train_loader):

            input1 = Variable(input1,requires_grad=True).to(device)
            input2 = Variable(input2,requires_grad=True).to(device)
            input3 = Variable(input3,requires_grad=True).to(device)
            gtHS = Variable(gtHS,requires_grad=True).to(device)

            HS_fusion = model(input1,input2,input3)
            optim_model.zero_grad()
            loss = loss_fn_l1(HS_fusion,gtHS)
            loss.backward()
            optim_model.step()

            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                torch.save(model.state_dict(), weights_path + '/resnet_weights.pth')
        print("epoch:",epoch,"loss:",loss.item())
        scheduler.step()

        #Output results every divid_epoches epochs
        if (epoch + 1) % divid_epoches == 0:

            index_test = 1
            with torch.no_grad():

                for step, (input1,input2,input3, _) in enumerate(test_loader):

                    input1 = input1.to(device)
                    input2 = input2.to(device)
                    input3 = input3.to(device)

                    model.load_state_dict(torch.load(weights_path + '/resnet_weights.pth'))
                    model.eval()

                    HS_fusion = model(input1,input2,input3)
                    HS_fusion = np.array(HS_fusion.cpu())

                    path_str = os.path.join(fusion_path, 'epoch' + str(epoch + 1))
                    if not os.path.exists(path_str):
                        os.mkdir(path_str)

                    path = os.path.join(path_str, 'fusion_' + str(index_test) + '.mat')
                    index_test = index_test + 1
                    sio.savemat(path, {'b': HS_fusion.squeeze()})

            print('best_loss:',best_loss,"best_epoch:",best_epoch)


if __name__ == "__main__":
    main()

