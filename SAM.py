import torch
import numpy as np


class SAM(torch.nn.Module):

    def __init__(self):
        super(SAM, self).__init__()

    def forward(self,output,HS):
        data1 = torch.sum(output * HS, dim = 1)
        data2 = torch.sqrt(torch.sum((output **2),dim = 1) * torch.sum((HS ** 2),dim = 1))
        data2[data2 == 0] = 1e-16
        sam_loss = torch.acos((data1 / data2)).view(-1).mean().type(torch.float32)*180/torch.tensor(np.pi)
        sam_loss = sam_loss.clone().detach().requires_grad_(True)

        return sam_loss


