import torch


class CC(torch.nn.Module):
    
    def __init__(self):
        super(CC,self).__init__()

    def forward(self, output, HS):
        out_ave = torch.mean(output, dim=[2, 3])
        HS_ave = torch.mean(HS, dim=[2, 3])
        LL = torch.sum((output.permute(2, 3, 0, 1) - out_ave) * (HS.permute(2, 3, 0, 1) - HS_ave), dim=[0, 1])
        MM = torch.sqrt(torch.sum((output.permute(2,3,0,1) - out_ave) ** 2, dim=[0, 1]) * torch.sum((HS.permute(2,3,0,1) - HS_ave) ** 2, dim=[0, 1]))
        loss = torch.div(LL, MM)
        loss = torch.mean(loss, dim=[0,1])
        loss = loss.clone().detach().requires_grad_(True)

        return loss
