import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvSmall(nn.Module):
    def __init__(self):
        super(ConvSmall, self).__init__()
        self.n_weights = 11330
        self.weight = torch.nn.Parameter(torch.zeros(self.n_weights), requires_grad=True)

        # initialization
        self.weight.data.normal_(0.0, 0.02)

    def forward(self, x):
        conv1_weight, conv1_bias = self.weight[0:250].view(10,1,5,5), self.weight[250:260].view(10)
        conv2_weight, conv2_bias = self.weight[260:2760].view(10,10,5,5), self.weight[2760:2770].view(10)
        fc1_weight, fc1_bias = self.weight[2770:10770].view(50,160), self.weight[10770:10820].view(50)
        fc2_weight, fc2_bias = self.weight[10820:11320].view(10,50), self.weight[11320:11330].view(10)
        out = x
        out = F.conv2d(out, conv1_weight, bias=conv1_bias)
        out = F.relu(F.max_pool2d(out, 2))
        out = F.conv2d(out, conv2_weight, bias=conv2_bias)
        out = F.relu(F.max_pool2d(out, 2))
        out = F.linear(out.view(-1, 160), fc1_weight, bias=fc1_bias)
        out = F.relu(out)
        out = F.linear(out, fc2_weight, bias=fc2_bias)
        return out
