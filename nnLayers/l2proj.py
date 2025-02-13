import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import math

class L2Proj(nn.Module):
    # L2Prox layer
    # Y = NN_L2TRPROX(X,EPSILON) computes the Proximal map layer for the
    #   indicator function :
    #
    #                      { 0 if ||X|| <= EPSILON
    #   i_C(D,EPSILON){X}= {
    #                      { +inf if ||X|| > EPSILON
    #
    #   X and Y are of size H x W x K x N, and EPSILON = exp(ALPHA)*V*STDN
    #   is a scalar or a 1 x N vector, where V = sqrt(H*W*K-1).
    #
    #   Y = K*X where K = EPSILON / max(||X||,EPSILON);

    def __init__(self):
        super(L2Proj, self).__init__()
    
    def forward(self,input,alpha,stdn):
        assert(input.dim() == 4), "Input is expected to be a 4-D tensor."
        assert(alpha is None or alpha.numel() == 1), "alpha needs to be "\
        "either None or a tensor of size 1."
                
        N = math.sqrt(input[0].numel()-1)
        batch = input.size(0)
               
        assert(stdn.numel() == 1 or stdn.numel() == batch), \
            "stdn must be either a tensor of size one or a tensor of size "\
            "equal to the batch number."
        assert(all(stdn.view(-1) > 0)), "The noise standard deviations must be positive."
        
        stdn = stdn.view(-1,1,1,1)
        
        if alpha is None:
            alpha = torch.Tensor([0]).type_as(stdn)
        
        epsilon = stdn.mul(alpha.exp())*N
        
        input_norm = input.contiguous().view(batch,-1).norm(p=2,dim=1).view(batch,1,1,1)
        max_norm = input_norm.max(epsilon)
        
        return input.mul(epsilon).div(max_norm)


def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    alpha = torch.FloatTensor([2.])
    alpha.requires_grad_()
    stdn = torch.FloatTensor([5.])
    loss_fn = mse_loss
    module = L2Proj()
    x = torch.randn(5, 3, 20, 20)
    orig = (x.data + torch.rand(x.shape) * 25)
    optimizer = optim.Adam([alpha], lr=0.01)
    for  i in range(10):
        out = module(x, alpha, stdn)
        loss = loss_fn(out, orig)
        loss.backward()
        print(loss.item())
        optimizer.step()
    print(alpha)
