import torch.nn as nn

class M_kemb(nn.Module):
    def __init__(self, in_dim=21 * 21, out_dim=128):
        super(M_kemb, self).__init__()

        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.act = nn.ReLU()

    def forward(self, k):
        # Reshape the input tensor to a flat
        k_flat = k.view(k.size(0), -1)

        # Apply the linear transformation
        k_emb = self.linear(k_flat)

        # Apply ReLU activation
        k_emb = self.act(k_emb)

        return k_emb

