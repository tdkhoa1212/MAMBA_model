import torch.nn as nn
# from mamba.mamba import Mamba, MambaBlock
from mamba_ssm import Mamba
import torch.nn.functional as F


class BidirectionalMambaBlock(nn.Module):
    def __init__(self, pred_len, d_model, d_state, seq_len, expand, hidden_dimention):
        super(BidirectionalMambaBlock, self).__init__()
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_state = d_state
        self.seq_len = seq_len

        self.mamba = Mamba(  
                            d_model=d_model,  # Model dimension d_model
                            d_state=d_state,  # SSM state expansion factor
                            expand=expand,
                            d_conv=2
                            )
        self.mamba_reversed = Mamba(
                            d_model=d_model,  # Model dimension d_model
                            d_state=d_state,  # SSM state expansion factor
                            expand=expand,
                            d_conv=2
                            )

        self.l2_lambda = 1e-4
        self.norm = nn.LayerNorm(d_model, eps=1e-5, elementwise_affine=True)
        self.dropout = nn.Dropout(p=0.2)
        self.activation = F.relu

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=hidden_dimention, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dimention, out_channels=hidden_dimention, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_dimention, out_channels=d_model, kernel_size=1)

    def forward(self, x):
        y1 = self.mamba(x)  
        y1 = self.dropout(y1)
    
        x_reversed = x.flip(dims=[1])  
        y2 = self.mamba_reversed(x_reversed)
        y2 = self.dropout(y2)
        
        y3 = self.norm(x + y1 + y2.flip(dims=[1]))

        y_prime = self.dropout(self.activation(self.conv1(y3.transpose(-1, 1))))
        y_prime = self.dropout(self.activation(self.conv3(y_prime)))
        y_prime = self.dropout(self.conv3(y_prime).transpose(-1, 1))
        
        out = self.norm(y_prime + y3)

        return out