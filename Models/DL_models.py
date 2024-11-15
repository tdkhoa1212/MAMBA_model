import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Blocks.BidirectionalMamba import BidirectionalMambaBlock
from Models.Blocks.AdaptiveGraphConvolution import AdaptiveGraphConvolutionBlock

class GRAPH_MAMBA(nn.Module):
    def __init__(self, configs):
        super(GRAPH_MAMBA, self).__init__()
        self.configs = configs
        
        self.mamba_block = BidirectionalMambaBlock(
            pred_len=configs.pred_len, 
            d_model=configs.d_model, 
            d_state=configs.d_state, 
            expand=configs.expand,
            hidden_dimention=configs.hidden_dimention,
            d_conv=configs.d_conv
        )

        self.agc_block = AdaptiveGraphConvolutionBlock(
            node_num=configs.node_num, 
            embed_dim=configs.embed_dim, 
            cheb_k=configs.cheb_k, 
            feature_dim=configs.feature_dim,
        )


        # Regularization parameters
        self.l2_lambda = 1e-4
        self.dropout = nn.Dropout(p=0.2)

        self.projection_h = nn.Linear(configs.node_num*configs.feature_dim, configs.linear_depth//2, bias=True)
        self.projection = nn.Linear(configs.linear_depth, configs.pred_len, bias=True)
        self.flatten = nn.Flatten()

    def forward(self, input_):
        """
        Forward pass through the model:
        1. Apply the Mamba block to process the input sequence.
        2. Apply the GCN layer to capture graph-based dependencies.
        3. Apply the final linear projection to get the prediction.
        """
        x = input_
        for i in range(self.configs.num_layers):
            x = self.mamba_block(x)
        
        x1 = self.flatten(x)
        x1 = self.projection_h(x1)
        x1 = self.dropout(x1)

        x2 = self.agc_block(x)
        x2 = self.dropout(x2)

        x_concat = torch.cat((x1, x2), dim=1)  

        x_out = self.projection(x_concat)
        
        return x_out