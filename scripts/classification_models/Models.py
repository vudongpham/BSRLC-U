import torch.nn as nn
import torch
from torchinfo import summary

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, dilation=1, padding=1, activation=True, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding
        )
        self.activation = nn.ReLU() if activation else None
        self.bn = nn.BatchNorm2d(out_channel) if bn else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x
    

class TransformerBlock(nn.Module):
    """
    A Transformer block that allows a query to attend to a key/value source.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        # query shape: (B, 1, D)
        # key_value shape: (B, N, D), where N is the number of pixels
        
        # --- Attention Part ---
        # Apply pre-normalization
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)
        
        # The query attends to the key/value source
        attn_output, _ = self.attn(q_norm, kv_norm, kv_norm)
        
        # Residual connection is added back to the original query
        x = query + self.dropout(attn_output)

        # --- FFN Part ---
        # The FFN part processes the updated query token
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        
        # Second residual connection
        out = x + self.dropout(ffn_output)
        
        return out

class ModelPatch(nn.Module):
    def __init__(self, in_feature=28, embed_dim=128, num_heads=4, num_classes=4, transformer_block_num=2):
        super().__init__()

        # 1. Unified CNN backbone 
        self.feature_extractor = nn.Sequential(
            Conv2d(in_feature, 32, 3),
            Conv2d(32, 32, 3),
            Conv2d(32, 64, 3),
            Conv2d(64, 64, 3),
            Conv2d(64, embed_dim, 3), # Output channels should match the attention dimension
            Conv2d(embed_dim, embed_dim, 3, bn=None, activation=None),
        )

        # 2. Multi-head attention layer
        self.attn_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim*4) for _ in range(transformer_block_num)]
        )

        # 3. Final classification head
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
    
    def forward(self, x_in):
        # x_in shape: (B, H, W, C), e.g., (8, 15, 15, 26)
        
        # --- Feature Extraction ---
        # Permute to (B, C, H, W) for convolutions
        x = torch.permute(x_in, (0, 3, 1, 2))
        features = self.feature_extractor(x) # Shape: (B, D, H', W') e.g. (8, 256, 15, 15)
        
        # --- Attention Block ---
        # Permute to (B, H', W', D) for easier indexing and compatibility with batch_first=True
        features = torch.permute(features, (0, 2, 3, 1))
        b, h_prime, w_prime, d = features.shape
        
        # The QUERY is the feature vector of the center pixel from the deep feature map
        query = features[:, h_prime // 2, w_prime // 2, :].unsqueeze(1) # Shape: (B, 1, D)
        
        # The KEY and VALUE are all pixel features from the map
        key_value = features.reshape(b, -1, d) # Shape: (B, H'*W', D)

        refined_query = query
        for block in self.attn_blocks:
            refined_query = block(refined_query, key_value)
        
        # Squeeze out the sequence dimension (dim=1)
        final_repr = refined_query.squeeze(1) # Shape: (B, D)
       

        # --- Final Classification ---
        x_out = self.classification_head(final_repr)

        return x_out
    

class ModelPixel(nn.Module):
    def __init__(self, in_feature=24, num_classes=4):
        super().__init__()
        self.NN_forward = nn.Sequential(
            nn.Linear(in_feature, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_in):
        x_out = self.NN_forward(x_in)
        return x_out


