import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        self.conv_small = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=10, stride=10) # 10 minute patches
        self.conv_medium = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=40, stride=40) # 40 minute patches
        self.conv_large = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=160, stride=160) # 160 minute patches

    def forward(self, x):

        num_samples = x.shape[0]
        patches_small = self.conv_small(x).transpose(1, 2)
        patches_medium = self.conv_medium(x).transpose(1, 2)
        patches_large = self.conv_large(x).transpose(1, 2)

        x = torch.cat((patches_small, patches_medium, patches_large), 1)

        x = x.reshape(1, num_samples * 2688, self.embed_dim)
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0, proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError
        
        qkv = self.qkv(x) # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)
        dp = (
            q @ k_t
        ) * self.scale
        attn = dp.softmax(dim=1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(
            1, 2
        )
        weighted_avg = weighted_avg.flatten(2)
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x
    
class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) 
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class Transformer(nn.Module):
    def __init__(self,
                 num_samples,
                 embed_dim=324,
                 n_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 p=0,
                 attn_p=0.):
        super().__init__()

        self.num_samples = num_samples

        self.embed_dim = num_samples * embed_dim

        self.n_patches = num_samples * 2688

        self.patch_embed = PatchEmbed(embed_dim=self.embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
    
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, self.embed_dim))

        self.pos_drop = nn.Dropout(p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p
                )
                for _ in range(12)
            ]
        )

        self.norm = nn.LayerNorm(self.embed_dim)

        self.normalize = nn.Hardtanh(0, 1)

        self.head = nn.Linear(self.embed_dim // self.num_samples, 3)

    def forward(self, x):

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(1, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token = x[:, 0]

        split_representations = cls_token.view(self.num_samples, self.embed_dim // self.num_samples)

        output = self.head(split_representations)

        output = self.normalize(output)

        return output

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = Transformer(1)

transformer.to(device)

output = transformer.forward(torch.randn((1, 1, 20480)))

print(output)
"""
""" Input Size [2, 1, 20480]
    tensor([[[-0.7001,  0.5462, -0.8139,  ..., -0.7308, -0.2046,  0.5984]],

        [[-0.6168, -1.3821, -0.2614,  ...,  1.2784,  0.4304, -0.6688]]])
"""

""" Output [2, 1, 20480]
tensor([[-0.8300,  0.5646, -0.1394],
        [-0.3589, -0.6848, -0.4229]], grad_fn=<AddmmBackward0>)
"""
