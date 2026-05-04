import torch
import torch.nn as nn
from functools import partial
from .model import Transformer_bkbone  # Load encoder from this module

# Sin-cos 1D positional embedding (simple implementation)
def get_1d_sincos_pos_embed(embed_dim, num_patches, cls_token=True):
    position = torch.arange(num_patches, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_dim))
    pe = torch.zeros(num_patches, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if cls_token:
        pe = torch.cat([torch.zeros([1, embed_dim]), pe], dim=0)
    return pe

class MaskedAutoencoderTimeSeries(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Transformer_bkbone(args)
        mlp_ratio = 4
        self.decoder_embed = nn.Linear(args.embed_dim, args.decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.encoder.patch_embed.num_patches + 1, args.decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                args.decoder_embed_dim, 
                args.decoder_num_heads, 
                int(args.decoder_embed_dim * mlp_ratio), 
                dropout = 0.3,
                batch_first=True)
            for _ in range(args.decoder_depth)])
        self.decoder_norm = nn.LayerNorm(args.decoder_embed_dim)
        self.decoder_pred = nn.Linear(args.decoder_embed_dim, args.patch_size * args.num_channels)

        self.initialize_weights()

    def initialize_weights(self):
        self.decoder_pos_embed.data.copy_(get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.encoder.patch_embed.num_patches, cls_token=True))
        nn.init.normal_(self.mask_token, std=0.02)  #mask_token shape (B,N,dec_embed)

    def patchify(self, ts):
        p = self.encoder.args.patch_size
        N, C, L = ts.shape  # N is number of samples, C is channel, L is original timelength
        ts = ts.reshape(N, C, L // p, p).permute(0, 2, 1, 3).reshape(N, L // p, C * p)  # split time axis L into (num_patches = L//p, patch_size = p)
        # [N, num_patches, C, p]
        # flatten channels×patch into features per patch
        return ts

    def unpatchify(self, x):
        p = self.encoder.args.patch_size
        N, L_p, D = x.shape  #(batch, number of patches, dimension)
        C = D // p  # recover channel since in patchify we multiply channels and patch
        ts = x.reshape(N, L_p, C, p).permute(0, 2, 1, 3).reshape(N, C, L_p * p)
        # [N, L_p, C, p]   split features into (C, p)
        # [N, C, L_p, p]   put channels before patches
        # [N, C, L]        stitch patches along time
        return ts

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)  # project from embed_dim --> decoder_embed_dim
        mask_tokens = self.mask_token.repeat(x.size(0), ids_restore.size(1) + 1 - x.size(1), 1)  # [B, num_masked, D_dec]
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # drop the CLS token for now, keep only visible tokens, concat masked after visible, now is [B, N, D_dec]
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).repeat(1, 1, x.size(2))) # ids_restore only has (B,N) shape, unsqueeze at last dimension fill with d_embed
        # now we have (batch, N, dec_embed), all tokens, in the original order with visible and masked in between
        x = torch.cat([x[:, :1, :], x_], dim=1)  # put CLS at the front of the sequence again
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)  # map from decoder_embed_dim --> patch_size * num_channels
        return x[:, 1:, :]

    # def forward_loss(self, ts, pred, mask):
    #     target = self.patchify(ts)
    #     loss = (pred - target) ** 2
    #     loss = loss.mean(dim=-1)
    #     loss = (loss * mask).sum() / mask.sum()
    #     return loss

    def forward(self, ts):
        # ts: [B, C, L]
        B, C, L = ts.shape
        p = self.encoder.args.patch_size
        N = L // p  # number of patches

        # ---- Patchify ----
        x = self.patchify(ts)  # becomes [B, N, C*p], batch, number of tokens, channels*patch_size

        # ---- Random Masking ----
        len_keep = int(N * (1 - self.encoder.args.masking_ratio))
        noise = torch.rand(B, N, device=ts.device)  # [B, N]
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2]))  # [B, N_keep, C*p]

        # ---- Encode ----
        x_encoded = self.encoder.patch_embed.input_layer(x_visible)  # [B, N_keep, C*p] -- > [B, N_keep, embed_dim]
        x_encoded = x_encoded + self.encoder.pos_embed[:, :len_keep, :]
        x_encoded = self.encoder.pos_drop(x_encoded)

        x_encoded, _ = self.encoder.encoder(x_encoded)  # return encoded and attention which is _

        # ---- Decode only masked tokens ----
        pred = self.forward_decoder(x_encoded, ids_restore)  # [B, N, D]

        # ---- Target ----
        target = x  # [B, N, C*p]

        # ---- Compute loss on masked patches only ----
        loss = (pred - target) ** 2  # [B, N, D]
        loss = loss.mean(dim=-1)  # [B, N]

        # Create mask: 1 for masked, 0 for keep
        mask = torch.ones([B, N], device=ts.device)
        mask.scatter_(1, ids_keep, 0)
        loss = (loss * mask).sum() / mask.sum()

        return loss, pred, mask