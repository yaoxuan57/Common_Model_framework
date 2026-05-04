import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath

from .Transformer_utils import FullAttention, AttentionLayer, Encoder, EncoderLayer, random_masking


# ------------ AUGMENTATIONS for Contrastive learning -------------------
def time_shift(x, shift_ratio=0.2):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    signal_length = x.shape[2]
    shift = int(signal_length * shift_ratio)
    shifted_sample = torch.cat((x[:, :, signal_length - shift:], x[:, :, :signal_length - shift]), dim=2)
    return shifted_sample


def scaling_with_jitter(x, sigma=0.05):
    factor = torch.normal(mean=1., std=sigma, size=(x.shape[0], x.shape[2])).cuda()
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append((xi * factor[:, :]).unsqueeze(1))
    return torch.cat((ai), dim=1)


# ------------------------------------------------------------------------


# ------------ Transformer design ----------------------------------------



class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=16, stride=16, embed_dim=768):
        super().__init__()

        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches

        self.kernel = patch_size
        self.stride = patch_size
        self.input_layer = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.kernel, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        x_out = self.input_layer(x)
        return x_out



class Transformer_bkbone(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.patch_embed = PatchEmbed(
            seq_len=args.seq_len, patch_size=args.patch_size, stride=args.patch_size, embed_dim=args.embed_dim
        )

        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, args.embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=args.dropout)

        # lora_rank = None
        # lora_alpha = 16
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=args.dropout,
                                      output_attention=False), args.embed_dim, args.heads),
                    args.embed_dim,
                    4 * args.embed_dim,
                    dropout=args.dropout,
                    activation='gelu'
                ) for _ in range(args.depth)
            ],
            norm_layer=torch.nn.LayerNorm(args.embed_dim)
        )

        self.input_layer = nn.Linear(args.patch_size, args.embed_dim)
        self.pretrain_head = nn.Linear(args.embed_dim, args.patch_size)

        # Classifier head
        self.head = nn.Linear(args.embed_dim, args.num_classes)

    def forward(self, x):
        x_patch = self.patch_embed(x)
        # x: [Batch * Channel, num of Patches, Embed_dim]
        x_patch = x_patch + self.pos_embed
        # x: [Batch * Channel, num of Patches, Embed_dim]
        x_patch = self.pos_drop(x_patch)
        # x: [Batch * Channel, num of Patches, Embed_dim]
        features, _ = self.encoder(x_patch)
        # x: [Batch * Channel, num of Patches, Embed_dim] --> [Batch, Channel * num_patches, Embed_dim]
        features = torch.reshape(features, (-1, self.args.num_channels * features.shape[-2], features.shape[-1]))

        return features

    def cl_pretrain(self, x):  # Contrastive-based pretraining
        x_aug1 = time_shift(x)
        x_aug2 = scaling_with_jitter(x, sigma=0.2)

        features1 = self.forward(x_aug1)
        features2 = self.forward(x_aug2)  # [Batch, Channel * num_patches, Embed_dim]
        features1 = self.predict(features1)  # [batch, num_classes]
        features2 = self.predict(features2)

        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)

        return features1, features2

    def reconstruct(self, x, mask_ratio=0.5):  # Reconstruction-based pretraining
        x_patch = self.patch_embed(x)  # [B, N, D]
        B, N, D = x_patch.shape

        x_patch = x_patch + self.pos_embed[:, :N, :]
        x_patch = self.pos_drop(x_patch)

        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x_patch, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)) # [B, len_keep, D]

        encoded, _ = self.encoder(x_visible) # [B, len_keep, D]

        # Insert mask tokens
        mask_tokens = torch.zeros(B, N - len_keep, D, device=x.device)
        x_full = torch.cat([encoded, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))

        # Predict and calculate loss
        pred_patches = self.pretrain_head(x_full)  # [B, N, embedding] --> [B, N, patch_size]
        target_patches = self.patch_embed.embed(x).view(B, N, -1)

        loss = (pred_patches - target_patches) ** 2
        loss = loss.mean(dim=-1)

        mask = torch.ones([B, N], device=x.device)
        mask.scatter_(1, ids_keep, 0)
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def predict(self, features):
        features_flat = features.mean(1)  # --> [batch, dimension]
        predictions = self.head(features_flat).squeeze()   # [batch, dimension] --> [batch, num_classes]
        return predictions

    # def pretrain(self, x):
    #     """
    #     Steps:
    #     1- Patchify the input: x[Batch, Channel, Length] --> x[Batch, Channel, num of Patches, Patch Len]
    #     2- Masking the input: x[Batch, Channel, num of Patches, Patch Len] --> x_masked[Batch, num of Patches, Channel, Patch Len]
    #     3- Converting the input to the embedding space: x_masked[Batch, num of Patches, Channel, Patch Len] --> x_patch[Batch, num of Patches, Channel, Embed_dim]
    #     4- Combine the batch with channel dimension: x_patch[Batch, num of Patches, Channel, Embed_dim] --> x_patch[Batch * Channel, num of Patches, Embed_dim]
    #     5- Applying the Transformer Encoder on Masked: x_masked[Batch * Channel, num of Patches, Embed_dim] --> x_masked[Batch * Channel, num of Patches, Embed_dim]
    #     6- Decouple the Batch dim and channel dim: x_masked[Batch * Channel, num of Patches, Embed_dim] --> x_masked[Batch, Channel, num of Patches, Embed_dim]
    #     6- Convert the Encoder output to the Patched input space: x_masked[Batch, num of Patches, Channel, Embed_dim] --> x_masked[Batch, Channel, num of Patches, Patch Len]
    #     7- Loss between Patchified input and the reconstructed output:

    #     """
    #     # Patchify the input:
    #     # X: [Batch, Channel, Length] --> x: [Batch, Channel, num of Patches, Patch Len]
    #     num_channels = x.shape[1]

    #     x_patch = x.unfold(dimension=-1, size=self.args.patch_size, step=self.args.patch_size)

    #     # Rearrange
    #     # x: [Batch, Channel, num of Patches, Patch Len]--> x:[Batch, num of Patches, Channel, Patch Len]
    #     x_to_mask = rearrange(x_patch, 'b c p l -> b p c l')

    #     # Masking Should be applied on the original input after patching it but before the embedding layer
    #     # x_masked: [Batch, num of Patches, Channel, Patch Len]
    #     x_masked, _, self.mask, _ = random_masking(x_to_mask, mask_ratio=self.args.masking_ratio)
    #     self.mask = self.mask.bool()  # mask: [bs x num_patch x n_vars]

    #     # Embedding the input: x_masked:
    #     # [Batch, num of Patches, Channel, Patch Len] --> x_embed: [Batch, num of Patches, Channel, Embed_dim]
    #     x_embed = self.input_layer(x_masked)

    #     # Combine the batch with channel dimension:
    #     # x_patch[Batch, num of Patches, Channel, Embed_dim] --> x_patch[Batch * Channel, num of Patches, Embed_dim]
    #     x_embed = rearrange(x_embed, 'b p c d -> (b c) p d')

    #     # Applying the Transformer Encoder on Masked: ]
    #     enc_out, _ = self.encoder(x_embed)

    #     # Decouple the Batch dim and channel dim:
    #     # x_masked[Batch * Channel, num of Patches, Embed_dim] --> x_masked[Batch, Channel, num of Patches, Embed_dim]
    #     enc_out = rearrange(enc_out, '(b c) p d -> b c p d', c=num_channels)

    #     # Convert out from embedding space to the original space:
    #     # enc_out[Batch, Channel, num of Patches, Embed_dim] --> x_hat[Batch, Channel, num of Patches, Patch Len]
    #     # select whether will use input space or embedding space
    #     # if self.args.reconstruct_type == 'input':
    #     # reconstruct the embedding to input space
    #     prediction = self.pretrain_head(enc_out)
    #     target = x_patch

    #     # # use the predictions in the space embeddings
    #     # prediction = enc_out
    #     # target = self.input_layer(x_patch)

    #     return prediction, target