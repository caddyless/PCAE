# This code is modified from MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class PCAEViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=768, decoder_depth=2, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, use_decoder: bool = True,
                 drop_path=0):
        super().__init__()

        # --------------------------------------------------------------------------
        # PCAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        print('drop path ' + str(dpr))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=dpr[i])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.use_decoder = use_decoder
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # PCAE decoder specifics
        if use_decoder:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=drop_path)
                for i in range(decoder_depth)])

            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
            # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.use_decoder:
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        if self.use_decoder:
            torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        # split patches into ids_keep and ids_remove
        ids_keep = ids_shuffle[:, :len_keep]
        ids_remove = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove

        return x_masked, ids_keep, ids_remove

    def forward_decoder(self, x, ids_keep, teacher_ids_keep):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], teacher_ids_keep.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        ids_keep = torch.cat([ids_keep, teacher_ids_keep], dim=1)

        b, n, d = x.size()

        # add pos embed
        decoder_pos_embed = torch.gather(self.decoder_pos_embed[:, 1:].repeat(b, 1, 1), dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))
        decoder_pos_embed = torch.cat([self.decoder_pos_embed[:, 0].repeat(b, 1, 1), decoder_pos_embed], dim=1)
        x = x + decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, (1 + ids_keep.size(1) - teacher_ids_keep.size(1)):]

        return x

    def forward_student(self, x, ids_keep, teacher_ids_keep):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(-1)))

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = self.forward_decoder(x, ids_keep, teacher_ids_keep)

        return x

    def forward_teacher(self, x, mask_ratio, drop_stage=3, drop_case=0, keep_ratio=0.5, random_drop=False):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        x, student_ids_keep, teacher_ids_keep = self.random_masking(x, mask_ratio=mask_ratio)

        # apply Transformer blocks
        num_blocks = len(self.blocks)
        if drop_case == 0:
            drop_indices = [num_blocks // drop_stage * i for i in range(drop_stage)]
        elif drop_case == 1:
            drop_indices = [num_blocks // drop_stage * (i + 1) - 1 for i in range(drop_stage)]
        elif drop_case == 2:
            drop_indices = np.linsapce(0, num_blocks - 1, drop_stage, endpoint=True, dtype=int)
        elif drop_case == 3:
            drop_indices = [i for i in range(drop_stage)]
        else:
            raise ValueError('Unknown drop case ', drop_case)
        
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i in drop_indices:
                b, n, d = x.size()

                if random_drop:
                    noise = torch.rand(b, n, device=x.device)
                    ids_shuffle = torch.argsort(noise, dim=1)
                else:
                    with torch.no_grad():
                        x_ = F.normalize(x, dim=-1)
                        score_matrix = torch.einsum('bd, bmd -> bm', x_.mean(1), x_)
                        ids_shuffle = torch.argsort(score_matrix, dim=1)
                
                ids_keep = ids_shuffle[:, :int(n * keep_ratio)]
                
                teacher_ids_keep = torch.gather(teacher_ids_keep, dim=1, index=ids_keep)

                x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))
        
        x = self.norm(x)

        return x, teacher_ids_keep, student_ids_keep

    def forward(self, imgs, mask_ratio=0.75, drop_stage=3, drop_case=0, keep_ratio=0.5, student_ids_keep=None, 
                teacher_ids_keep=None, random_drop=False):

        if self.use_decoder:
            return self.forward_student(imgs, student_ids_keep, teacher_ids_keep)
        else:
            return self.forward_teacher(imgs, mask_ratio, drop_stage, drop_case, keep_ratio, random_drop)


def pcae_vit_base_patch16_dec512d8b(**kwargs):
    model = PCAEViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def pcae_vit_large_patch16_dec512d8b(**kwargs):
    model = PCAEViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def pcae_vit_huge_patch14_dec512d8b(**kwargs):
    model = PCAEViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = pcae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = pcae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = pcae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
