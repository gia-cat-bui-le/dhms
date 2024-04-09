# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # import clip
# from model.rotation2xyz import Rotation2xyz
# from teach.data.tools import lengths_to_mask

# from model.rotary_embedding_torch import RotaryEmbedding
# from model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like

# from einops import rearrange, reduce, repeat
# from einops.layers.torch import Rearrange, Reduce

# from typing import Any, Callable, List, Optional, Union

# from torch import Tensor

# class DenseFiLM(nn.Module):
#     """Feature-wise linear modulation (FiLM) generator."""

#     def __init__(self, embed_channels):
#         super().__init__()
#         self.embed_channels = embed_channels
#         self.block = nn.Sequential(
#             nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
#         )

#     def forward(self, position):
#         pos_encoding = self.block(position)
#         pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
#         scale_shift = pos_encoding.chunk(2, dim=-1)
#         return scale_shift

# class TransformerEncoderLayer(nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         nhead: int,
#         dim_feedforward: int = 2048,
#         dropout: float = 0.1,
#         activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
#         layer_norm_eps: float = 1e-5,
#         batch_first: bool = False,
#         norm_first: bool = True,
#         device=None,
#         dtype=None,
#         rotary=None,
#     ) -> None:
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(
#             d_model, nhead, dropout=dropout, batch_first=batch_first
#         )
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm_first = norm_first
#         self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
#         self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = activation

#         self.rotary = rotary
#         self.use_rotary = rotary is not None

#     def forward(
#         self,
#         src: Tensor,
#         src_mask: Optional[Tensor] = None,
#         src_key_padding_mask: Optional[Tensor] = None,
#     ) -> Tensor:
#         x = src
#         if self.norm_first:
#             x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
#             x = x + self._ff_block(self.norm2(x))
#         else:
#             x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
#             x = self.norm2(x + self._ff_block(x))

#         return x

#     # self-attention block
#     def _sa_block(
#         self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
#     ) -> Tensor:
#         qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
#         x = self.self_attn(
#             qk,
#             qk,
#             x,
#             attn_mask=attn_mask,
#             key_padding_mask=key_padding_mask,
#             need_weights=False,
#         )[0]
#         return self.dropout1(x)

#     # feed forward block
#     def _ff_block(self, x: Tensor) -> Tensor:
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout2(x)
    
# def featurewise_affine(x, scale_shift):
#     scale, shift = scale_shift
#     return (scale + 1) * x + shift
    
# class FiLMTransformerDecoderLayer(nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         nhead: int,
#         dim_feedforward=2048,
#         dropout=0.1,
#         activation=F.relu,
#         layer_norm_eps=1e-5,
#         batch_first=False,
#         norm_first=True,
#         device=None,
#         dtype=None,
#         rotary=None,
#     ):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(
#             d_model, nhead, dropout=dropout, batch_first=batch_first
#         )
#         self.multihead_attn = nn.MultiheadAttention(
#             d_model, nhead, dropout=dropout, batch_first=batch_first
#         )
#         # Feedforward
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm_first = norm_first
#         self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
#         self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
#         self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)
#         self.activation = activation

#         self.film1 = DenseFiLM(d_model)
#         self.film2 = DenseFiLM(d_model)
#         self.film3 = DenseFiLM(d_model)

#         self.rotary = rotary
#         self.use_rotary = rotary is not None

#     # x, cond, t
#     def forward(
#         self,
#         tgt,
#         memory,
#         t,
#         tgt_mask=None,
#         memory_mask=None,
#         tgt_key_padding_mask=None,
#         memory_key_padding_mask=None,
#     ):
#         x = tgt
#         if self.norm_first:
#             # self-attention -> film -> residual
#             x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
#             x = x + featurewise_affine(x_1, self.film1(t))
#             # cross-attention -> film -> residual
#             x_2 = self._mha_block(
#                 self.norm2(x), memory, memory_mask, memory_key_padding_mask
#             )
#             x = x + featurewise_affine(x_2, self.film2(t))
#             # feedforward -> film -> residual
#             x_3 = self._ff_block(self.norm3(x))
#             x = x + featurewise_affine(x_3, self.film3(t))
#         else:
#             x = self.norm1(
#                 x
#                 + featurewise_affine(
#                     self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
#                 )
#             )
#             x = self.norm2(
#                 x
#                 + featurewise_affine(
#                     self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
#                     self.film2(t),
#                 )
#             )
#             x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
#         return x

#     # self-attention block
#     # qkv
#     def _sa_block(self, x, attn_mask, key_padding_mask):
#         qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
#         x = self.self_attn(
#             qk,
#             qk,
#             x,
#             attn_mask=attn_mask,
#             key_padding_mask=key_padding_mask,
#             need_weights=False,
#         )[0]
#         return self.dropout1(x)

#     # multihead attention block
#     # qkv
#     def _mha_block(self, x, mem, attn_mask, key_padding_mask):
#         q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
#         k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
#         x = self.multihead_attn(
#             q,
#             k,
#             mem,
#             attn_mask=attn_mask,
#             key_padding_mask=key_padding_mask,
#             need_weights=False,
#         )[0]
#         return self.dropout2(x)

#     # feed forward block
#     def _ff_block(self, x):
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout3(x)


# class DecoderLayerStack(nn.Module):
#     def __init__(self, stack):
#         super().__init__()
#         self.stack = stack

#     def forward(self, x, cond, t):
#         for layer in self.stack:
#             x = layer(x, cond, t)
#         return x


# class MDM(nn.Module):
#     def __init__(self, modeltype, njoints, nfeats, translation, pose_rep, glob, glob_rot,
#                  latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
#                  ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='aist', #clip_dim=512,
#                  arch='trans_enc', emb_trans_dec=False, use_rotary=True, #clip_version=None,
#                  **kargs):
#         super().__init__()

#         #self.legacy = legacy
#         self.rotary = None
#         self.modeltype = modeltype
#         self.njoints = njoints
#         self.nfeats = nfeats
#         # self.num_actions = num_actions
#         self.data_rep = data_rep
#         self.dataset = dataset

#         self.pose_rep = pose_rep
#         self.glob = glob
#         self.glob_rot = glob_rot
#         self.translation = translation

#         self.latent_dim = latent_dim

#         self.ff_size = ff_size
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.dropout = dropout

#         self.ablation = ablation
#         self.activation = activation
#         # self.clip_dim = clip_dim
#         self.action_emb = kargs.get('action_emb', None)

#         #!: this setting follow EDGE
#         self.music_dim = 35 #baseline feats
#         pos_dim = 3
#         rot_dim = self.njoints * self.nfeats  # 24 joints, 6dof
#         self.input_feats = pos_dim + rot_dim + 4
#         output_feats = self.input_feats
#         # print(f"total feats: {self.input_feats}")
#         # self.input_feats = self.njoints * self.nfeats

#         self.normalize_output = kargs.get('normalize_encoder_output', False)
        
#         self.cond_drop_prob = kargs.get('cond_drop_prob')

#         self.cond_mode = kargs.get('cond_mode', 'no_cond') # it's 'music' here
#         self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
#         self.arch = arch
#         self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)
#         self.cond_encoder = nn.Sequential()
#         for _ in range(2):
#             self.cond_encoder.append(
#                 TransformerEncoderLayer(
#                     d_model=latent_dim,
#                     nhead=num_heads,
#                     dim_feedforward=ff_size,
#                     dropout=dropout,
#                     activation=activation,
#                     batch_first=True,
#                     rotary=self.rotary,
#                 )
#             )

#         self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
#         self.emb_trans_dec = emb_trans_dec

#         self.motion_mask = kargs['motion_mask']
#         self.hist_frames = kargs['hist_frames']
        
#         self.rotary = None
#         self.abs_pos_encoding = nn.Identity()
#         # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
#         if use_rotary:
#             self.rotary = RotaryEmbedding(dim=latent_dim)
#         else:
#             self.abs_pos_encoding = PositionalEncoding(
#                 latent_dim, dropout, batch_first=True
#             )
            
#         self.time_mlp = nn.Sequential(
#             SinusoidalPosEmb(latent_dim),  # learned?
#             nn.Linear(latent_dim, latent_dim * 4),
#             nn.Mish(),
#         )

#         self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)

#         self.to_time_tokens = nn.Sequential(
#             nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
#             Rearrange("b (r d) -> b r d", r=2),
#         )
        
#         self.null_cond_embed = nn.Parameter(torch.randn(1, 150, latent_dim))
#         self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))
        
#         self.non_attn_cond_projection = nn.Sequential(
#             nn.LayerNorm(latent_dim),
#             nn.Linear(latent_dim, latent_dim),
#             nn.SiLU(),
#             nn.Linear(latent_dim, latent_dim),
#         )

#         self.norm_cond = nn.LayerNorm(latent_dim)
        
#         # input projection
#         self.cond_encoder = nn.Sequential()
#         for _ in range(2):
#             self.cond_encoder.append(
#                 TransformerEncoderLayer(
#                     d_model=latent_dim,
#                     nhead=num_heads,
#                     dim_feedforward=ff_size,
#                     dropout=dropout,
#                     activation=activation,
#                     batch_first=True,
#                     rotary=self.rotary,
#                 )
#             )
#         # conditional projection
#         self.cond_projection = nn.Linear(self.music_dim, latent_dim)
#         self.non_attn_cond_projection = nn.Sequential(
#             nn.LayerNorm(latent_dim),
#             nn.Linear(latent_dim, latent_dim),
#             nn.SiLU(),
#             nn.Linear(latent_dim, latent_dim),
#         )
#         # decoder
#         decoderstack = nn.ModuleList([])
#         for _ in range(num_layers):
#             decoderstack.append(
#                 FiLMTransformerDecoderLayer(
#                     latent_dim,
#                     num_heads,
#                     dim_feedforward=ff_size,
#                     dropout=dropout,
#                     activation=activation,
#                     batch_first=True,
#                     rotary=self.rotary,
#                 )
#             )

#         self.seqTransDecoder = DecoderLayerStack(decoderstack)
        
#         self.final_layer = nn.Linear(latent_dim, output_feats)

#         if self.hist_frames > 0:
#             # self.hist_frames = 5
#             self.seperation_token = nn.Parameter(torch.randn(latent_dim))
#             self.skel_embedding = nn.Linear(self.input_feats, self.latent_dim)
#         seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
#                                                               nhead=self.num_heads,
#                                                               dim_feedforward=self.ff_size,
#                                                               dropout=self.dropout,
#                                                               activation=self.activation)

#         self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
#                                                          num_layers=self.num_layers)
        
#         self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

#         self.embed_music = nn.Linear(self.music_dim, self.latent_dim)

#         self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
#                                             self.nfeats)

#         self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

#     def parameters_wo_clip(self):
#         return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]


#     def mask_cond(self, cond, force_mask=False):
#         bs, s, d = cond.shape
#         if force_mask:
#             return torch.zeros_like(cond)
#         elif self.training and self.cond_mask_prob > 0.:
#             print("Masking")
#             mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
#             # print(f"mask shape: {mask.shape}\t\tcond shape: {cond.shape}")
#             return cond * (1. - mask)
#         else:
#             return cond

#     def forward(self, x, timesteps, y=None):
#         # print("GOTO: MDM forward")
#         """
#         # x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
#         timesteps: [batch_size] (int)
#         """
#         # bs, njoints, nfeats, nframes = x.shape
#         bs, input_feats, nframes = x.shape
#         device = x.device
        
#         x = self.input_process(x)
#         # add the positional embeddings of the input sequence to provide temporal information
#         x = self.abs_pos_encoding(x)

#         # create music conditional embedding with conditional dropout
#         keep_mask = prob_mask_like((bs,), 1 - self.cond_drop_prob, device=device)
#         keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
#         keep_mask_hidden = rearrange(keep_mask, "b -> b 1")

#         cond_tokens = self.embed_music(y['music'])
#         # encode tokens
#         cond_tokens = self.abs_pos_encoding(cond_tokens)
#         cond_tokens = self.cond_encoder(cond_tokens)

#         null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
#         cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)

#         mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
#         cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

#         # create the diffusion timestep embedding, add the extra music projection
#         t_hidden = self.time_mlp(timesteps)

#         # project to attention and FiLM conditioning
#         t = self.to_time_cond(t_hidden)
#         t_tokens = self.to_time_tokens(t_hidden)

#         # FiLM conditioning
#         null_cond_hidden = self.null_cond_hidden.to(t.dtype)
#         cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
#         t += cond_hidden
        
#         if self.hist_frames == 0 or y.get('hframes', None) == None:
#             c = torch.cat((cond_tokens, t_tokens), dim=-2)
#             cond_tokens = self.norm_cond(c)
            
#             output = self.seqTransDecoder(x, cond_tokens, t)
#         else:
#             hframes = y['hframes'].squeeze(2).permute(0, 2, 1) #todo find out the diff 
#             hframes_emb = self.skel_embedding(hframes)
            
#             c = torch.cat((cond_tokens, t_tokens, hframes_emb), dim=-2)
#             cond_tokens = self.norm_cond(c)
            
#             output = self.seqTransDecoder(x, cond_tokens, t)

#         output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
#         return output


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)

#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # not used in the final model
#         x = x + self.pe[:x.shape[0], :]
#         return self.dropout(x)

# class TimestepEmbedder(nn.Module):
#     def __init__(self, latent_dim, sequence_pos_encoder):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.sequence_pos_encoder = sequence_pos_encoder

#         time_embed_dim = self.latent_dim
#         self.time_embed = nn.Sequential(
#             nn.Linear(self.latent_dim, time_embed_dim),
#             nn.SiLU(),
#             nn.Linear(time_embed_dim, time_embed_dim),
#         )

#     def forward(self, timesteps):
#         return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


# class  InputProcess(nn.Module):
#     def __init__(self, data_rep, input_feats, latent_dim):
#         super().__init__()
#         self.data_rep = data_rep
#         self.input_feats = input_feats
#         self.latent_dim = latent_dim
#         self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

#     def forward(self, x):
        
#         x = x.permute((0, 2, 1)) # bs, nframes, nfeatures

#         if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
#             # print(x.device)
#             x = self.poseEmbedding(x)  # [seqlen, bs, d]
#             return x
#         else:
#             raise ValueError


# class OutputProcess(nn.Module):
#     def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
#         super().__init__()
#         self.data_rep = data_rep
#         self.input_feats = input_feats
#         self.latent_dim = latent_dim
#         self.njoints = njoints
#         self.nfeats = nfeats
#         self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

#     def forward(self, output):
#         nframes, bs, d = output.shape
#         if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
#             output = self.poseFinal(output)  # [seqlen, bs, 150]
#         else:
#             raise ValueError
#         output = output.permute(0, 2, 1)  # [bs, njoints, nfeats, nframes]
#         return output

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from teach.data.tools import lengths_to_mask

# from model.rotary_embedding_torch import RotaryEmbedding
# from model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like

# from einops import rearrange, reduce, repeat
# from einops.layers.torch import Rearrange, Reduce

# from typing import Any, Callable, List, Optional, Union

# from torch import Tensor

class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='aistpp', #clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()

        #self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        # self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        # self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)
        
        self.music_dim = 35 #baseline feats
        pos_dim = 3
        rot_dim = self.njoints * self.nfeats  # 24 joints, 6dof
        self.input_feats = pos_dim + rot_dim + 4
        # output_feats = self.input_feats

        # self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_drop_prob = kargs.get('cond_drop_prob')

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        self.motion_mask = kargs['motion_mask']
        self.hist_frames = kargs['hist_frames']

        if self.arch == 'past_cond':
            if self.hist_frames > 0:
                # self.hist_frames = 5
                self.seperation_token = nn.Parameter(torch.randn(latent_dim))
                self.skel_embedding = nn.Linear(self.njoints, self.latent_dim)
        if self.arch == 'trans_enc' or self.arch == 'past_cond' or self.arch == 'inpainting':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')
            if 'music' in self.cond_mode:
                self.embed_music = nn.Linear(self.music_dim, self.latent_dim)
                print("EMBED MUSIC")

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        # self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, seq, d = cond.shape
        cond = cond.reshape(bs, seq*d)
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond.reshape(bs, seq, d)

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit', 'babel'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)
        if "music" in self.cond_mode:
            music_emb = self.embed_music(y['music'])
            emb += self.mask_cond(music_emb, force_mask=force_mask)
        
        # if self.arch == 'gru':
        #     x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
        #     emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]
        #     emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
        #     emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
        #     x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]

        x = self.input_process(x)

        if self.arch == 'trans_enc':
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'past_cond' or self.arch == 'inpainting':
            mask = lengths_to_mask(y['lengths'], x.device)
            if self.arch == 'inpainting' or self.hist_frames == 0 or y.get('hframes', None) == None:
                token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
                aug_mask = torch.cat((token_mask, mask), 1)
                xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
                xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
                if self.motion_mask:
                    output = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
                else:
                    output = self.seqTransEncoder(xseq)[1:]
            else:
                token_mask = torch.ones((bs, 2 + self.hist_frames), dtype=bool, device=x.device)
                aug_mask = torch.cat((token_mask, mask), 1)
                sep_token = torch.tile(self.seperation_token, (bs,)).reshape(bs, -1).unsqueeze(0)
                hframes = y['hframes'].squeeze(2).permute(2, 0, 1) #TODO find out the diff 
                hframes_emb = self.skel_embedding(hframes)
                # hframes_emb = hframes_emb.permute(1, 0, 2) # [5 b dim]
                xseq = torch.cat((emb, hframes_emb, sep_token, x), axis=0)
                # TODO add attention mask
                xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
                output = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)[2 + self.hist_frames:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nfeats, nframes = x.shape
        x = x.permute((2, 0, 1))

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.nfeats)
        output = output.permute(1, 2, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output