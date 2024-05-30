import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from teach.data.tools import lengths_to_mask
from vis import SMPLSkeleton
from einops import rearrange, reduce, repeat

from model.x_transformers.x_transformers import ContinuousTransformerWrapper, Encoder
from .rotary_embedding_torch import RotaryEmbedding

class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift

class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
        self,
        tgt,
        memory,
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            # self-attention -> film -> residual
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            # cross-attention -> film -> residual
            x_2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            # feedforward -> film -> residual
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t):
        for layer in self.stack:
            x = layer(x, cond, t)
        return x

class BPE_Schedule():
    def __init__(self, training_rate: float, inference_step: int, max_steps: int) -> None:
        assert training_rate >= 0 and training_rate <= 1, "training_rate must be between 0 and 1"
        assert inference_step == -1 or (inference_step >= 0 and inference_step <= max_steps), "inference_step must be between 0 and max_steps"
        self.training_rate = training_rate
        self.inference_step = inference_step
        self.max_steps = max_steps
        self.last_random = None

    def step(self, t: torch.Tensor, training: bool):
        self.last_random = torch.rand(t.shape[0], device=t.device)

    def get_schedule_fn(self, t: torch.Tensor, training: bool) -> torch.Tensor:
        # False --> absolute
        # True --> relative
        if training: # at TRAINING: then random dropout
            return self.last_random < self.training_rate
        # at INFERENCE: step function as BPE schedule
        elif self.inference_step == -1: # --> all denoising chain with APE (absolute)
            return torch.zeros_like(t, dtype=torch.bool)
        elif self.inference_step == 0: # --> all denoising chain with RPE (relative)
            return torch.ones_like(t, dtype=torch.bool)
        else: # --> BPE with binary step function. Step from APE to RPE at "self.inference_step"
            return ~(t > self.max_steps - self.inference_step)
    
    def use_bias(self, t: torch.Tensor, training: bool) -> torch.Tensor:
        # function that returns True if we should use the absolute bias (only when using multi-segments **inference**)
        assert (t[0] == t).all(), "Bias from mixed schedule only supported when using same timestep for all batch elements: " + str(t)
        return ~self.get_schedule_fn(t[0], training) # if APE --> use bias to limit attention to the each subsequence

    def get_time_weights(self, t: torch.Tensor, training: bool) -> torch.Tensor:
        # 0 --> absolute
        # 1 --> relative
        return self.get_schedule_fn(t, training).to(torch.int32)

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
        self.action_emb = kargs.get('action_emb', None)
        
        self.music_dim = 35 * 90 #baseline feats
        # pos_dim = 3
        # rot_dim = self.njoints * self.nfeats  # 24 joints, 6dof
        # self.input_feats = pos_dim + rot_dim + 4
    
        # output_feats = self.input_feats

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_drop_prob = kargs.get('cond_drop_prob')

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        
        self.max_seq_att = kargs.get('max_seq_att', 1024)
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
        self.process_cond_input = [nn.Linear(2*self.latent_dim, self.latent_dim) for _ in range(self.num_layers)]
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        self.motion_mask = kargs['motion_mask']
        self.hist_frames = kargs['hist_frames']
        
        use_rotary = True
        
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )
        
        #!here
        # #########################
        # self.use_chunked_att = kargs.get('use_chunked_att', False)
        # bpe_training_rate = kargs.get('bpe_training_ratio', 0.5) # for training, we dropout with prob 50% --> APE vs RPE
        # bpe_inference_step = kargs.get('bpe_denoising_step', None)
        # diffusion_steps = kargs.get('diffusion_steps', None)
        # self.bpe_schedule = BPE_Schedule(bpe_training_rate, bpe_inference_step, diffusion_steps)
        # ws = kargs.get('rpe_horizon', -1) # Max attention horizon
        # self.local_attn_window_size = 200 if ws == -1 else ws
        # print("[Training] RPE/APE rate:", bpe_training_rate)
        # print(f"[Inference] BPE switch from APE to RPE at denoising step {bpe_inference_step}/{diffusion_steps}.")
        # print("Local attention window size:", self.local_attn_window_size)

        # self.seqTransEncoder = ContinuousTransformerWrapper(
        #     dim_in = self.latent_dim, dim_out = self.latent_dim,
        #     emb_dropout = self.dropout,
        #     max_seq_len = self.max_seq_att,
        #     use_abs_pos_emb = True,
        #     absolute_bpe_schedule = self.bpe_schedule, # bpe schedule for absolute embeddings (APE)
        #     attn_layers = Encoder(
        #         dim = self.latent_dim,
        #         depth = self.num_layers,
        #         heads = self.num_heads,
        #         ff_mult = int(np.round(self.ff_size / self.latent_dim)), # 2 for MDM hyper params
        #         layer_dropout = self.dropout, cross_attn_tokens_dropout = 0,

        #         # ======== FLOWMDM ========
        #         custom_layers=('A', 'f'), # A --> PCCAT
        #         custom_query_fn = self.process_cond_input, # function that merges the condition into the query --> PCCAT dense layer (see Fig. 3)
        #         attn_max_attend_past = self.local_attn_window_size,
        #         attn_max_attend_future = self.local_attn_window_size,
        #         # ======== RELATIVE POSITIONAL EMBEDDINGS ========
        #         rotary_pos_emb = True, # rotary embeddings
        #         rotary_bpe_schedule = self.bpe_schedule, # bpe schedule for rotary embeddings (RPE)
        #     )
        # )
        # ########################
        
        if self.hist_frames > 0:
            # self.hist_frames = 5
            self.seperation_token = nn.Parameter(torch.randn(latent_dim))
            self.skel_embedding = nn.Linear(self.njoints, self.latent_dim)
        
        if self.arch == 'inpainting':
            # print("TRANS_ENC init")
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
        
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'music' in self.cond_mode:
                self.embed_music = nn.Linear(self.music_dim, self.latent_dim)
                # print("EMBED MUSIC")

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)
        
        self.refine_final_layer1 = nn.Linear(self.latent_dim, self.nfeats)
        self.refine_input_projection1 = nn.Linear(self.nfeats, self.latent_dim)
        self.refine_cond_projection1 = nn.Linear(48, self.latent_dim)
        self.refine_norm_cond1 = nn.LayerNorm(latent_dim)
        
        refine_decoderstack = nn.ModuleList([])
        for _ in range(1):
            refine_decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        self.refine_seqTransDecoder1 = DecoderLayerStack(refine_decoderstack)

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
        bs, d = cond.shape
        # print("Original Shape:", cond.shape)
        # print("Reshaped Shape:", cond.shape)
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            # print("Mask Shape:", mask.shape)
            masked_cond = cond * (1. - mask)
            # print("Masked Cond Shape:", masked_cond.shape)
            return masked_cond
        else:
            return cond

    def get_rcond(self, output):
        # with torch.no_grad():
            from data_loaders.d2m.quaternion import ax_from_6v, quat_slerp
            if self.normalizer is not None:
                output = self.normalizer.unnormalize(output)
                
            device = "cuda" if torch.cuda.is_available() else "cpu"
            smpl = SMPLSkeleton(device=device)
            
            bs, nframes, nfeats = output.shape
            # bs, njoints, nfeats, nframes = output.shape
            # output = output.reshape(bs, njoints*nfeats, nframes).permute(0, 2, 1)
            model_contact, output = torch.split(
                output, (4, output.shape[2] - 4), dim=2
            )
            model_x = output[:, :, :3]
            model_q = ax_from_6v(output[:, :, 3:].reshape(bs, nframes, -1, 6))
            
            joints3d = smpl(model_q, model_x)[:,:,:22,:]
            B,T,J,_ = joints3d.shape
            l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
            relevant_joints = [l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx]
            pred_foot = joints3d[:, :, relevant_joints, :]          # B,T,J,4
            foot_vel = torch.zeros_like(pred_foot)
            foot_vel[:, :-1] = (
                pred_foot[:, 1:, :, :] - pred_foot[:, :-1, :, :]
            )  # (N, S-1, 4, 3)
            foot_y_ankle = pred_foot[:, :, :2, 1]
            foot_y_toe = pred_foot[:, :, 2:, 1]
            fc_mask_ankle = torch.unsqueeze((foot_y_ankle <= (-1.2+0.012)), dim=3).repeat(1, 1, 1, 3)
            fc_mask_teo = torch.unsqueeze((foot_y_toe <= (-1.2+0.05)), dim=3).repeat(1, 1, 1, 3)
            contact_lable = torch.cat([fc_mask_ankle, fc_mask_teo], dim=2).int().to(output).reshape(B, T, -1)

            contact_toe_thresh, contact_ankle_thresh, contact_vel_thresh = -1.2+0.08, -1.2+0.015, 0.3 / 30           # 30 is fps
            contact_score_toe = torch.sigmoid((contact_toe_thresh - pred_foot[:, :, :2, 1])/contact_toe_thresh*5) * \
            torch.sigmoid((contact_vel_thresh - torch.norm(foot_vel[:, :, :2, [0, 2]], dim=-1))/contact_vel_thresh*5)
            contact_score_toe = torch.unsqueeze(contact_score_toe, dim=3).repeat(1, 1, 1, 3)
            contact_score_ankle = torch.sigmoid((contact_ankle_thresh - pred_foot[:, :, 2:, 1])/contact_ankle_thresh*5) * \
            torch.sigmoid((contact_vel_thresh - torch.norm(foot_vel[:, :, 2:, [0, 2]], dim=-1))/contact_vel_thresh*5)
            contact_score_ankle = torch.unsqueeze(contact_score_ankle, dim=3).repeat(1, 1, 1, 3)
            contact_score = torch.cat([contact_score_ankle, contact_score_ankle], dim = -2).reshape(B, T, -1)
            r_cond = torch.cat([contact_lable, contact_score, pred_foot.reshape(B,T,-1), foot_vel.reshape(B,T,-1)], dim = -1) 
            return r_cond
    
    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        
        time_emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        
        music_emb = self.embed_music(y['music'])
        emb = time_emb + self.mask_cond(music_emb, force_mask=force_mask)

        x = self.input_process(x)
        
        if self.arch == 'inpainting':
            mask = lengths_to_mask(y['lengths'], x.device)
            if y.get('hframes', None) == None:
                token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
                aug_mask = torch.cat((token_mask, mask), 1)
                xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
                xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
                if self.motion_mask:
                    output = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
                else:
                    output = self.seqTransEncoder(xseq)[1:]
            else:
                token_mask = torch.ones((bs, 2 + 2*self.hist_frames), dtype=bool, device=x.device)
                aug_mask = torch.cat((token_mask, mask), 1)
                sep_token = torch.tile(self.seperation_token, (bs,)).reshape(bs, -1).unsqueeze(0)
                hframes = y['hframes'].squeeze(2).permute(2, 0, 1) #TODO find out the diff 
                hframes_emb = self.skel_embedding(hframes)
                fut_frames = y['fut_frames'].squeeze(2).permute(2, 0, 1)
                fut_frames_emb = self.skel_embedding(fut_frames)
                # hframes_emb = hframes_emb.permute(1, 0, 2) # [5 b dim]
                xseq = torch.cat((emb, hframes_emb, fut_frames_emb, sep_token, x), axis=0)
                # TODO add attention mask
                xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
                output = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)[2 + 2*self.hist_frames:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        
        # output = output.reshape(bs, njoints*nfeats, nframes).permute(0, 2, 1)
        output = output.permute(1, 0, 2)
        
        r_output = self.refine_input_projection1(output)
        r_cond = self.get_rcond(output)
        r_cond = self.refine_cond_projection1(r_cond)
        rc = torch.cat((r_cond, time_emb), dim=-2)
        r_cond_tokens = self.refine_norm_cond1(rc)
        refine_output = self.refine_seqTransDecoder1(r_output, r_cond_tokens, emb)
        refine_output = self.refine_final_layer1(refine_output)     # / 10
        out = output + refine_output
        
        out = out.reshape(nframes, bs, njoints, nfeats)
        out = out.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        
        return out

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
        bs, njoints, nfeats, nframes = x.shape
        # print("INPUT PROCESS: ", bs, njoints, nfeats, nframes)
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

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
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output