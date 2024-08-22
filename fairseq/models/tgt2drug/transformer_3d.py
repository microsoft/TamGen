import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

from typing import Any, Dict
from fairseq import checkpoint_utils, utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerModel, Embedding, LayerNorm, TransformerDecoder
from fairseq.modules import MultiheadAttention

from fairseq.modules import MLP, PositionalEmbedding, TransformerEncoderLayer, DistanceTransformerLayer
from ...data import Dictionary


def get_random_rotation_3d(pos):
    random_quaternions = torch.randn(4).to(pos)
    random_quaternions = random_quaternions / random_quaternions.norm(
        dim=-1, keepdim=True)
    return torch.einsum("bkj,ij->bki", pos,
                        quaternion_to_rotation_matrix(random_quaternions))


def quaternion_to_rotation_matrix(quaternion):
    q0 = quaternion[0]
    q1 = quaternion[1]
    q2 = quaternion[2]
    q3 = quaternion[3]
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22],
                       dim=-1).reshape(3, 3)


def upgrade_state_dict_with_gpt_weights(
    state_dict: Dict[str, Any],
    pretrained_gpt_checkpoint: str,
    our_dict: Dictionary,
    gpt_dict: Dictionary = None,
) -> Dict[str, Any]:

    if not os.path.exists(pretrained_gpt_checkpoint):
        raise IOError(
            "Model file not found: {}".format(pretrained_gpt_checkpoint))

    state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_gpt_checkpoint)
    with torch.no_grad():
        for key in state["model"].keys():
            gpt_param = state['model'][key]
            if key.startswith("decoder"):
                # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
                subkey = key[len("decoder") + 1:]
                if subkey in state_dict.keys():
                    my_param = state_dict[subkey]
                    if subkey.endswith('embed_tokens.weight'):
                        if my_param.shape[1] != gpt_param.shape[1]:
                            raise RuntimeError(
                                f'embed_tokens.weight embedding size mismatch: '
                                f'{my_param.shape[1]} != {gpt_param.shape[1]}')
                        if gpt_dict is None:
                            # No dict path, assume using same dictionary.
                            if my_param.shape[0] != gpt_param.shape[0]:
                                raise RuntimeError(
                                    f'embed_tokens.weight vocabulary size mismatch: '
                                    f'{my_param.shape[0]} != {gpt_param.shape[0]}'
                                )
                            my_param.copy_(gpt_param)
                        else:
                            # Copy by symbol indices.
                            roberta_symbol_indices = [
                                gpt_dict.index(sym) for sym in our_dict.symbols
                            ]
                            my_param.copy_(gpt_param[roberta_symbol_indices])
                    else:
                        my_param.copy_(gpt_param)

                # fairseq-0.10.2's Multihead attention has seperated in_proj to 3 parts, \
                # thus concat q_proj, k_proj, v_proj for this version of fairseq
                else:
                    if "q_proj" in subkey:
                        dim = state["model"][key].size()[0]
                        state_dict[subkey.replace(
                            'q_proj.', 'in_proj_')][:dim] = state["model"][key]
                    elif "k_proj" in subkey:
                        dim = state["model"][key].size()[0]
                        state_dict[subkey.replace(
                            'k_proj.',
                            'in_proj_')][dim:dim * 2] = state["model"][key]
                    elif "v_proj" in subkey:
                        dim = state["model"][key].size()[0]
                        state_dict[subkey.replace(
                            'v_proj.',
                            'in_proj_')][dim * 2:] = state["model"][key]
    return state_dict

@register_model('transformer_3d')
class Transformer3D(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)


    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)

        # Add model-specific arguments to the parser for simple 3d Transformer
        parser.add_argument('--add-noise', default=False, action='store_true',
                            help='add noise to the coordinates of the input targets')
        parser.add_argument('--std', type=float, default=0.1, metavar='D',
                            help='noise std')
        parser.add_argument('--dist-attn', default=False, action='store_true',
                            help='use distance attention')
        parser.add_argument('--dist-decay', type=float, default=3000, metavar='D',
                            help='dist decay')
        parser.add_argument('--dist-embedding', default=False, action='store_true',
                            help='')
        parser.add_argument('--move-to-origin', default=False, action='store_true',)
        parser.add_argument('--random-rotation', default=False, action='store_true',)
        parser.add_argument('--mlp-layers', type=int, default=0,)
        parser.add_argument('--concat', default=False, action='store_true',)
        parser.add_argument(
            "--pretrained-model-checkpoint",
            type=str,
            metavar="STR",
            help="If specified, load the pretrained model to initialize the overall model",
        )
        parser.add_argument(
            "--pretrained-gpt-checkpoint",
            type=str,
            metavar="STR",
            help="If specified, load the gpt model to initialize transformer decoder",
        )
        parser.add_argument(
            "--fix-decoder-params",
            default=False,
            action='store_true',
            help='fix the parameters of the decoder'
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, tgt_dict, decoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if hasattr(args, "pretrained_model_checkpoint"):
            print(f"| loading pretrained model from {args.pretrained_model_checkpoint}")
            return cls(encoder, decoder).load_from_pretrained(
                args.pretrained_model_checkpoint,
                )
        else:
            return cls(encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, encoder_embed_tokens, tgt_dict, decoder_embed_tokens):
        if args.vae:
            return Encoder3D(args, src_dict, tgt_dict, encoder_embed_tokens, decoder_embed_tokens)
        return Transformer3DEncoder(args, src_dict, encoder_embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, decoder_embed_tokens):
        return TransformerDecoderFromPretrained(args, tgt_dict, decoder_embed_tokens)

    def load_from_pretrained(self, checkpoint_fn):
        state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_fn)
        self.load_state_dict(state['model'])
        return self
    
    def extract_encoder_features(self, src_tokens, src_lengths, **kwargs):
        return self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        try:
            return decoder_out, encoder_out['latent_mean'], encoder_out['latent_logstd']
        except:
            return decoder_out


class VAEEncoder(FairseqEncoder):
    def __init__(self, args, src_dict, tgt_dict, encoder_embed_tokens,
                 decoder_embed_tokens):
        super().__init__(src_dict)
        self.src_padding_idx = src_dict.pad()
        self.tgt_padding_idx = tgt_dict.pad()
        self.left_pad_source = args.left_pad_source
        self.left_pad_target = args.left_pad_target
        self.dim = args.encoder_embed_dim
        self.dropout = args.dropout
        
        self.use_src_coord = args.use_src_coord
        self.use_tgt_coord = args.use_tgt_coord
        self.src_token_embed = encoder_embed_tokens
        
        if self.use_src_coord:
            self.src_pos_embed = MLP(
                    3, [4 * self.dim] * args.mlp_layers + [self.dim], dropout=self.dropout, layernorm_before=True,
                ) if args.mlp_layers > 0 else nn.Linear(3, self.dim)
        else:
            self.src_pos_embed = None
        self.tgt_token_embed = decoder_embed_tokens
        if self.use_tgt_coord:
            self.tgt_pos_embed = MLP(
                    3, [4 * self.dim] * args.mlp_layers + [self.dim], dropout=self.dropout, layernorm_before=True,
                ) if args.mlp_layers > 0 else nn.Linear(3, self.dim)
        else:
            self.tgt_pos_embed = None

        self.proj = nn.Linear(args.decoder_embed_dim, self.dim)

        self.embed_scale = math.sqrt(self.dim)
        self.add_noise = args.add_noise
        self.std = args.std
        self.move_to_origin = args.move_to_origin
        self.random_rotation = args.random_rotation

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args) for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(self.dim)
        else:
            self.layer_norm = None

        self.out_proj = nn.Linear(self.dim, self.dim * 2)

    def forward(self,
                src_tokens,
                tgt_tokens,
                src_coord=None,
                tgt_coord=None,
                **kwargs):

        # src embed
        x_src = self.embed_scale * self.src_token_embed(src_tokens)
        src_mask = src_tokens.eq(self.src_padding_idx)
        tgt_mask = tgt_tokens.eq(self.tgt_padding_idx)

        if self.use_src_coord and (src_coord is not None):
            if self.move_to_origin:
                if src_mask.any():
                    mean = torch.sum(src_coord, dim=1) / (~src_mask).sum(
                        dim=1).unsqueeze(-1)
                else:
                    mean = torch.mean(src_coord, dim=1)
                src_coord = (src_coord - mean.unsqueeze(1))
            if self.random_rotation and self.training:
                src_coord = get_random_rotation_3d(src_coord)
            if self.add_noise and self.training:
                noise = self.std * torch.randn_like(src_coord)
                src_coord = src_coord + noise
            x_src += self.src_pos_embed(src_coord)

        # tgt embed
        x_tgt = self.proj(self.embed_scale * self.tgt_token_embed(tgt_tokens))

        if self.use_tgt_coord and (tgt_coord is not None):
            if self.add_noise and self.training:
                noise = self.std * torch.randn_like(tgt_coord)
                tgt_coord = tgt_coord + noise
            x_tgt += self.tgt_pos_embed(tgt_coord)

        x = torch.cat((x_src, x_tgt), dim=1)
        encoder_padding_mask = torch.cat((src_mask, tgt_mask), dim=1)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        x = x[:src_mask.shape[1]]
        if self.layer_norm:
            x = self.layer_norm(x)

        x = self.out_proj(x)

        return x


class Transformer3DEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout

        self.dim = embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        if args.use_src_coord:
            self.embed_coords = MLP(
                    3, [4 * self.dim] * args.mlp_layers + [self.dim], dropout=self.dropout, layernorm_before=True,
                ) if args.mlp_layers > 0 else nn.Linear(3, self.dim)
            self.add_noise = args.add_noise
            self.std = args.std
            self.move_to_origin = args.move_to_origin
            self.random_rotation = args.random_rotation
            self.gen_coord_noise = args.gen_coord_noise
            self.gen_rot = args.gen_rot
        else:
            self.embed_coords = None
            self.add_noise = False
            self.std = args.std
            self.move_to_origin = False
            self.random_rotation = False
            self.gen_coord_noise = False
            self.gen_rot = False

        self.dist_attn = args.dist_attn
        self.layers = nn.ModuleList([])
        if self.dist_attn:
            self.layers.extend([
                DistanceTransformerLayer(args)
                for i in range(args.encoder_layers)
            ])
        else:
            self.layers.extend([
                TransformerEncoderLayer(args)
                for i in range(args.encoder_layers)
            ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, src_tokens, src_lengths, src_coord=None):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        if src_coord is not None and self.embed_coords is not None:
            if self.move_to_origin:
                if encoder_padding_mask is not None:
                    mean = torch.sum(src_coord, dim=1) / src_lengths.unsqueeze(-1)
                else:
                    mean = torch.mean(src_coord, dim=1)
                src_coord = (src_coord - mean.unsqueeze(1))
            if self.random_rotation and (self.training or (not self.training and self.gen_rot)):
                src_coord = get_random_rotation_3d(src_coord)
            if self.add_noise and (self.training or (not self.training and self.gen_coord_noise)):
                noise = self.std * torch.randn_like(src_coord)
                src_coord = src_coord + noise
            x += self.embed_coords(src_coord)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # encoder layers
        if self.dist_attn and src_coord is not None:
            for layer in self.layers:
                x = layer(x, src_coord, encoder_padding_mask)
            x = x.transpose(0, 1)
        else:
            x = x.transpose(0, 1)
            for layer in self.layers:
                x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

class Encoder3D(FairseqEncoder):

    def __init__(self, args, src_dict, tgt_dict, encoder_embed_tokens, decoder_embed_tokens):
        super().__init__(src_dict)
        self.main_encoder = Transformer3DEncoder(args, src_dict, encoder_embed_tokens)
        self.vae_encoder = VAEEncoder(args, src_dict, tgt_dict, encoder_embed_tokens, decoder_embed_tokens)
        self.sample_beta = args.sample_beta
        self.concat = args.concat
        self.gen_vae = args.gen_vae

    def forward(self, src_tokens, src_lengths, src_coord=None, tgt_tokens=None, tgt_coord=None):
        main_encoder_out = self.main_encoder(src_tokens, src_lengths, src_coord)
        if (self.training or (not self.training and self.gen_vae)) and (tgt_tokens is not None):
            vae_encoder_out = self.vae_encoder(src_tokens, tgt_tokens, src_coord=src_coord, tgt_coord=tgt_coord)
            mean, logstd = torch.chunk(vae_encoder_out, chunks=2, dim=-1)
            if self.training:
                z = mean + torch.exp(logstd) * torch.randn_like(mean)
            else:
                z = mean + torch.exp(logstd) * torch.randn_like(mean) * self.sample_beta
        else:
            z = torch.randn_like(main_encoder_out['encoder_out']) * self.sample_beta
            mean, logstd = None, None
        return {
            'encoder_out': torch.cat((main_encoder_out['encoder_out'], z), dim=-1) if self.concat else main_encoder_out['encoder_out'] + z,
            'encoder_padding_mask': main_encoder_out['encoder_padding_mask'],
            'latent_mean': mean,
            'latent_logstd': logstd,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

class TransformerDecoderFromPretrained(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if hasattr(args, "pretrained_gpt_checkpoint"):
            pretrained_gpt_checkpoint = args.pretrained_gpt_checkpoint
            gpt_loaded_state_dict = upgrade_state_dict_with_gpt_weights(
                state_dict=self.state_dict(),
                pretrained_gpt_checkpoint=pretrained_gpt_checkpoint,
                our_dict=dictionary,
                gpt_dict=Dictionary.load(
                    os.path.join(os.path.dirname(pretrained_gpt_checkpoint),
                                 'dict.txt')),
            )
            self.load_state_dict(gpt_loaded_state_dict, strict=True)
            if args.fix_decoder_params:
                for param in self.parameters():
                    param.requires_grad = False
                for layer in self.layers:
                    for param in layer.encoder_attn.parameters():
                        param.requires_grad = True

        for layer in self.layers:
            layer.encoder_attn = MultiheadAttention(
                layer.embed_dim,
                args.decoder_attention_heads,
                kdim=args.encoder_embed_dim if not args.concat else 2*args.encoder_embed_dim ,
                vdim=args.encoder_embed_dim if not args.concat else 2*args.encoder_embed_dim ,
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )

@register_model_architecture('transformer_3d', 'transformer_3d')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
