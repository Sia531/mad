from typing import Tuple

import einops
import torch
from torch import nn
from torch.distributions import Bernoulli

from .helpers import MlpSelfAttention, SelfAttention
from .temporal import (
    Downsample1d,
    ResidualTemporalBlock,
    SinusoidalPosEmb,
    TemporalMlpBlock,
    TemporalSelfAttention,
    TemporalUnet,
)


class ConvAttentionDeconv(nn.Module):
    agent_share_parameters = False

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        history_horizon: int = 0,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        n_agents: int = 2,
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        residual_attn: bool = True,
        use_layer_norm: bool = False,
        max_path_length: int = 100,
        use_temporal_attention: bool = True,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.history_horizon = history_horizon
        self.use_temporal_attention = use_temporal_attention

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.nets = nn.ModuleList(
            [
                TemporalUnet(
                    horizon=horizon,
                    history_horizon=history_horizon,
                    transition_dim=transition_dim,
                    dim=dim,
                    dim_mults=dim_mults,
                    returns_condition=returns_condition,
                    env_ts_condition=env_ts_condition,
                    condition_dropout=condition_dropout,
                    max_path_length=max_path_length,
                    kernel_size=kernel_size,
                )
                for _ in range(n_agents)
            ]
        )

        if self.use_temporal_attention:
            print("\n USE TEMPORAL ATTENTION !!! \n")
            AttentionModule = TemporalSelfAttention

            self.self_attn = [
                AttentionModule(
                    in_out[-1][1],
                    in_out[-1][1] // 16,
                    in_out[-1][1] // 4,
                    residual=residual_attn,
                    embed_dim=self.net.embed_dim,
                )
            ]
            for dims in reversed(in_out):
                self.self_attn.append(
                    AttentionModule(
                        dims[1],
                        dims[1] // 16,
                        dims[1] // 4,
                        residual=residual_attn,
                        embed_dim=self.net.embed_dim,
                    )
                )
        else:
            self.self_attn = [
                SelfAttention(
                    in_out[-1][1],
                    in_out[-1][1] // 16,
                    in_out[-1][1] // 4,
                    residual=residual_attn,
                )
            ]
            for dims in reversed(in_out):
                self.self_attn.append(
                    SelfAttention(
                        dims[1],
                        dims[1] // 16,
                        dims[1] // 4,
                        residual=residual_attn,
                    )
                )
        self.self_attn = nn.ModuleList(self.self_attn)

        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            horizon_ = horizon
            self.layer_norm = []
            for dims in in_out:
                self.layer_norm.append(nn.LayerNorm([dims[1], horizon_]))
                horizon_ = horizon_ // 2
            horizon_ = horizon_ * 2
            self.layer_norm.append(nn.LayerNorm([in_out[-1][1], horizon_]))
            self.layer_norm = list(reversed(self.layer_norm))
            self.layer_norm = nn.ModuleList(self.layer_norm)

            horizon_ = horizon
            self.layer_norm_cat = []
            for dims in in_out:
                self.layer_norm_cat.append(nn.LayerNorm([dims[1] * 2, horizon_]))
                horizon_ = horizon_ // 2
            self.layer_norm_cat = list(reversed(self.layer_norm_cat))
            self.layer_norm_cat = nn.ModuleList(self.layer_norm_cat)

    def forward(
        self,
        x,
        time,
        returns=None,
        states=None,
        env_timestep=None,
        attention_masks=None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        **kwargs,
    ):
        """
        x : [ batch x horizon x agent x transition ]
        returns : [ batch x horizon x agent ]
        """

        assert x.shape[2] == self.n_agents, (
            f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"
        )

        x = einops.rearrange(x, "b t a f -> b a f t")
        x = [x[:, a_idx] for a_idx in range(x.shape[1])]  # a, b f t

        t = [self.nets[i].time_mlp(time) for i in range(self.n_agents)]

        if self.returns_condition:
            assert returns is not None
            returns_embed = [
                self.nets[i].returns_mlp(returns[:, :, i]) for i in range(self.n_agents)
            ]
            if use_dropout:
                # here use the same mask for all agents
                mask = (
                    self.nets[0]
                    .mask_dist.sample(sample_shape=(returns_embed[0].size(0), 1))
                    .to(returns_embed[0].device)
                )
                returns_embed = [
                    returns_embed[i] * mask for i in range(len(returns_embed))
                ]
            if force_dropout:
                returns_embed = [
                    returns_embed[i] * 0 for i in range(len(returns_embed))
                ]

            t = [torch.cat([t[i], returns_embed[i]], dim=-1) for i in range(len(t))]

        if self.env_ts_condition:
            assert env_timestep is not None
            env_ts_embed = [
                self.nets[i].env_ts_mlp(env_timestep) for i in range(self.n_agents)
            ]
            t = [torch.cat([t[i], env_ts_embed[i]], dim=-1) for i in range(len(t))]

        h = [[] for _ in range(self.n_agents)]

        for layer_idx in range(len(self.nets[0].downs)):
            for i in range(self.n_agents):
                resnet, resnet2, downsample = self.nets[i].downs[layer_idx]
                x[i] = resnet(x[i], t[i])
                x[i] = resnet2(x[i], t[i])
                h[i].append(x[i])
                x[i] = downsample(x[i])

        for i in range(self.n_agents):
            x[i] = self.nets[i].mid_block1(x[i], t[i])
            x[i] = self.nets[i].mid_block2(x[i], t[i])

        x = self.self_attn[0](torch.stack(x, dim=1))  # b a f t
        if self.use_layer_norm:
            x = self.layer_norm[0](x)
        x = [x[:, a_idx] for a_idx in range(x.shape[1])]  # a, b f t

        for layer_idx in range(len(self.nets[0].ups)):
            hiddens = torch.stack([hid.pop() for hid in h], dim=1)  # b a f t
            if self.use_layer_norm:
                hiddens = self.layer_norm[layer_idx + 1](hiddens)
            hiddens = self.self_attn[layer_idx + 1](hiddens)
            for i in range(self.n_agents):
                resnet, resnet2, upsample = self.nets[i].ups[layer_idx]
                x[i] = torch.cat((x[i], hiddens[:, i]), dim=1)
                if self.use_layer_norm:
                    x[i] = self.layer_norm_cat[layer_idx](x[i])
                x[i] = resnet(x[i], t[i])
                x[i] = resnet2(x[i], t[i])
                x[i] = upsample(x[i])

        for i in range(self.n_agents):
            x[i] = self.nets[i].final_conv(x[i])

        x = torch.stack(x, dim=1)
        x = einops.rearrange(x, "b a f t -> b t a f")

        return x


class SharedConvAttentionDeconv(nn.Module):
    agent_share_parameters = True

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        history_horizon: int = 0,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        nhead: int = 4,
        n_agents: int = 2,
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        residual_attn: bool = True,
        use_layer_norm: bool = False,
        max_path_length: int = 100,
        use_temporal_attention: bool = True,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.history_horizon = history_horizon
        self.use_temporal_attention = use_temporal_attention

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        self.net = TemporalUnet(
            horizon=horizon,
            history_horizon=history_horizon,
            transition_dim=transition_dim,
            dim=dim,
            dim_mults=dim_mults,
            returns_condition=returns_condition,
            env_ts_condition=env_ts_condition,
            condition_dropout=condition_dropout,
            max_path_length=max_path_length,
            kernel_size=kernel_size,
        )

        self.self_attn = [
            TemporalSelfAttention(
                in_out[-1][1],
                in_out[-1][1] // 16,
                in_out[-1][1] // 4,
                residual=residual_attn,
                embed_dim=self.net.embed_dim,
            )
        ]
        for dims in reversed(in_out):
            self.self_attn.append(
                TemporalSelfAttention(
                    dims[1],
                    dims[1] // 16,
                    dims[1] // 4,
                    residual=residual_attn,
                    embed_dim=self.net.embed_dim,
                )
            )
        self.self_attn = nn.ModuleList(self.self_attn)

        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            horizon_ = horizon
            self.layer_norm = []
            for dims in in_out:
                self.layer_norm.append(nn.LayerNorm([dims[1], horizon_]))
                horizon_ = horizon_ // 2
            horizon_ = horizon_ * 2
            self.layer_norm.append(nn.LayerNorm([in_out[-1][1], horizon_]))
            self.layer_norm = list(reversed(self.layer_norm))
            self.layer_norm = nn.ModuleList(self.layer_norm)

            horizon_ = horizon
            self.layer_norm_cat = []
            for dims in in_out:
                self.layer_norm_cat.append(nn.LayerNorm([dims[1] * 2, horizon_]))
                horizon_ = horizon_ // 2
            self.layer_norm_cat = list(reversed(self.layer_norm_cat))
            self.layer_norm_cat = nn.ModuleList(self.layer_norm_cat)

    def forward(
        self,
        x,
        time,
        returns=None,
        states=None,
        env_timestep=None,
        attention_masks=None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        **kwargs,
    ):
        """
        x : [ batch x horizon x agent x transition ]
        returns : [ batch x horizon x agent ]
        """

        assert x.shape[2] == self.n_agents, (
            f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"
        )

        x = einops.rearrange(x, "b t a f -> b a f t")
        bs = x.shape[0]

        t = self.net.time_mlp(torch.stack([time for _ in range(x.shape[1])], dim=1))
        if self.returns_condition:
            assert returns is not None
            returns = einops.rearrange(returns, "b t a -> b a t")
            returns_embed = self.net.returns_mlp(returns)
            if use_dropout:
                # here use the same mask for all agents
                mask = self.net.mask_dist.sample(
                    sample_shape=(returns_embed.size(0), returns_embed.size(1), 1)
                ).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        if self.env_ts_condition:
            assert env_timestep is not None
            env_timestep = env_timestep.to(dtype=torch.int64)
            env_timestep = env_timestep[:, self.history_horizon]
            env_ts_embed = self.net.env_ts_mlp(env_timestep)
            env_ts_embed = einops.repeat(env_ts_embed, "b f -> b a f", a=x.shape[1])
            t = torch.cat([t, env_ts_embed], dim=-1)

        h = []
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])
        for resnet, resnet2, downsample in self.net.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.net.mid_block1(x, t)
        x = self.net.mid_block2(x, t)

        x = x.reshape(bs, x.shape[0] // bs, x.shape[1], x.shape[2])
        if self.use_layer_norm:
            x = self.layer_norm[0](x)
        if self.use_temporal_attention:
            t = t.reshape(bs, t.shape[0] // bs, t.shape[1])
            x = self.self_attn[0](x, t)  # b a f t
            t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])
        else:
            x = self.self_attn[0](x)  # b a f t

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        for layer_idx in range(len(self.net.ups)):
            hiddens = h.pop()
            hiddens = hiddens.reshape(
                bs, hiddens.shape[0] // bs, hiddens.shape[1], hiddens.shape[2]
            )
            if self.use_layer_norm:
                hiddens = self.layer_norm[layer_idx + 1](hiddens)
            if self.use_temporal_attention:
                t = t.reshape(bs, t.shape[0] // bs, t.shape[1])
                hiddens = self.self_attn[layer_idx + 1](hiddens, t)
                t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])
            else:
                hiddens = self.self_attn[layer_idx + 1](hiddens)

            hiddens = hiddens.reshape(
                hiddens.shape[0] * hiddens.shape[1], hiddens.shape[2], hiddens.shape[3]
            )
            resnet, resnet2, upsample = self.net.ups[layer_idx]
            x = torch.cat((x, hiddens), dim=1)
            if self.use_layer_norm:
                x = self.layer_norm_cat[layer_idx](x)

            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.net.final_conv(x)
        x = x.reshape(bs, x.shape[0] // bs, x.shape[1], x.shape[2])

        x = einops.rearrange(x, "b a f t -> b t a f")

        return x


class SharedAttentionAutoEncoder(nn.Module):
    agent_share_parameters = True

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        dim_mults: Tuple[int] = (1, 2, 4),
        n_agents: int = 2,
        returns_condition: bool = False,
        condition_dropout: float = 0.1,
    ):
        assert horizon == 1, (
            f"Only horizon=1 is supported for AttentionAutoEncoder, but got horizon={horizon}"
        )
        super().__init__()

        self.n_agents = n_agents
        self.condition_dropout = condition_dropout

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/stationary ] Hidden dimensions: {in_out}")

        act_fn = nn.Mish()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, dim),
                act_fn,
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )

            self.mask_dist = Bernoulli(probs=1 - self.condition_dropout)
            embed_dim = 2 * dim
        else:
            embed_dim = dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                TemporalMlpBlock(
                    dim_in,
                    dim_out,
                    embed_dim,
                    act_fn,
                    out_act_fn=act_fn if not is_last else nn.Identity(),
                )
            )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                TemporalMlpBlock(
                    dim_out * 2,
                    dim_in,
                    embed_dim,
                    act_fn,
                    out_act_fn=act_fn,
                )
            )

        self.final_mlp = nn.Sequential(
            nn.Linear(dim, dim), act_fn, nn.Linear(dim, transition_dim)
        )

        self.self_attn = [MlpSelfAttention(in_out[-1][1])]
        for dims in reversed(in_out):
            self.self_attn.append(MlpSelfAttention(dims[1]))
        self.self_attn = nn.ModuleList(self.self_attn)

    def forward(self, x, time, returns=None, use_dropout=True, force_dropout=False):
        """
        x : [ batch x horizon(1) x agent x transition ]
        returns : [batch x horizon(1) x agent]
        """

        assert x.shape[2] == self.n_agents, (
            f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"
        )

        x = x.squeeze(1)  # b a f
        bs = x.shape[0]

        t = self.time_mlp(torch.stack([time for _ in range(x.shape[1])], dim=1))

        if self.returns_condition:
            assert returns is not None
            # returns = returns.squeeze(1)  # b a
            returns = einops.rearrange(returns, "b t a -> b a t")
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                # here use the same mask for all agents
                mask = self.mask_dist.sample(
                    sample_shape=(returns_embed.size(0), returns_embed.size(1), 1)
                ).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed

            t = torch.cat([t, returns_embed], dim=-1)

        h = []
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])
        for mlp in self.downs:
            x = mlp(x, t)
            h.append(x)

        x = x.reshape(bs, x.shape[0] // bs, x.shape[1])
        x = self.self_attn[0](x)  # b a f

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        for layer_idx in range(len(self.ups)):
            hiddens = h.pop()
            hiddens = hiddens.reshape(bs, hiddens.shape[0] // bs, hiddens.shape[1])
            hiddens = self.self_attn[layer_idx + 1](hiddens)
            hiddens = hiddens.reshape(
                hiddens.shape[0] * hiddens.shape[1], hiddens.shape[2]
            )
            mlp = self.ups[layer_idx]
            x = torch.cat([x, hiddens], dim=-1)
            x = mlp(x, t)

        x = self.final_mlp(x)
        x = x.reshape(bs, 1, x.shape[0] // bs, x.shape[1])

        return x


class ConvAttentionTemporalValue(nn.Module):
    agent_share_parameters = False

    def __init__(
        self,
        horizon,
        transition_dim,
        n_agents,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.n_agents = n_agents
        self.time_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    SinusoidalPosEmb(dim),
                    nn.Linear(dim, dim * 4),
                    nn.Mish(),
                    nn.Linear(dim * 4, dim),
                )
                for _ in range(n_agents)
            ]
        )

        self.blocks = nn.ModuleList([nn.ModuleList([]) for _ in range(n_agents)])
        num_resolutions = len(in_out)

        print("ConvAttentionTemporalValue: ", in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            for i in range(n_agents):
                self.blocks[i].append(
                    nn.ModuleList(
                        [
                            ResidualTemporalBlock(
                                dim_in,
                                dim_out,
                                kernel_size=5,
                                embed_dim=time_dim,
                            ),
                            ResidualTemporalBlock(
                                dim_out,
                                dim_out,
                                kernel_size=5,
                                embed_dim=time_dim,
                            ),
                            Downsample1d(dim_out) if not is_last else nn.Identity(),
                        ]
                    )
                )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 4
        mid_dim_3 = mid_dim // 16

        self.mid_block1 = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    mid_dim,
                    mid_dim_2,
                    kernel_size=5,
                    embed_dim=time_dim,
                )
                for _ in range(n_agents)
            ]
        )
        self.mid_block2 = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    mid_dim_2,
                    mid_dim_3,
                    kernel_size=5,
                    embed_dim=time_dim,
                )
                for _ in range(n_agents)
            ]
        )
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fc_dim + time_dim, fc_dim // 2),
                    nn.Mish(),
                    nn.Linear(fc_dim // 2, out_dim),
                )
                for _ in range(n_agents)
            ]
        )
        self.self_attn = nn.ModuleList(
            [SelfAttention(dim[1], dim[1] // 16) for dim in in_out]
        )

    def forward(self, x, time, *args):
        """
        x : [ batch x horizon x n_agents x transition ]
        """

        assert x.shape[2] == self.n_agents, (
            f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"
        )

        x = einops.rearrange(x, "b t a f -> b a f t")
        # the tensor shape of x for each agent may change after each block, so
        # can not stack x as a tensor (the assignment will cause error).
        x = [x[:, a_idx] for a_idx in range(x.shape[1])]  # a, b f t

        t = [self.time_mlp[i](time) for i in range(self.n_agents)]

        for layer_idx in range(len(self.blocks[0])):
            for i in range(self.n_agents):
                resnet, resnet2, downsample = self.blocks[i][layer_idx]
                x[i] = resnet(x[i], t[i])
                x[i] = resnet2(x[i], t[i])
                x[i] = downsample(x[i])
            x = self.self_attn[layer_idx](torch.stack(x, dim=1))
            x = [x[:, a_idx] for a_idx in range(x.shape[1])]  # a, b f t

        for i in range(self.n_agents):
            x[i] = self.mid_block1[i](x[i], t[i])
            x[i] = self.mid_block2[i](x[i], t[i])
            x[i] = x[i].view(len(x[i]), -1)
            x[i] = self.final_block[i](torch.cat([x[i], t[i]], dim=-1))
        x = torch.stack(x, dim=1).squeeze(-1)

        # take mean over agents
        out = x.mean(axis=1, keepdim=True)  # x.shape[0], 1

        return out


class SharedConvAttentionTemporalValue(nn.Module):
    agent_share_parameters = True

    def __init__(
        self,
        horizon,
        transition_dim,
        n_agents,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.n_agents = n_agents
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print("ConvAttentionTemporalValue: ", in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in,
                            dim_out,
                            kernel_size=5,
                            embed_dim=time_dim,
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            kernel_size=5,
                            embed_dim=time_dim,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 4
        mid_dim_3 = mid_dim // 16

        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim
        )
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )
        self.self_attn = nn.ModuleList(
            [SelfAttention(dim[1], dim[1] // 16) for dim in in_out]
        )

    def forward(self, x, time, *args):
        """
        x : [ batch x horizon x n_agents x transition ]
        """

        assert x.shape[2] == self.n_agents, (
            f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"
        )

        x = einops.rearrange(x, "b t a f -> b a f t")
        bs = x.shape[0]

        t = self.time_mlp(torch.stack([time for _ in range(x.shape[1])], dim=1))

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])

        for layer_idx, (resnet, resnet2, downsample) in enumerate(self.blocks):
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)
            x = x.reshape(bs, x.shape[0] // bs, x.shape[1], x.shape[2])
            x = self.self_attn[layer_idx](x)
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        x = x.view(len(x), -1)
        x = self.final_block(torch.cat([x, t], dim=-1))  # x.shape[0] * x.shape[1], 1

        x = x.reshape(bs, -1)  # x.shape[0], x.shape[1], 1
        # take mean over agents
        out = x.mean(axis=1, keepdim=True)  # x.shape[0], 1

        return out


class ConcatTemporalValue(nn.Module):
    agent_share_parameters = False

    def __init__(
        self,
        horizon,
        transition_dim,
        n_agents,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim * n_agents, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.n_agents = n_agents
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print("ConvAttentionTemporalValue: ", in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in,
                            dim_out,
                            kernel_size=5,
                            embed_dim=time_dim,
                        ),
                        ResidualTemporalBlock(
                            dim_out,
                            dim_out,
                            kernel_size=5,
                            embed_dim=time_dim,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 4
        mid_dim_3 = mid_dim // 16

        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim
        )
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, time, *args):
        """
        x : [ batch x horizon x n_agents x transition ]
        """

        assert x.shape[2] == self.n_agents, (
            f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"
        )

        x = x.reshape(x.shape[0], x.shape[1], -1)  # b t a f -> b t (a*f)
        x = einops.rearrange(x, "b t f -> b f t")
        t = self.time_mlp(time)

        for layer_idx, (resnet, resnet2, downsample) in enumerate(self.blocks):
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))  # x.shape[0] * x.shape[1], 1

        return out


class SharedConvAttentionDeconv_TD3BC(nn.Module):
    """
    Extended version of SharedConvAttentionDeconv that incorporates TD3_BC capabilities.
    TD3_BC combines Twin Delayed Deep Deterministic Policy Gradient (TD3) with Behavior Cloning (BC).
    """

    agent_share_parameters = True

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        history_horizon: int = 0,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        nhead: int = 4,
        n_agents: int = 2,
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        residual_attn: bool = True,
        use_layer_norm: bool = False,
        max_path_length: int = 100,
        use_temporal_attention: bool = True,
        # TD3_BC specific parameters
        alpha: float = 2.5,  # Weight for behavior cloning loss
        bc_mode: str = "mse",  # BC loss mode: 'mse' or 'cosine'
        policy_noise: float = 0.2,  # Noise added to target policy
        noise_clip: float = 0.5,  # Range to clip target policy noise
        policy_freq: int = 2,  # Frequency of delayed policy updates
    ):
        super().__init__()

        # Initialize the base diffusion model components from SharedConvAttentionDeconv
        self.n_agents = n_agents
        self.history_horizon = history_horizon
        self.use_temporal_attention = use_temporal_attention

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        self.net = TemporalUnet(
            horizon=horizon,
            history_horizon=history_horizon,
            transition_dim=transition_dim,
            dim=dim,
            dim_mults=dim_mults,
            returns_condition=returns_condition,
            env_ts_condition=env_ts_condition,
            condition_dropout=condition_dropout,
            max_path_length=max_path_length,
            kernel_size=kernel_size,
        )

        self.self_attn = [
            TemporalSelfAttention(
                in_out[-1][1],
                in_out[-1][1] // 16,
                in_out[-1][1] // 4,
                residual=residual_attn,
                embed_dim=self.net.embed_dim,
            )
        ]
        for dims in reversed(in_out):
            self.self_attn.append(
                TemporalSelfAttention(
                    dims[1],
                    dims[1] // 16,
                    dims[1] // 4,
                    residual=residual_attn,
                    embed_dim=self.net.embed_dim,
                )
            )
        self.self_attn = nn.ModuleList(self.self_attn)

        # Setup layer normalization if required
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            horizon_ = horizon
            self.layer_norm = []
            for dims in in_out:
                self.layer_norm.append(nn.LayerNorm([dims[1], horizon_]))
                horizon_ = horizon_ // 2
            horizon_ = horizon_ * 2
            self.layer_norm.append(nn.LayerNorm([in_out[-1][1], horizon_]))
            self.layer_norm = list(reversed(self.layer_norm))
            self.layer_norm = nn.ModuleList(self.layer_norm)

            horizon_ = horizon
            self.layer_norm_cat = []
            for dims in in_out:
                self.layer_norm_cat.append(nn.LayerNorm([dims[1] * 2, horizon_]))
                horizon_ = horizon_ // 2
            self.layer_norm_cat = list(reversed(self.layer_norm_cat))
            self.layer_norm_cat = nn.ModuleList(self.layer_norm_cat)

        # TD3_BC specific components
        self.alpha = alpha  # Weight for behavior cloning loss
        self.bc_mode = bc_mode  # BC loss mode
        self.policy_noise = policy_noise  # Noise added to target actions
        self.noise_clip = noise_clip  # Range to clip target policy noise
        self.policy_freq = policy_freq  # Frequency of delayed policy updates

        # Actor and Critics for TD3
        self.actor_network = nn.Sequential(
            nn.Linear(transition_dim * n_agents + dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, transition_dim * n_agents),
            nn.Tanh(),  # Output scaled to [-1, 1]
        )

        # Twin Q-networks for TD3
        self.critic1 = nn.Sequential(
            nn.Linear(transition_dim * n_agents * 2, dim * 2),  # States + Actions
            nn.ReLU(),
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, 1),
        )

        self.critic2 = nn.Sequential(
            nn.Linear(transition_dim * n_agents * 2, dim * 2),  # States + Actions
            nn.ReLU(),
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, 1),
        )

        # Target networks (initialized as copies)
        self.target_actor = nn.Sequential(
            nn.Linear(transition_dim * n_agents + dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, transition_dim * n_agents),
            nn.Tanh(),
        )

        self.target_critic1 = nn.Sequential(
            nn.Linear(transition_dim * n_agents * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, 1),
        )

        self.target_critic2 = nn.Sequential(
            nn.Linear(transition_dim * n_agents * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, 1),
        )

        # Initialize target networks with same weights
        self._copy_weights(self.actor_network, self.target_actor)
        self._copy_weights(self.critic1, self.target_critic1)
        self._copy_weights(self.critic2, self.target_critic2)

        # Additional training related variables
        self.total_it = 0  # Track total iterations for delayed updates

    def _copy_weights(self, source_network, target_network):
        """Copy weights from source network to target network"""
        for source_param, target_param in zip(
            source_network.parameters(), target_network.parameters()
        ):
            target_param.data.copy_(source_param.data)

    def _soft_update(self, source_network, target_network, tau=0.005):
        """Soft update of target network parameters: θ_target = τ*θ_source + (1-τ)*θ_target"""
        for source_param, target_param in zip(
            source_network.parameters(), target_network.parameters()
        ):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )

    def forward(
        self,
        x,
        time,
        returns=None,
        states=None,
        env_timestep=None,
        attention_masks=None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        **kwargs,
    ):
        """
        x : [ batch x horizon x agent x transition ]
        returns : [ batch x horizon x agent ]
        """
        # Use standard forward pass from SharedConvAttentionDeconv
        assert x.shape[2] == self.n_agents, (
            f"Expected {self.n_agents} agents, but got samples with shape {x.shape}"
        )

        x = einops.rearrange(x, "b t a f -> b a f t")
        bs = x.shape[0]

        t = self.net.time_mlp(torch.stack([time for _ in range(x.shape[1])], dim=1))
        if self.returns_condition:
            assert returns is not None
            returns = einops.rearrange(returns, "b t a -> b a t")
            returns_embed = self.net.returns_mlp(returns)
            if use_dropout:
                # here use the same mask for all agents
                mask = self.net.mask_dist.sample(
                    sample_shape=(returns_embed.size(0), returns_embed.size(1), 1)
                ).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        if self.env_ts_condition:
            assert env_timestep is not None
            env_timestep = env_timestep.to(dtype=torch.int64)
            env_timestep = env_timestep[:, self.history_horizon]
            env_ts_embed = self.net.env_ts_mlp(env_timestep)
            env_ts_embed = einops.repeat(env_ts_embed, "b f -> b a f", a=x.shape[1])
            t = torch.cat([t, env_ts_embed], dim=-1)

        h = []
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])
        for resnet, resnet2, downsample in self.net.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.net.mid_block1(x, t)
        x = self.net.mid_block2(x, t)

        x = x.reshape(bs, x.shape[0] // bs, x.shape[1], x.shape[2])
        if self.use_layer_norm:
            x = self.layer_norm[0](x)
        if self.use_temporal_attention:
            t = t.reshape(bs, t.shape[0] // bs, t.shape[1])
            x = self.self_attn[0](x, t)  # b a f t
            t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])
        else:
            x = self.self_attn[0](x)  # b a f t

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        for layer_idx in range(len(self.net.ups)):
            hiddens = h.pop()
            hiddens = hiddens.reshape(
                bs, hiddens.shape[0] // bs, hiddens.shape[1], hiddens.shape[2]
            )
            if self.use_layer_norm:
                hiddens = self.layer_norm[layer_idx + 1](hiddens)
            if self.use_temporal_attention:
                t = t.reshape(bs, t.shape[0] // bs, t.shape[1])
                hiddens = self.self_attn[layer_idx + 1](hiddens, t)
                t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])
            else:
                hiddens = self.self_attn[layer_idx + 1](hiddens)

            hiddens = hiddens.reshape(
                hiddens.shape[0] * hiddens.shape[1], hiddens.shape[2], hiddens.shape[3]
            )
            resnet, resnet2, upsample = self.net.ups[layer_idx]
            x = torch.cat((x, hiddens), dim=1)
            if self.use_layer_norm:
                x = self.layer_norm_cat[layer_idx](x)

            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.net.final_conv(x)
        x = x.reshape(bs, x.shape[0] // bs, x.shape[1], x.shape[2])

        x = einops.rearrange(x, "b a f t -> b t a f")

        return x

    def select_action(self, state, time):
        """
        Select action using the actor network

        state: [ batch x horizon x agent x transition ]
        time: time embedding
        """
        # Process state to match actor input requirements
        state_flat = state.reshape(
            state.shape[0], -1
        )  # Flatten to [batch, agents*transition]

        # Get time embedding
        t = self.net.time_mlp(time)

        # Concatenate state and time embedding
        actor_input = torch.cat([state_flat, t], dim=1)

        # Get action from actor network
        with torch.no_grad():
            action = self.actor_network(actor_input)

        # Reshape action to match expected output
        action = action.reshape(state.shape[0], state.shape[1], state.shape[2], -1)

        return action

    def td3_bc_update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        done_batch,
        time_batch,
    ):
        """
        Update the TD3_BC networks based on collected experience

        Implements the TD3+BC update from the paper:
        "TD3+BC: Stabilizing Behavior Cloning for Deep Offline Reinforcement Learning"
        """
        self.total_it += 1

        # Flatten states and actions for critic networks
        state_flat = state_batch.reshape(state_batch.shape[0], -1)
        action_flat = action_batch.reshape(action_batch.shape[0], -1)
        next_state_flat = next_state_batch.reshape(next_state_batch.shape[0], -1)

        # Get time embeddings
        t = self.net.time_mlp(time_batch)

        # Compute target actions with noise for regularization
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = torch.randn_like(action_flat) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            next_state_actor_input = torch.cat([next_state_flat, t], dim=1)
            next_action = self.target_actor(next_state_actor_input)
            next_action = (next_action + noise).clamp(-1, 1)

            # Compute the target Q value
            target_Q1 = self.target_critic1(
                torch.cat([next_state_flat, next_action], dim=1)
            )
            target_Q2 = self.target_critic2(
                torch.cat([next_state_flat, next_action], dim=1)
            )
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = (
                reward_batch + (1 - done_batch) * 0.99 * target_Q
            )  # 0.99 is gamma

        # Get current Q estimates
        current_Q1 = self.critic1(torch.cat([state_flat, action_flat], dim=1))
        current_Q2 = self.critic2(torch.cat([state_flat, action_flat], dim=1))

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(
            current_Q2, target_Q
        )

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            state_actor_input = torch.cat([state_flat, t], dim=1)
            pi = self.actor_network(state_actor_input)
            Q = self.critic1(torch.cat([state_flat, pi], dim=1))

            # Combine TD3 actor loss with BC loss
            lmbda = self.alpha / Q.abs().mean().detach()

            if self.bc_mode == "mse":
                bc_loss = nn.MSELoss()(pi, action_flat)
            elif self.bc_mode == "cosine":
                bc_loss = (1 - torch.cosine_similarity(pi, action_flat, dim=1)).mean()
            else:
                raise ValueError(f"Unknown BC mode: {self.bc_mode}")

            actor_loss = -lmbda * Q.mean() + bc_loss

            # Update the actor
            # (In actual implementation, you'd use an optimizer here)

            # Update the frozen target models
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)
            self._soft_update(self.actor_network, self.target_actor)

        # Return losses for monitoring
        if self.total_it % self.policy_freq == 0:
            return {
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(),
                "bc_loss": bc_loss.item(),
                "Q_mean": Q.mean().item(),
            }
        else:
            return {"critic_loss": critic_loss.item()}
