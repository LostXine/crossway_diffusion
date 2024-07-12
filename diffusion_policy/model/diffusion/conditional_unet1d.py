from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import math

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward_w_mid(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        mid = x.clone()
        # x.shape is 64 x 2048 (down_dims[-1]) x 4
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x, mid

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        x, mid = self.forward_w_mid(sample=sample,
                                    timestep=timestep,
                                    local_cond=local_cond,
                                    global_cond=global_cond)
        return x

from diffusion_policy.model.diffusion.conv2d_components import (Upsample2d, Conv2dBlock)


class ResidualBlock2D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.Sequential(
            Conv2dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv2dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        )
        
        # make sure dimensions compatible
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        '''
            x : [ batch_size x in_channels x H x W ]
            returns:
            out : [ batch_size x out_channels x H x W ]
        '''
        out = self.blocks(x)
        out = out + self.residual_conv(x)
        return out
    

class ConditionalUnet1DwDecTypeA(ConditionalUnet1D):
    # select the first token, and split the channels into 4 pixels
    def __init__(self, 
        input_dim,
        n_obs_steps: int,
        obs_shape_meta: dict,
        decode_unet_feat=True,
        decode_pe_dim=64,
        decode_resolution=2,
        decode_dims=[64, 128],
        decode_low_dim_dims=[4, 2],
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        
        ):
        super().__init__(
            input_dim,
            local_cond_dim,
            global_cond_dim,
            diffusion_step_embed_dim,
            down_dims,
            kernel_size,
            n_groups,
            cond_predict_scale
        )
        
        all_dims = [input_dim] + list(decode_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        decode_low_dim_dims.append(1)
        
        self.obs_shape_meta = obs_shape_meta
        self.n_obs_steps = n_obs_steps
        decs = {}
        
        self.decode_unet_feat = decode_unet_feat
        # if this is true, decode the output from the mid module of unet
        # else, decode global cond
        if decode_unet_feat:
            dec_chn = down_dims[-1] // self.n_obs_steps // decode_resolution // decode_resolution 
            assert dec_chn == decode_dims[-1], f'The decoder dim must match the dim of UNet + PE ({dec_chn} vs {decode_dims[-1]})'
            self.decode_resolution = decode_resolution
            self.decode_pe_dim = decode_pe_dim
        else:
            raise NotImplementedError('Haven\'t finished yet.')
        
        for key, obs in obs_shape_meta.items():
            if obs['type'] == 'rgb':
                # if crop_shape is not None:
                #    self.rgb_shape = crop_shape
                # else:
                if 'recon_shape' in obs:
                    self.rgb_shape = obs['recon_shape'][1:]
                else:
                    self.rgb_shape = obs['shape'][1:] # C H W => H W
                    
                start_dim = decode_dims[0]
                
                rgb_dec = nn.ModuleList([])
                for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
                    is_last = ind >= (len(in_out) - 1)
                    rgb_dec.append(nn.Sequential(
                        ResidualBlock2D(
                            dim_out + decode_pe_dim, dim_in,
                            kernel_size=kernel_size, n_groups=n_groups),
                        ResidualBlock2D(
                            dim_in, dim_in, kernel_size=kernel_size, n_groups=n_groups),
                        Upsample2d(dim_in) if not is_last else nn.Identity()
                    ))
                
                final_conv = nn.Sequential(
                    Conv2dBlock(start_dim, start_dim, kernel_size=kernel_size),
                    nn.Conv2d(start_dim, obs['shape'][0], 1),
                )
                decs[key] = rgb_dec
                decs[key + '_final'] = final_conv
            
            elif obs['type'] == 'low_dim':
                low_dim_dims = list(zip(decode_low_dim_dims[:-1], decode_low_dim_dims[1:]))
                obs_dim = obs['shape'][0]
                ld_dec = [
                    nn.Linear(down_dims[-1] // self.n_obs_steps, obs_dim * decode_low_dim_dims[0]),
                ]
                for dim_in, dim_out in low_dim_dims:
                    ld_dec.append(nn.Mish())
                    ld_dec.append(nn.Linear(obs_dim * dim_in, obs_dim * dim_out))
                    
                decs[key] = nn.ModuleList(ld_dec)
                
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        self.decs = nn.ModuleDict(decs)
        

    def generate_positional_embedding(self, x, dim):
        b, _, h, w = x.shape
        hidx = torch.linspace(-1, 1, steps=h)
        widx = torch.linspace(-1, 1, steps=w)
        freq = dim // 4
        sh = [(2 ** i) * torch.pi * hidx for i in range(freq)]
        sw = [(2 ** i) * torch.pi * widx for i in range(freq)]
        
        grids = [torch.stack(torch.meshgrid(hi, wi, indexing='ij'), axis=0) for hi, wi in zip(sh, sw)]
        
        phases = torch.concat(grids, 0)
        assert phases.shape == (dim // 2, h, w)
        pe = torch.concat([torch.sin(phases), torch.cos(phases)], axis=0)
        bpe = pe.unsqueeze(0).repeat(b, 1, 1, 1)
        bpe = bpe.to(x.device)
        return bpe


    def forward_and_recon(self,
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        local_cond=None, global_cond=None, **kwargs):
        x_res, mid = self.forward_w_mid(sample=sample,
                                timestep=timestep,
                                local_cond=local_cond,
                                global_cond=global_cond)
        
        # mid.shape is 64 x 2048 (actually down_dims[-1]) x 4
        # let's select the first time step
       
        if self.decode_unet_feat:
            mid_pre = mid[:, :, 0] # shape: 64 x 2048
            mid_ld = einops.rearrange(mid_pre, 'b (t c) -> (b t) c', t=self.n_obs_steps)
            b, c = mid_ld.shape
            mid_rgb = mid_ld.reshape(b, -1, self.decode_resolution, self.decode_resolution)
            # 128 (64 x 2) x 128 x 2 x 2
            
            recons = {}
            for key, obs in self.obs_shape_meta.items():
                if obs['type'] == 'rgb':
                    # generate a grid
                    n_upsamples = len(self.decs[key])
                    h_res = self.rgb_shape[0] // (2 ** n_upsamples)
                    w_res = self.rgb_shape[1] // (2 ** n_upsamples)
                    
                    h_scale = math.ceil(h_res / self.decode_resolution)
                    w_scale = math.ceil(w_res / self.decode_resolution)
                    
                    x = mid_rgb.repeat(1, 1, h_scale, w_scale)
                    x = x[:, :, :h_res, :w_res]
                    
                    for resnet in self.decs[key]:
                        x = torch.cat((x, self.generate_positional_embedding(x, self.decode_pe_dim)), dim=1)
                        x = resnet(x)
                    x = self.decs[key + '_final'](x)
                    recons[key] = x
                    
                elif obs['type'] == 'low_dim':
                    x = mid_ld
                    for layer in self.decs[key]:
                        x = layer(x)
                    recons[key] = x
                else:
                    raise RuntimeError(f"Unsupported obs type: {type}")
                
        else:
            raise NotImplementedError('Haven\'t finished yet.')
                    
        return x_res, recons
    
class ConditionalUnet1DwDecTypeD(ConditionalUnet1DwDecTypeA):
    # no intersection, directly reconstruct from the state representation
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # merge the gap between input state and dec
        self.mid_proj = nn.Sequential(
            nn.Linear(kwargs['global_cond_dim'] // self.n_obs_steps, kwargs['decode_dims'][-1]),
            nn.ReLU()
        )
        print('Initialize TypeD')

    def forward_and_recon(self,
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        local_cond=None, global_cond=None, **kwargs):
        x_res, mid = self.forward_w_mid(sample=sample,
                                timestep=timestep,
                                local_cond=local_cond,
                                global_cond=global_cond)
        
        if self.decode_unet_feat:
            assert len(global_cond.shape) == 2
            
            mid_id = einops.rearrange(global_cond, 'b (t c) -> (b t) c', t=self.n_obs_steps)
            mid_id = self.mid_proj(mid_id).unsqueeze(-1).unsqueeze(-1)
            mid_rgb = mid_id.repeat(1, 1, self.decode_resolution, self.decode_resolution)
            # 128 (64 x 2) x 128 x 2 x 2
            
            recons = {}
            for key, obs in self.obs_shape_meta.items():
                if obs['type'] == 'rgb':
                    # generate a grid
                    n_upsamples = len(self.decs[key])
                    h_res = self.rgb_shape[0] // (2 ** n_upsamples)
                    w_res = self.rgb_shape[1] // (2 ** n_upsamples)
                    
                    h_scale = math.ceil(h_res / self.decode_resolution)
                    w_scale = math.ceil(w_res / self.decode_resolution)
                    
                    x = mid_rgb.repeat(1, 1, h_scale, w_scale)
                    x = x[:, :, :h_res, :w_res]
                    
                    for resnet in self.decs[key]:
                        x = torch.cat((x, self.generate_positional_embedding(x, self.decode_pe_dim)), dim=1)
                        x = resnet(x)
                    x = self.decs[key + '_final'](x)
                    recons[key] = x
        else:
            raise NotImplementedError('Haven\'t finished yet.')
                    
        return x_res, recons
    

class ConditionalUnet1DwDecTypeC(ConditionalUnet1DwDecTypeA):
    # 4 tokens as 4 pixels, project all channels to smaller channel numbers
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        proj = dict()
        for key, obs in self.obs_shape_meta.items():
            if obs['type'] == 'rgb':
                proj[key] = nn.Sequential(
                    nn.Linear(kwargs['down_dims'][-1] // self.n_obs_steps, kwargs['decode_dims'][-1]),
                    nn.Mish()
                )
        self.proj = nn.ModuleDict(proj)
        print('Initialize TypeC')
    
    def forward_and_recon(self,
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        local_cond=None, global_cond=None, **kwargs):
        x_res, mid = self.forward_w_mid(sample=sample,
                                timestep=timestep,
                                local_cond=local_cond,
                                global_cond=global_cond)
        
        if self.decode_unet_feat:
            # mid.shape: 64 x 2048 x 4
            mid_ld = einops.rearrange(mid, 'b (t c) h -> (b t) c h', t=self.n_obs_steps) # 128 x 1024 x 4
            b, c, h = mid_ld.shape
            mid_rgb = mid_ld.reshape(b, c, self.decode_resolution, self.decode_resolution)
            # 128 (64 x 2) x 128 x 2 x 2
            
            recons = {}
            for key, obs in self.obs_shape_meta.items():
                if obs['type'] == 'rgb':
                    # generate a grid
                    n_upsamples = len(self.decs[key])
                    h_res = self.rgb_shape[0] // (2 ** n_upsamples)
                    w_res = self.rgb_shape[1] // (2 ** n_upsamples)
                    
                    h_scale = math.ceil(h_res / self.decode_resolution)
                    w_scale = math.ceil(w_res / self.decode_resolution)
                    
                    x = mid_rgb.repeat(1, 1, h_scale, w_scale)
                    x = x[:, :, :h_res, :w_res]
                    x = einops.rearrange(x, 'b c h w -> (b h w) c')
                    
                    for i, resnet in enumerate(self.decs[key]):
                        if i == 0:
                            x = self.proj[key](x) # match the dimension
                            x = einops.rearrange(x, '(b h w) c -> b c h w', h=h_res, w=w_res)
                        x = torch.cat((x, self.generate_positional_embedding(x, self.decode_pe_dim)), dim=1)
                        x = resnet(x)
                    x = self.decs[key + '_final'](x)
                    recons[key] = x
                    
                elif obs['type'] == 'low_dim':
                    x = torch.mean(mid_ld, dim=-1) # 128 x 1024 x 4 => 128 x 1024
                    for layer in self.decs[key]:
                        x = layer(x)
                    recons[key] = x
                else:
                    raise RuntimeError(f"Unsupported obs type: {type}")
                
        else:
            raise NotImplementedError('Haven\'t finished yet.')
                    
        return x_res, recons
    

class ConditionalUnet1DwDecTypeB(ConditionalUnet1DwDecTypeA):
    # 4 tokens as 4 pixels, project part of channels to smaller channel numbers
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.keep_ratio = 2
        proj = dict()
        for key, obs in self.obs_shape_meta.items():
            if obs['type'] == 'rgb':
                proj[key] = nn.Sequential(
                    nn.Linear(kwargs['down_dims'][-1] // self.n_obs_steps // self.keep_ratio, kwargs['decode_dims'][-1]),
                    nn.Mish()
                )
            elif obs['type'] == 'low_dim':
                obs_dim = obs['shape'][0]
                self.decs[key][0] = nn.Linear(kwargs['down_dims'][-1] // self.n_obs_steps // self.keep_ratio, obs_dim * kwargs['decode_low_dim_dims'][0])
                
        self.proj = nn.ModuleDict(proj)
        print('Initialize TypeB')
        
                
    
    def forward_and_recon(self,
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        local_cond=None, global_cond=None, **kwargs):
        x_res, mid = self.forward_w_mid(sample=sample,
                                timestep=timestep,
                                local_cond=local_cond,
                                global_cond=global_cond)
        
        if self.decode_unet_feat:
            # mid.shape: 64 x 2048 x 4
            dis = mid.shape[1] // self.keep_ratio
            mid_ld = einops.rearrange(mid[:, :dis], 'b (t c) h -> (b t) c h', t=self.n_obs_steps) # 128 x 512 x 4
            b, c, h = mid_ld.shape
            mid_rgb = mid_ld.reshape(b, c, self.decode_resolution, self.decode_resolution)
            # 128 (64 x 2) x 128 x 2 x 2
            
            recons = {}
            for key, obs in self.obs_shape_meta.items():
                if obs['type'] == 'rgb':
                    # generate a grid
                    n_upsamples = len(self.decs[key])
                    h_res = self.rgb_shape[0] // (2 ** n_upsamples)
                    w_res = self.rgb_shape[1] // (2 ** n_upsamples)
                    
                    h_scale = math.ceil(h_res / self.decode_resolution)
                    w_scale = math.ceil(w_res / self.decode_resolution)
                    
                    x = mid_rgb.repeat(1, 1, h_scale, w_scale)
                    x = x[:, :, :h_res, :w_res]
                    x = einops.rearrange(x, 'b c h w -> (b h w) c')
                    
                    for i, resnet in enumerate(self.decs[key]):
                        if i == 0:
                            x = self.proj[key](x) # match the dimension
                            x = einops.rearrange(x, '(b h w) c -> b c h w', h=h_res, w=w_res)
                        x = torch.cat((x, self.generate_positional_embedding(x, self.decode_pe_dim)), dim=1)
                        x = resnet(x)
                    x = self.decs[key + '_final'](x)
                    recons[key] = x
                    
                elif obs['type'] == 'low_dim':
                    x = torch.mean(mid_ld, dim=-1) # 128 x 1024 x 4 => 128 x 1024
                    for layer in self.decs[key]:
                        x = layer(x)
                    recons[key] = x
                else:
                    raise RuntimeError(f"Unsupported obs type: {type}")
                
        else:
            raise NotImplementedError('Haven\'t finished yet.')
                    
        return x_res, recons
    
    

