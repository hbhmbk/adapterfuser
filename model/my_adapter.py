from torch import nn
import torch
from einops import rearrange, repeat
import torch

from collections import OrderedDict
from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
)
import torch.nn.functional as F
import math
from .utils import SpatialTransformer
from diffusers.models import unet_2d_condition
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            from torch.nn import MaxUnpool2d
            self.op = MaxUnpool2d(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)



class Linear(nn.Module):
    def __init__(self, temb_channels, out_channels):
        super(Linear, self).__init__()
        self.linear = nn.Linear(temb_channels, out_channels)

    def forward(self, x):
        return self.linear(x)








class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, down, ksize=3, sk=True, use_conv=True, enable_timestep=False,
                 temb_channels=None, use_norm=False):
        super().__init__()
        self.use_norm = use_norm
        self.enable_timestep = enable_timestep
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, 1)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        if use_norm:
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=out_c, eps=1e-6, affine=True)
        # self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down:
            self.down_opt = Downsample(out_c, use_conv=use_conv)

        if enable_timestep:
            self.timestep_proj = Linear(temb_channels, out_c)

    def forward(self, x, output_size=None, temb=None):


        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        if temb is not None:
            temb = self.timestep_proj(temb)[:, :, None, None]
            h = h + temb
        if self.use_norm:
            h = self.norm1(h)
        h = self.act(h)
        if self.down == True:
            x = self.down_opt(x)
        # h = self.block2(h)
        if self.down_opt:
            return self.down_opt(x+h)
        else:
            return h + x

class AdapterResnetBlock(nn.Module):
    r"""
    An `AdapterResnetBlock` is a helper model that implements a ResNet-like block.

    Parameters:
        channels (`int`):
            Number of channels of AdapterResnetBlock's input and output.
    """

    def __init__(self, channels: int,temb_channels=320):
        super().__init__()
        self.block1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.timestep_proj = Linear(temb_channels,channels)

    def forward(self, x: torch.Tensor,temb=None) -> torch.Tensor:
        r"""
        This method takes input tensor x and applies a convolutional layer, ReLU activation, and another convolutional
        layer on the input tensor. It returns addition with the input tensor.
        """

        h = self.act(self.block1(x))
        if temb is not None:
            temb = self.timestep_proj(temb)[:, :, None, None]
            h = h + temb
        h = self.block2(h)

        return h + x


class AdapterBlock(nn.Module):
    r"""
    An AdapterBlock is a helper model that contains multiple ResNet-like blocks. It is used in the `FullAdapter` and
    `FullAdapterXL` models.

    Parameters:
        in_channels (`int`):
            Number of channels of AdapterBlock's input.
        out_channels (`int`):
            Number of channels of AdapterBlock's output.
        num_res_blocks (`int`):
            Number of ResNet blocks in the AdapterBlock.
        down (`bool`, *optional*, defaults to `False`):
            Whether to perform downsampling on AdapterBlock's input.
    """

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int, down: bool = False):
        super().__init__()

        self.downsample = None
        if down:
            # self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.downsample = nn.Conv2d(in_channels,in_channels, kernel_size=3, stride=2, padding=1)

        self.in_conv = None
        if in_channels != out_channels:
            self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # self.resnets = nn.Sequential(
        #     *[AdapterResnetBlock(out_channels) for _ in range(num_res_blocks)],
        # )
        self.resnets=AdapterResnetBlock(out_channels)
    def forward(self, x: torch.Tensor,temb) -> torch.Tensor:
        r"""
        This method takes tensor x as input and performs operations downsampling and convolutional layers if the
        self.downsample and self.in_conv properties of AdapterBlock model are specified. Then it applies a series of
        residual blocks to the input tensor.
        """
        if self.downsample is not None:
            x = self.downsample(x)

        if self.in_conv is not None:
            x = self.in_conv(x)

        x = self.resnets(x,temb=temb)

        return x


class control_adapterblock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int, down: bool = False,n_head=8):
        super().__init__()
        self.resblock=AdapterBlock(in_channels,out_channels,num_res_blocks,down)
        self.attn=SpatialTransformer(out_channels,n_heads=n_head,d_head=72,context_dim=out_channels)
    def forward(self,x,context,temb=None):
        x=self.resblock(x,temb=temb)
        x=self.attn(x,context)
        return x



class Adapter_XL(nn.Module):
    r"""
    See [`T2IAdapter`] for more information.
    """

    def __init__(
        self,
        in_channel: int = 4,
        in_channels= [320, 320, 640, 1280],
        out_channels=[320,640,1280,1280],
        down=[False,False,True,False],
        conds=['sketch', 'openpose', 'lineart', 'depth'],
        enable_timestep=True
    ):
        super().__init__()
        self.conds=conds
        self.conv_in = nn.Conv2d(in_channel, in_channels[0], kernel_size=3,stride=2,padding=1)
        block=[]
        out=[]
        task_embeding_proj = []
        for i in range(len(in_channels)):
            block.append(control_adapterblock(in_channels[i],out_channels[i],1,down[i]))
            out.append((self.make_zero_conv(out_channels[i])))
            task_embeding_proj.append(nn.Sequential(nn.Linear(320, out_channels[i]), nn.SiLU()))

        self.block=nn.ModuleList(block)
        self.zero_out=nn.ModuleList(out)
        self.task_embeding_proj=nn.ModuleList(task_embeding_proj)
        self.task_embeding = nn.Parameter(torch.randn(16, 320))
        if enable_timestep:
            a = 320

            time_embed_dim = a
            self.time_proj = Timesteps(a, True, 0)
            timestep_input_dim = a

            self.time_embedding = TimestepEmbedding(
                timestep_input_dim,
                time_embed_dim,
                act_fn='silu',
                post_act_fn=None,
                cond_proj_dim=None,
            )
    def make_zero_conv(self, channels):

        return zero_module(nn.Conv2d(channels, channels, 1, padding=0))
    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)


    def forward(self,latent,context,temb=None,keep_cond=None):
        b,c,h,w=latent.shape
        temb = self.time_proj(temb)  # b, 320
        temb = self.time_embedding(temb)
        influence_functions={}
        feature=[]
        for cond_name in context.keys():
            cond_index = self.conds.index(cond_name)
            x=self.conv_in(latent)
            influence_function = []
            task_embeding=self.task_embeding[cond_index][None,:]
            task_embeding=torch.cat([task_embeding]*b)
            
            for i in range(len(context[cond_name])):
                task_embeding1=self.task_embeding_proj[i](task_embeding)
                x=self.block[i](x,context[cond_name][i]+task_embeding1[:,:,None,None],temb)
                out=self.zero_out[i](x)
                if keep_cond is not None:
                        indices_to_zero = torch.nonzero(keep_cond[cond_name] == 1).squeeze()
                        out[indices_to_zero] = -1e12
                influence_function.append(out)
                if i == len(context[cond_name])-1:
                    feature.append(x)
            influence_functions[cond_name]=influence_function

        stack_feature=[[],[],[],[]]
        stack_score=[[],[],[],[]]

        for cond_name in influence_functions.keys():

            for i in range(len(influence_functions[cond_name])):
                stack_score[i].append(influence_functions[cond_name][i])
                stack_feature[i].append(context[cond_name][i])

        for i in range (len(stack_score)):
            stack_score[i]=torch.stack(stack_score[i],dim=1)
            stack_score[i]=F.sigmoid(stack_score[i])
            # stack_score[i]=F.softmax(stack_score[i],dim=1)
            stack_feature[i]=torch.stack(stack_feature[i],dim=1)
            stack_feature[i]=stack_score[i]*stack_feature[i]
            stack_feature[i]=stack_feature[i].sum(dim=1)

        return stack_feature

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


if __name__ == "__main__":

    # model=SpatialTransformer(in_channels=640,n_heads=8,d_head=64,context_dim=640).to('cuda')
    # num_parameters = sum(p.numel() for p in model.parameters())
    # input=torch.randn(4,640,64,64).to('cuda')
    # c=torch.randn(4,640,64,64).to('cuda')
    # out=model(input,c)
    # print(out.shape)
    # # 打印参数数量
    # print(f"模型参数总数: {num_parameters}")
    # # model=SpatialSelfAttention(1280)
    a=torch.randn(4,4,96,128)
    context=[torch.randn(4, 320, 48, 64), torch.randn(4, 640, 48, 64), torch.randn(4, 1280, 24, 32) ,torch.randn(4, 1280, 24, 32)]
    # x = [torch.randn(4, 320, 64, 64), torch.randn(4, 640, 64, 64) , torch.randn(4, 1280, 32, 32),torch.randn(4, 1280, 32, 32)]
    input={}

    input['sketch']=context.copy()
    input['lineart'] = context.copy()
    # # out=model(a,context)
    # # print(out.shape)
    resblock=Adapter_XL()
    print(resblock)
    timesteps = torch.randint(0, 1000, (4,))
    timesteps = timesteps.long()
    out=resblock(latent=a,context=input,temb=timesteps)
    print(out[0].shape,out[1].shape,out[2].shape,out[3].shape)

    num_parameters = sum(p.numel() for p in resblock.parameters())

    # 打印参数数量
    print(f"模型参数总数: {num_parameters}")