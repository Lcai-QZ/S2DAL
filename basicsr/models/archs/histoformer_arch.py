import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

#########################################################################

Conv2d = nn.Conv2d
##########################################################################
## Layer Norm
def to_2d(x):
    return rearrange(x, 'b c h w -> b (h w c)')

def to_3d(x):
#    return rearrange(x, 'b c h w -> b c (h w)')
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
#    return rearrange(x, 'b c (h w) -> b c h w',h=h,w=w)
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

#        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) #* self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

#        self.weight = nn.Parameter(torch.ones(normalized_shape))
#        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) #* self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Dual-scale Gated Feed-Forward Network (DGFF)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv_5 = Conv2d(hidden_features//4, hidden_features//4, kernel_size=5, stride=1, padding=2, groups=hidden_features//4, bias=bias)
        self.dwconv_dilated2_1 = Conv2d(hidden_features//4, hidden_features//4, kernel_size=3, stride=1, padding=2, groups=hidden_features//4, bias=bias, dilation=2)
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)

        self.project_out = Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)

        x1, x2 = x.chunk(2, dim=1)
#        x2_1, x2_2 = x2.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1( x2 )
#        x2_2 = self.dwconv_dilated3_1( x2_2 )
#        x2 = torch.cat([x2_1, x2_2], dim=1)
        x = F.mish( x2 ) * x1
        x = self.p_unshuffle(x)
        x = self.project_out(x)

        
#        x1 = self.dwconv_5(x)
#        x2 = self.dwconv_dilated_2(x)
#        x = F.mish(x2) * x1 + x 

        return x



##########################################################################
## Dynamic-range Histogram Self-Attention (DHSA)

class Attention_histogram(nn.Module):
    def __init__(self, dim, num_heads, bias, ifBox=True):
        super(Attention_histogram, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim*5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim*5, dim*5, kernel_size=3, stride=1, padding=1, groups=dim*5, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)


    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:,:,t_pad[0]:hw-t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit  = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) #* self.weight + self.bias
    

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b, head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b,c,h,w = x.shape
        x_sort, idx_h = x[:,:c//2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:,:c//2] = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1,k1,q2,k2,v = qkv.chunk(5, dim=1) # b,c,x,x

        v, idx = v.view(b,c,-1).sort(dim=-1)
        q1 = torch.gather(q1.view(b,c,-1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b,c,-1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b,c,-1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b,c,-1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)
        
        out1 = torch.scatter(out1, 2, idx, out1).view(b,c,h,w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b,c,h,w)
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:,:c//2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:,:c//2] = out_replace
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.attn_g = Attention_histogram(dim, num_heads, bias, True)
        self.norm_g = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.norm_ff1 = LayerNorm(dim, LayerNorm_type)



    def forward(self, x):
        x = x + self.attn_g(self.norm_g(x))
        x_out = x + self.ffn(self.norm_ff1(x))

        return x_out



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class SkipPatchEmbed(nn.Module):
    def __init__(self, in_c=3, dim=48, bias=False):
        super(SkipPatchEmbed, self).__init__()

        self.proj = nn.Sequential(
            nn.AvgPool2d( 2, stride=2, padding=0 , ceil_mode=False , count_include_pad=True , divisor_override=None ),
            Conv2d(in_c, dim, kernel_size=1, bias=bias),
            Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        )

    def forward(self, x, ):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
class Histoformer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Histoformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.skip_patch_embed1 = SkipPatchEmbed(3, 3)
        self.skip_patch_embed2 = SkipPatchEmbed(3, 3)
        self.skip_patch_embed3 = SkipPatchEmbed(3, 3)
        self.reduce_chan_level_1 = Conv2d(int(dim*2**1)+3, int(dim*2**1), kernel_size=1, bias=bias)
        self.reduce_chan_level_2 = Conv2d(int(dim*2**2)+3, int(dim*2**2), kernel_size=1, bias=bias)
        self.reduce_chan_level_3 = Conv2d(int(dim*2**3)+3, int(dim*2**3), kernel_size=1, bias=bias)

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, ):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1) # c,h,w

        inp_enc_level2 = self.down1_2(out_enc_level1) # 2c, h/2, w/2
        skip_enc_level1 = self.skip_patch_embed1(inp_img)
        inp_enc_level2 = self.reduce_chan_level_1(torch.cat([inp_enc_level2, skip_enc_level1], 1))

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        skip_enc_level2 = self.skip_patch_embed2(skip_enc_level1)
        inp_enc_level3 = self.reduce_chan_level_2(torch.cat([inp_enc_level3, skip_enc_level2], 1))

        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        skip_enc_level3 = self.skip_patch_embed3(skip_enc_level2)
        inp_enc_level4 = self.reduce_chan_level_3(torch.cat([inp_enc_level4, skip_enc_level3], 1))

        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        ###########################

        out_dec_level1 = self.output(out_dec_level1)
        return out_dec_level1 + inp_img



class ch_shuffle_high_text(nn.Module):
    def __init__(self, ch_dim,num_heads,LayerNorm_type,ffn_expansion_factor, bias,lin_ch=512):
        super(ch_shuffle_high_text, self).__init__()
        self.dim = ch_dim
        self.linear_layer1 = nn.Linear(lin_ch, lin_ch)
        self.linear_layer3 = nn.Linear(lin_ch, 2 * ch_dim)
        # -----------------------------------------------------------------------
        self.conv1x1 = nn.Conv2d(ch_dim, 2 * ch_dim, kernel_size=1, stride=1, padding=0)  #
        self.conv_out = nn.Conv2d(2 * ch_dim, ch_dim, kernel_size=1, stride=1, padding=0)  #
        self.norm1 = LayerNorm(ch_dim, LayerNorm_type)
        self.norm2 = LayerNorm(ch_dim, LayerNorm_type)
        self.norm3 = LayerNorm(ch_dim, LayerNorm_type)
        # self.select_attn = Topm_CrossAttention_Restormer(ch_dim, num_heads, bias=False)
        # self.ffn = FeedForward(ch_dim, ffn_expansion_factor, bias)
    def forward(self, img_featur, text_code):
        b, c, _, _ = img_featur.shape
        img_feature2 = img_featur
        text_code = self.linear_layer1(text_code)
        text_code = self.linear_layer3(text_code)
        # soft_values, soft_indices = torch.topk(text_code, k=2 * self.dim)
        soft_indices = torch. randperm(2 * self.dim).reshape(1, 2 * self.dim)
        img_featur = self.conv1x1(img_featur)
        shuffled_img = img_featur[torch.arange(b).unsqueeze(1), soft_indices, :, :]  # shuffle
        output = self.conv_out(shuffled_img)
        return output, img_feature2


#----------------Histoformer_ship_ChannelShuffle--------------------#
class Histoformer_skip_ChannelShuffle(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',  ## Other option 'BiasFree'
        # device="cuda:0",
        dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Histoformer_skip_ChannelShuffle, self).__init__()
        # self.device = device

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_shuffle_channel1 = ch_shuffle_high_text(ch_dim = dim,num_heads=heads[0],LayerNorm_type=LayerNorm_type,ffn_expansion_factor=ffn_expansion_factor,bias=bias) # encoder level1 shuffle


        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_shuffle_channel2 = ch_shuffle_high_text(ch_dim = int(dim*2**1),num_heads=heads[1],LayerNorm_type=LayerNorm_type,ffn_expansion_factor=ffn_expansion_factor,bias=bias) # encoder level2 shuffle

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_shuffle_channel3 = ch_shuffle_high_text(ch_dim = int(dim*2**2),num_heads=heads[2],LayerNorm_type=LayerNorm_type,ffn_expansion_factor=ffn_expansion_factor,bias=bias) # encoder level3 shuffle

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.latent_shuffle_channel = ch_shuffle_high_text(ch_dim=int(dim * 2 ** 3), num_heads=heads[3],LayerNorm_type=LayerNorm_type, ffn_expansion_factor=ffn_expansion_factor, bias=bias)  # latent latent shuffle

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.skip_patch_embed1 = SkipPatchEmbed(3, 3)
        self.skip_patch_embed2 = SkipPatchEmbed(3, 3)
        self.skip_patch_embed3 = SkipPatchEmbed(3, 3)
        self.reduce_chan_level_1 = Conv2d(int(dim * 2 ** 1) + 3, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level_2 = Conv2d(int(dim * 2 ** 2) + 3, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan_level_3 = Conv2d(int(dim * 2 ** 3) + 3, int(dim * 2 ** 3), kernel_size=1, bias=bias)

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, text_code):
        # text_code = torch.randn(1,512).to(self.device) # 这个在测试模型参数量和计算量时候加上
        inp_enc_level1 = self.patch_embed(inp_img)  # ch 3-->dim:48
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # c,h,w

        inp_enc_level2 = self.down1_2(out_enc_level1)  # 2c, h/2, w/2
        skip_enc_level1 = self.skip_patch_embed1(inp_img)
        inp_enc_level2 = self.reduce_chan_level_1(torch.cat([inp_enc_level2, skip_enc_level1], 1))
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        skip_enc_level2 = self.skip_patch_embed2(skip_enc_level1)
        inp_enc_level3 = self.reduce_chan_level_2(torch.cat([inp_enc_level3, skip_enc_level2], 1))
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        skip_enc_level3 = self.skip_patch_embed3(skip_enc_level2)
        inp_enc_level4 = self.reduce_chan_level_3(torch.cat([inp_enc_level4, skip_enc_level3], 1))
        latent = self.latent(inp_enc_level4)
        latent, _ = self.latent_shuffle_channel(latent, text_code)  # latent latent shuffle

        inp_dec_level3 = self.up4_3(latent)
        outt1, _ = self.encoder_shuffle_channel3(out_enc_level3, text_code)
        inp_dec_level3 = torch.cat([inp_dec_level3, outt1], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        outt2, _ = self.encoder_shuffle_channel2(out_enc_level2, text_code)
        inp_dec_level2 = torch.cat([inp_dec_level2, outt2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        outt3, _ = self.encoder_shuffle_channel1(out_enc_level1, text_code)
        inp_dec_level1 = torch.cat([inp_dec_level1, outt3], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)
        return out_dec_level1 + inp_img

















