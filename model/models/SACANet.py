from collections import namedtuple

import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
import collections




class SACANet(nn.Module):
    def __init__(self, load_weights=False):
        super(SACANet, self).__init__()

        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]

        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels = 512, dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.convin = nn.Conv2d(1024, 512, kernel_size=1)
        self.convskip = nn.Conv2d(512, 128, kernel_size=1)


        self.d1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.GroupNorm(num_groups=8, num_channels=128),#1,16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True)
        )
        self.d2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True)
        )
        self.d3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True)
        )


        self.atten1_1 = Self_Attn( 128 )
        self.atten1_2 = Self_Attn( 128 )
        self.atten1_3 = Self_Attn( 128 )
        self.atten1_4 = Self_Attn( 128 )

        self.up1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=9, stride=1, padding=4),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.ReLU(inplace=True)
        )


        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            fsd=collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key=list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key]=list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)


    def forward(self,x):  

        xb = self.downsample(x)
        xb = self.frontend(xb)
        x = self.frontend(x)
        m_batchsize, C, width, height = x.size()

        xb = self.upsample(xb)
        m_batchsized, Cd, widthd, heightd = xb.size()
        xt = torch.zeros(m_batchsize, C, width, height)
        xt[:m_batchsized, :Cd, :widthd, :heightd] = xb
        xt = xt.cuda()
        
        x4 = torch.cat([x, xt], 1)
        x4 = shuffle_channels(x4, groups=2)
        x4 = self.convin(x4)

        f1 = self.d1(x4)
        f2 = self.d2(x4)
        f3 = self.d3(x4)

        i1 = self.atten1_1(f1)
        i2 = self.atten1_2(f2)
        i3 = self.atten1_3(f3)

        l1 = torch.cat([i1, i2, i3], 1)

        sk = self.convskip(x)
        l1 = torch.cat([l1, sk], 1)
        o1 = self.backend(l1)
        o5 = self.output_layer(o1)

        return o5

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    x = x.transpose(1, 2).contiguous()

    x = x.view(batch_size, channels, height, width)
    return x


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

            
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  


class RefineBlock(nn.Module):
    def __init__(self, dim):

        super(RefineBlock, self).__init__()
        self.residual_block = self.build_residual_block(dim)
        self.output = nn.ReLU(inplace=True)

    def build_residual_block(self, dim):

        residual_block = []
        residual_block += [            
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        ]
        residual_block += [
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        ]
        residual_block += [
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*residual_block)

    def forward(self, x):
        out = x + self.residual_block(x)
        out = self.output(out)

        return out


class AttenBlock(nn.Module):
    def __init__(self, dim):

        super(AttenBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):

        conv_block = []
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True)
        ]
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out





class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, X):

        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs",['relu1_2','relu2_2','relu3_3','relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

# testing
if __name__=="__main__":
    csrnet=SACANet().to('cuda')
    input_img=torch.ones((1,3,256,256)).to('cuda')
    out=csrnet(input_img)
    print(out.mean())
