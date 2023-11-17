import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import sys
from .Utils import *
from .Utils.CubePad import CustomPad
import ipdb
import torchvision.models as models
try:
    from attention_blocks import MultiHeadedCrossmodalAttentionModule
except:
    from .attention_blocks import MultiHeadedCrossmodalAttentionModule

class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        # currently not compatible with running on CPU
        self.weights = torch.autograd.Variable(
            torch.zeros(num_channels, 1, stride, stride))
        self.weights[:, :, 0, 0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights.cuda(), stride=self.stride, groups=self.num_channels)


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class Decoder(nn.Module):
    # Decoder is the base class for all decoders

    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size >= 2, "kernel_size out of range: {}".format(
            kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2*padding + kernel_size + \
                output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                (module_name, nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size,
                                                 stride, padding, output_padding, bias=False)),
                ('batchnorm', nn.BatchNorm2d(in_channels//2)),
                ('relu',      nn.ReLU(inplace=True)),
            ]))

        self.layer1 = convt(in_channels)
        self.layer2 = convt(in_channels // 2)
        self.layer3 = convt(in_channels // (2 ** 2))
        self.layer4 = convt(in_channels // (2 ** 3))


class UpConv(Decoder):
    # UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
    def upconv_module(self, in_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(collections.OrderedDict([
            ('unpool',    Unpool(in_channels)),
            ('conv',      nn.Conv2d(in_channels, in_channels//2,
                                    kernel_size=5, stride=1, padding=2, bias=False)),
            ('batchnorm', nn.BatchNorm2d(in_channels//2)),
            ('relu',      nn.ReLU()),
        ]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels)
        self.layer2 = self.upconv_module(in_channels//2)
        self.layer3 = self.upconv_module(in_channels//4)
        self.layer4 = self.upconv_module(in_channels//8)


class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels, out_channels=None, padding=None):
            super(UpProj.UpProjModule, self).__init__()
            if out_channels is None:
                out_channels = in_channels//2
            self.pad_3 = padding(1)
            self.pad_5 = padding(2)

            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
                ('pad1', CustomPad(self.pad_5)),
                ('conv1',      nn.Conv2d(in_channels, out_channels,
                                         kernel_size=5, stride=1, padding=0, bias=False)),
                ('batchnorm1', nn.BatchNorm2d(out_channels)),
                ('relu',      nn.ReLU()),
                ('pad2', CustomPad(self.pad_3)),
                ('conv2',      nn.Conv2d(out_channels, out_channels,
                                         kernel_size=3, stride=1, padding=0, bias=False)),
                ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
                ('pad', CustomPad(self.pad_5)),
                ('conv',      nn.Conv2d(in_channels, out_channels,
                                        kernel_size=5, stride=1, padding=0, bias=False)),
                ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels, padding):
        super(UpProj, self).__init__()
        self.padding = getattr(CubePad, padding)
        self.layer1 = self.UpProjModule(in_channels   , padding=self.padding)
        self.layer2 = self.UpProjModule(in_channels//2, padding=self.padding)
        self.layer3 = self.UpProjModule(in_channels//4, padding=self.padding)
        self.layer4 = self.UpProjModule(in_channels//8, padding=self.padding)

def choose_decoder(decoder, in_channels, padding):
    # iheight, iwidth = 10, 8
    if decoder[:6] == 'deconv':
        assert len(decoder) == 7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    elif decoder == "upproj":
        return UpProj(in_channels, padding=padding)
    elif decoder == "upconv":
        return UpConv(in_channels)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)

def e2c(equirectangular):
    cube = Equirec2Cube.ToCubeTensor(equirectangular.cuda())
    return cube

def c2e(cube):
    equirectangular = Cube2Equirec.ToEquirecTensor(cube.cuda())
    return equirectangular

class PreprocBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_lst, stride=2):
        super(PreprocBlock, self).__init__()
        assert len(kernel_size_lst) == 4 and out_channels % 4 == 0
        self.lst = nn.ModuleList([])

        for (h, w) in kernel_size_lst:
            padding = (h//2, w//2)
            tmp = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//4, kernel_size=(h,w), stride=stride, padding=padding),
                        nn.BatchNorm2d(out_channels//4),
                        nn.ReLU(inplace=True)
                    )
            self.lst.append(tmp)

    def forward(self, x):
        out = []
        for conv in self.lst:
            out.append(conv(x))
        out = torch.cat(out, dim=1)
        return out

class fusion_ResNet(nn.Module):
    _output_size_init = (256, 256)

    def __init__(self, bs, layers, decoder, output_size=None, in_channels=3, pretrained=True, padding='ZeroPad', audio_enhanced = True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(
                'Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(fusion_ResNet, self).__init__()
        self.padding = getattr(CubePad, padding)
        self.pad_7 = self.padding(3)
        self.pad_3 = self.padding(1)
        self.audio_enhanced = audio_enhanced
        print("audio_enhanced for fusion_ResNet:{}".format(self.audio_enhanced))
        try: from . import resnet
        except: import resnet
        pretrained_model = getattr(resnet, 'resnet%d'%layers)(pretrained=pretrained, padding=padding)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.output_size = output_size
        if output_size == None:
            output_size = _output_size_init
        else:
            assert isinstance(output_size, tuple)
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048
            
        if self.audio_enhanced:
            print("cross attention and sum used, thus self.conv2 num_channels instead of num_channels*2\n")
            self.conv2 = nn.Conv2d(num_channels, num_channels //
                                2, kernel_size=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(num_channels, num_channels //
                                2, kernel_size=1, bias=False)
            
        self.bn2 = nn.BatchNorm2d(num_channels//2)

        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        
        self.pre1 = PreprocBlock(3, 64, [[3, 9], [5, 11], [5, 7], [7, 7]])
        self.pre1.apply(weights_init)

    def forward(self, inputs):
        # resnet
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x0 = x
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.conv2(x4)
        x = self.bn2(x)

        return x

    def pre_encoder(self, x):
        x = self.conv1(self.pad_7(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(self.pad_3(x))

        return x
    
    def pre_encoder2(self, x):
        x = self.pre1(x)
        x = self.maxpool(self.pad_3(x))

        return x

class CETransform(nn.Module):
    def __init__(self):
        super(CETransform, self).__init__()
        #equ_h = [512, 128, 64, 32, 16]
        #cube_h = [256, 64, 32, 16, 8]
        equ_h = [512, 256, 128, 64, 32, 16, 8]
        cube_h = [256, 128, 64, 32, 16, 8, 4]

        self.c2e = dict()
        self.e2c = dict()

        for h in equ_h:
            a = Equirec2Cube(1, h, h*2, h//2, 90)
            self.e2c['(%d,%d)' % (h, h*2)] = a

        for h in cube_h:
            a = Cube2Equirec(1, h, h*2, h*4)
            self.c2e['(%d)' % (h)] = a

    def E2C(self, x):
        [bs, c, h, w] = x.shape
        key = '(%d,%d)' % (h, w)
        assert key in self.e2c
        return self.e2c[key].ToCubeTensor(x)

    def C2E(self, x):
        [bs, c, h, w] = x.shape
        key = '(%d)' % (h)
        assert key in self.c2e and h == w
        return self.c2e[key].ToEquirecTensor(x)

    def forward(self, equi, cube):
        return self.e2c(equi), self.c2e(cube)


def lstm_forward(bilstm_list, CE, feat_equi, feat_cube, idx, max_iters=3):
    bs, bs1 = feat_equi.shape[0], feat_cube.shape[0]
    assert bs1 == bs * 6
    init_tmp_c2e = CE.C2E(feat_cube)
    he0_fw, ce0_fw = [x.cuda()
                      for x in bilstm_list[idx].lstm_fw.init_hidden(bs)]
    he0_bw, ce0_bw = [x.cuda()
                      for x in bilstm_list[idx].lstm_bw.init_hidden(bs)]
    e_xte = torch.stack((feat_equi, init_tmp_c2e), dim=1)
    for i in range(max_iters):
        if i == 1:
            e_xte[:, 0, :, :, :] += e_out_list[1]
            e_xte[:, 1, :, :, :] += e_out_list[0]
        elif i >= 2:
            e_xte[:, 0, :, :, :] += (e_out_list[1] + cube_out)
            e_xte[:, 1, :, :, :] += (e_out_list[0] + equi_out)
        e_out_list, (ht, ct, h_bwout, c_bwout) = bilstm_list[idx](
            e_xte, he0_fw, ce0_fw, he0_bw, ce0_bw)
        he0_fw, ce0_fw, he0_bw, ce0_bw = ht, ct, h_bwout, c_bwout
        equi_out = e_xte[:, 0, :, :, :].clone()
        cube_out = e_xte[:, 1, :, :, :].clone()
    return e_out_list

class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.refine_1 = nn.Sequential(
                        nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
                        )
        self.refine_2 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        )
        self.deconv_1 = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0, groups=1, bias=True, dilation=1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(inplace=True),
                        )
        self.deconv_2 = nn.Sequential(
                        nn.ConvTranspose2d(192, 32, kernel_size=4, stride=2, padding=1, output_padding=0, groups=1, bias=True, dilation=1),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(inplace=True),
                        )
        self.refine_3 = nn.Sequential(
                        nn.Conv2d(96, 16, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(16),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)
                        )
        #self.bilinear_1 = nn.UpsamplingBilinear2d(size=(256,512))
        self.bilinear_1 = nn.UpsamplingBilinear2d(size=(128,256))
        #self.bilinear_2 = nn.UpsamplingBilinear2d(size=(512,1024))
        self.bilinear_2 = nn.UpsamplingBilinear2d(size=(256,512))
    def forward(self, inputs):
        x = inputs
        out_1 = self.refine_1(x)
        out_2 = self.refine_2(out_1)
        deconv_out1 = self.deconv_1(out_2)
        up_1 = self.bilinear_1(out_2)
        deconv_out2 = self.deconv_2(torch.cat((deconv_out1, up_1), dim = 1))
        up_2 = self.bilinear_2(out_1)
        out_3 = self.refine_3(torch.cat((deconv_out2, up_2), dim = 1))

        return out_3                


class SpecEnc(nn.Module):
    def __init__(self):
        """
        ResNet-18.
        Takes in observations and produces an embedding of the rgb and depth components
        """
        super().__init__()

        self._n_input_spec = 2
        new_dict = {}
          
        self.cnn = models.resnet18(pretrained=False)
        #self.cnn = models.resnet34(pretrained=False)
        self.cnn.fc_backup = self.cnn.fc
        self.cnn.fc = nn.Sequential()
        """
        self.cnn = models.resnet18(pretrained=False)
        pretrained_dict = torch.load(pretrained_minc2500_resnet18_path)['params']
        model_dict = self.cnn.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.cnn.load_state_dict(model_dict)
        """

        if self._n_input_spec != 3:
            self.cnn.conv1 = nn.Conv2d(self._n_input_spec,
                                       self.cnn.conv1.out_channels,
                                       kernel_size=self.cnn.conv1.kernel_size,
                                       stride=self.cnn.conv1.stride,
                                       padding=self.cnn.conv1.padding,
                                       bias=False)

            nn.init.kaiming_normal_(
                self.cnn.conv1.weight, mode="fan_out", nonlinearity="relu",
            )

        
        value_scale = 1. #255
        mean = [0.485, 0.456, 0.406]
        self.mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        self.std = [item * value_scale for item in std]


    @property
    def is_blind(self):
        """
        get if network produces any output features or not
        :return: if network produces any output features or not
        """
        return False

    @property
    def n_out_feats(self):
        """
        get number of visual encoder output features
        :return: number of visual encoder output features
        """
        if self.is_blind:
            return 0
        else:
            # resnet-18
            return 512

    def forward(self, cnn_input,):
        """
        does forward pass in visual encoder
        :param observations: observations
        :return: visual features
        """

        """
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations.float() / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.float().permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        cnn_input = torch.cat(cnn_input, dim=1)
        """
        #cnn_input[:,0:3,:,:] = TTR.Normalize(self.mean, self.std)(cnn_input[:,0:3,:,:])
        #cnn_input = TTR.Normalize(self.mean, self.std)(cnn_input)
        return self.cnn(cnn_input)

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])
    
class attentionNet(nn.Module):
    def __init__(self, att_out_nc, input_nc):
        super(attentionNet, self).__init__()
        #initialize layers
        
        self.attention_img = nn.Bilinear(512, 512, att_out_nc)
        self.attention_material = nn.Bilinear(512, 512, att_out_nc)
        self.upconvlayer1 = unet_upconv(input_nc, 512) 
        self.upconvlayer2 = unet_upconv(512, 256)
        self.upconvlayer3 = unet_upconv(256, 128)
        self.upconvlayer4 = unet_upconv(128, 64)
        self.upconvlayer5 = unet_upconv(64, 1, True)
        
    def forward(self, rgb_feat, echo_feat, mat_feat):
        rgb_feat = rgb_feat.permute(0, 2, 3, 1).contiguous()
        echo_feat = echo_feat.permute(0, 2, 3, 1).contiguous()
        mat_feat = mat_feat.permute(0, 2, 3, 1).contiguous()
        
        attentionImg = self.attention_img(rgb_feat, echo_feat)
        attentionMat = self.attention_material(mat_feat, echo_feat)
    
        attentionImg = attentionImg.permute(0, 3, 1, 2).contiguous()
        attentionMat = attentionMat.permute(0, 3, 1, 2).contiguous()
        
        audioVisual_feature = torch.cat((attentionImg, attentionMat), dim=1)
        
        upconv1feature = self.upconvlayer1(audioVisual_feature)
        upconv2feature = self.upconvlayer2(upconv1feature)
        upconv3feature = self.upconvlayer3(upconv2feature)
        upconv4feature = self.upconvlayer4(upconv3feature)
        attention = self.upconvlayer5(upconv4feature)
        return attention, audioVisual_feature

class MyModel(nn.Module):
    def __init__(self, layers, decoder, output_size=None, in_channels=3, pretrained=True, audio_enhanced = True):
        super(MyModel, self).__init__()
        bs = 1
        self.audio_enhanced = audio_enhanced
        print("audio_enhanced Bifuse: {}".format(audio_enhanced))
        """
        self.equi_model = fusion_ResNet(
            bs, layers, decoder, (512, 1024), 3, pretrained, padding='ZeroPad', audio_enhanced = self.audio_enhanced)
        self.cube_model = fusion_ResNet(
            bs*6, layers, decoder, (256, 256), 3, pretrained, padding='SpherePad', audio_enhanced = False)
        """
        """
        self.equi_model = fusion_ResNet(
            bs, layers, decoder, (256, 512), 3, pretrained, padding='ZeroPad', audio_enhanced = False)
        self.cube_model = fusion_ResNet(
            bs*6, layers, decoder, (128, 128), 3, pretrained, padding='SpherePad', audio_enhanced = False)
        """
        
        self.equi_model = fusion_ResNet(
            bs, layers, decoder, (256, 512), 3, pretrained, padding='ZeroPad', audio_enhanced = self.audio_enhanced)
        self.cube_model = fusion_ResNet(
            bs*6, layers, decoder, (128, 128), 3, pretrained, padding='SpherePad', audio_enhanced = False)
        
        
        self.refine_model = Refine()
        
        self.cross_attn_visual = MultiHeadedCrossmodalAttentionModule(d_model = 128, num_heads = 8)
        self.cross_attn_audio = MultiHeadedCrossmodalAttentionModule(d_model = 128, num_heads = 8)
        """
        self.cross_attn_visual = MultiHeadedCrossmodalAttentionModule(d_model = 512, num_heads = 8)
        self.cross_attn_audio = MultiHeadedCrossmodalAttentionModule(d_model = 512, num_heads = 8)
        """

        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.equi_decoder = choose_decoder(decoder, num_channels//2, padding='ZeroPad')
        self.equi_conv3 = nn.Sequential(
                nn.Conv2d(num_channels//32, 1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.UpsamplingBilinear2d(size=(256, 512))
                #nn.UpsamplingBilinear2d(size=(512, 1024))
                )
        self.cube_decoder = choose_decoder(decoder, num_channels//2, padding='SpherePad')
        mypad = getattr(CubePad, 'SpherePad')
        self.cube_conv3 = nn.Sequential(
                mypad(1),
                nn.Conv2d(num_channels//32, 1, kernel_size=3, stride=1, padding=0, bias=False),
                #nn.UpsamplingBilinear2d(size=(256, 256))
                nn.UpsamplingBilinear2d(size=(128, 128))
                )

        self.equi_decoder.apply(weights_init)
        self.equi_conv3.apply(weights_init)
        self.cube_decoder.apply(weights_init)
        self.cube_conv3.apply(weights_init)

        self.ce = CETransform()
        if self.audio_enhanced:
            print("SpecEnc created.\n")
            self.spec_enc = SpecEnc()
        
        if layers <= 34:
            ch_lst = [64, 64, 128, 256, 512, 256, 128, 64, 32]
        else:
            ch_lst = [64, 256, 512, 1024, 2048, 1024, 512, 256, 128]

        self.conv_e2c = nn.ModuleList([])
        self.conv_c2e = nn.ModuleList([])
        self.conv_mask = nn.ModuleList([])
        for i in range(9):
            conv_c2e = nn.Sequential(
                        nn.Conv2d(ch_lst[i], ch_lst[i], kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    )
            conv_e2c = nn.Sequential(
                        nn.Conv2d(ch_lst[i], ch_lst[i], kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    )
            conv_mask = nn.Sequential(
                        nn.Conv2d(ch_lst[i]*2, 1, kernel_size=1, padding=0),
                        nn.Sigmoid()
                    )
            self.conv_e2c.append(conv_e2c)
            self.conv_c2e.append(conv_c2e)
            self.conv_mask.append(conv_mask)

        #self.grid = Equirec2Cube(None, 512, 1024, 256, 90).GetGrid()
        self.grid = Equirec2Cube(None, 256, 512, 128, 90).GetGrid()
        self.d2p = Depth2Points(self.grid)
    def forward_FCRN_fusion(self, equi, audio, fusion=False):
        cube = self.ce.E2C(equi)
        feat_equi = self.equi_model.pre_encoder2(equi)
        feat_cube = self.cube_model.pre_encoder(cube)
        for e in range(5):
            if fusion:
                aaa = self.conv_e2c[e](feat_equi)
                #print("aaa.shape:", aaa.shape)
                tmp_cube = self.ce.E2C(aaa)
                tmp_equi = self.conv_c2e[e](self.ce.C2E(feat_cube))
                mask_equi = self.conv_mask[e](torch.cat([aaa, tmp_equi], dim=1))
                mask_cube = 1 - mask_equi
                tmp_cube = tmp_cube.clone() * self.ce.E2C(mask_cube)
                tmp_equi = tmp_equi.clone() * mask_equi
            else:
                tmp_cube = 0
                tmp_equi = 0
            feat_cube = feat_cube + tmp_cube
            feat_equi = feat_equi + tmp_equi
            if e < 4:
                feat_cube = getattr(self.cube_model, 'layer%d'%(e+1))(feat_cube)
                feat_equi = getattr(self.equi_model, 'layer%d'%(e+1))(feat_equi)
            else:
                #ipdb.set_trace()
                feat_cube = self.cube_model.conv2(feat_cube)
                #feat_equi = self.equi_model.conv2(feat_equi)
                if self.audio_enhanced:
                    #ipdb.set_trace()
                    """
                    spec_feat = self.spec_enc(audio).unsqueeze(-1).unsqueeze(-1).repeat(1,1,feat_equi.shape[-2:][0],feat_equi.shape[-2:][1])
                    feat_equi = self.equi_model.conv2(torch.cat([feat_equi, spec_feat], 1))
                    #feat_equi = self.equi_model.conv2(feat_equi + spec_feat)
                    """
                    #ablation only sound, zero as rgb
                    size1 = feat_equi.shape[-2:][0]
                    size2 = feat_equi.shape[-2:][1]
                    spec_feat = self.spec_enc(audio).unsqueeze(-1).unsqueeze(-1).repeat(1,1,size1, size2)
                    #print("feat_equi.shape:",feat_equi.shape)
                    #ipdb.set_trace()
                    spec_feat = spec_feat.contiguous().view(spec_feat.shape[0], 512, -1)
                    feat_equi = feat_equi.contiguous().view(feat_equi.shape[0], 512, -1)
                    
                    audio_skip = self.cross_attn_visual(inputs = feat_equi, img_feat = spec_feat)
                    visual_skip = self.cross_attn_audio(inputs = spec_feat , img_feat = feat_equi)
                    #skip = self.fusion_conv(torch.cat([visual_out, audio_out], dim=1).unsqueeze(-1)).squeeze(-1)
                    #conv4 = skip #conv4_orig + skip
                    """
                    feat_equi_fused = feat_equi + visual_skip
                    spec_feat_fused = spec_feat + audio_skip
                    feat_equi_fused_reshaped = feat_equi_fused.contiguous().view(feat_equi_fused.shape[0], 512, size1, size2)
                    spec_feat_fused_reshaped = spec_feat_fused.contiguous().view(feat_equi_fused.shape[0], 512, size1, size2)
                    #ipdb.set_trace()
                    #feat_equi = self.equi_model.conv2(feat_equi_fused_reshaped)
                    feat_equi = self.equi_model.conv2(torch.cat([feat_equi_fused_reshaped, spec_feat_fused_reshaped], 1))
                    #feat_equi = feat_equi_fused_reshaped
                    #feat_equi = self.equi_model.conv2(feat_equi)
                    """
                    
                    feat_equi_fused = feat_equi + visual_skip + spec_feat + audio_skip
                    feat_equi_fused_reshaped = feat_equi_fused.contiguous().view(feat_equi_fused.shape[0], 512, size1, size2)
                    feat_equi = self.equi_model.conv2(feat_equi_fused_reshaped)
                    
                    
            
                else:
                    feat_equi = self.equi_model.conv2(feat_equi)
                feat_cube = self.cube_model.bn2(feat_cube)
                feat_equi = self.equi_model.bn2(feat_equi)
                

        for d in range(4):
            if fusion:
                aaa = self.conv_e2c[d+5](feat_equi)
                tmp_cube = self.ce.E2C(aaa)
                tmp_equi = self.conv_c2e[d+5](self.ce.C2E(feat_cube))
                mask_equi = self.conv_mask[d+5](torch.cat([aaa, tmp_equi], dim=1))
                mask_cube = 1 - mask_equi
                tmp_cube = tmp_cube.clone() * self.ce.E2C(mask_cube)
                tmp_equi = tmp_equi.clone() * mask_equi
                tmp_equi = tmp_equi.clone() * mask_equi
            else:
                tmp_cube = 0
                tmp_equi = 0
            feat_cube = feat_cube + tmp_cube
            feat_equi = feat_equi + tmp_equi

            feat_equi = getattr(self.equi_decoder, 'layer%d'%(d+1))(feat_equi)
            feat_cube = getattr(self.cube_decoder, 'layer%d'%(d+1))(feat_cube)
        equi_depth = self.equi_conv3(feat_equi)
        cube_depth = self.cube_conv3(feat_cube)

           
        cube_pts = self.d2p(cube_depth)   
        #ipdb.set_trace()   
        c2e_depth = self.ce.C2E(torch.norm(cube_pts, p=2, dim=3).unsqueeze(1))
        
        feat_cat = torch.cat((equi, equi_depth, c2e_depth), dim = 1)

        refine_final = self.refine_model(feat_cat)
        outputs = {}
        outputs["pred_depth"] = refine_final
        #ipdb.set_trace() 

        return outputs #equi_depth, cube_depth, refine_final
   
    def forward_FCRN_cube(self, equi):
        cube = self.ce.E2C(equi)
        feat_cube = self.cube_model.pre_encoder(cube)
        for e in range(5):
            if e < 4:
                feat_cube = getattr(self.cube_model, 'layer%d'%(e+1))(feat_cube)
            else:
                feat_cube = self.cube_model.conv2(feat_cube)
                feat_cube = self.cube_model.bn2(feat_cube)
        for d in range(4):
            feat_cube = getattr(self.cube_decoder, 'layer%d'%(d+1))(feat_cube)
        cube_depth = self.cube_conv3(feat_cube)
        return cube_depth

    def forward_FCRN_equi(self, equi):
        feat_equi = self.equi_model.pre_encoder2(equi)
        for e in range(5):
            if e < 4:
                feat_equi = getattr(self.equi_model, 'layer%d'%(e+1))(feat_equi)
            else:
                feat_equi = self.equi_model.conv2(feat_equi)
                feat_equi = self.equi_model.bn2(feat_equi)
        for d in range(4):
            feat_equi = getattr(self.equi_decoder, 'layer%d'%(d+1))(feat_equi)
        equi_depth = self.equi_conv3(feat_equi)
        return equi_depth

    def forward(self, x, audio):
        return self.forward_FCRN_fusion(x, audio, True)


if __name__ == "__main__":
    attention_net = attentionNet()
    equi_feat = torch.rand(2, 512,8,16)
    audio_feat = torch.rand(2, 512,8,16)