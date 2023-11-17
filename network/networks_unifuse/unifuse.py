from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from .resnet import *
from .mobilenet import *
from .layers import Conv3x3, ConvBlock, upsample, Cube2Equirec, Concat, BiProj, CEELayer, CEEAudioLayer
import torchvision.models as models
from collections import OrderedDict
import ipdb
try:
    from workspace.PanoFormer.PanoFormer.network.networks_unifuse.attention_blocks import MultiHeadedCrossmodalAttentionModule
except:
    from .attention_blocks import MultiHeadedCrossmodalAttentionModule

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


class UniFuse(nn.Module):
    """ UniFuse Model: Resnet based Euqi Encoder and Cube Encoder + Euqi Decoder
    """
    def __init__(self, num_layers = 18, equi_h = 512, equi_w = 1024, pretrained=True, max_depth=16.0,
                 fusion_type="cee", se_in_fusion=True, audio_enhanced = False):
        super(UniFuse, self).__init__()

        self.num_layers = num_layers
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.cube_h = equi_h//2

        self.fusion_type = fusion_type
        self.se_in_fusion = se_in_fusion
        self.audio_enhanced = audio_enhanced

        # encoder
        encoder = {2: mobilenet_v2,
                   18: resnet18,
                   34: resnet34,
                   50: resnet50,
                   101: resnet101,
                   152: resnet152}

        if num_layers not in encoder:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
        self.equi_encoder = encoder[num_layers](pretrained)
        self.cube_encoder = encoder[num_layers](pretrained)

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        if num_layers < 18:
            self.num_ch_enc = np.array([16, 24, 32, 96, 320])

        # decoder
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.equi_dec_convs = OrderedDict()
        self.c2e = {}

        Fusion_dict = {"cat": Concat,
                       "biproj": BiProj,
                       "cee": CEELayer,
                       "ceeAudio": CEEAudioLayer}
        
        if self.audio_enhanced:
            #print("cuz sum ablation, thus still CEELayer used.\n")
            print("CEEAudioLayer used.\n")
            FusionLayer_audio = Fusion_dict["ceeAudio"]
            #FusionLayer_audio = Fusion_dict[self.fusion_type]
            FusionLayer = Fusion_dict[self.fusion_type]
        else:
            print("CEELayer used.\n")
            FusionLayer = Fusion_dict[self.fusion_type]


        self.c2e["5"] = Cube2Equirec(self.cube_h // 32, self.equi_h // 32, self.equi_w // 32)

        if self.audio_enhanced:
            self.equi_dec_convs["fusion_5"] = FusionLayer_audio(self.num_ch_enc[4], SE=self.se_in_fusion)
        else:
            self.equi_dec_convs["fusion_5"] = FusionLayer(self.num_ch_enc[4], SE=self.se_in_fusion)
            
        self.equi_dec_convs["upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])

        self.c2e["4"] = Cube2Equirec(self.cube_h // 16, self.equi_h // 16, self.equi_w // 16)
        self.equi_dec_convs["fusion_4"] = FusionLayer(self.num_ch_enc[3], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_4"] = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])
        self.equi_dec_convs["upconv_4"] = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])

        self.c2e["3"] = Cube2Equirec(self.cube_h // 8, self.equi_h // 8, self.equi_w // 8)
        self.equi_dec_convs["fusion_3"] = FusionLayer(self.num_ch_enc[2], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_3"] = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])
        self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])

        self.c2e["2"] = Cube2Equirec(self.cube_h // 4, self.equi_h // 4, self.equi_w // 4)
        self.equi_dec_convs["fusion_2"] = FusionLayer(self.num_ch_enc[1], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_2"] = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])
        self.equi_dec_convs["upconv_2"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])

        self.c2e["1"] = Cube2Equirec(self.cube_h // 2, self.equi_h // 2, self.equi_w // 2)
        self.equi_dec_convs["fusion_1"] = FusionLayer(self.num_ch_enc[0], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])
        self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])

        self.equi_dec_convs["deconv_0"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])

        self.equi_dec_convs["depthconv_0"] = Conv3x3(self.num_ch_dec[0], 1)

        self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))
        self.projectors = nn.ModuleList(list(self.c2e.values()))

        self.spec_enc = SpecEnc()
        self.cross_attn_visual = MultiHeadedCrossmodalAttentionModule(d_model = 512, num_heads = 8)
        self.cross_attn_audio = MultiHeadedCrossmodalAttentionModule(d_model = 512, num_heads = 8)


        self.sigmoid = nn.Sigmoid()

        self.max_depth = nn.Parameter(torch.tensor(max_depth), requires_grad=False)


    def forward(self, input_equi_image, input_cube_image, spec):

        # euqi image encoding
        #print("zero like used for input_equi_image and input_cube_image")
        #input_equi_image = torch.zeros_like(input_equi_image)
        #input_cube_image = torch.zeros_like(input_cube_image)

        if self.num_layers < 18:
            equi_enc_feat0, equi_enc_feat1, equi_enc_feat2, equi_enc_feat3, equi_enc_feat4 \
                = self.equi_encoder(input_equi_image)
        else:
            x = self.equi_encoder.conv1(input_equi_image)
            x = self.equi_encoder.relu(self.equi_encoder.bn1(x))
            equi_enc_feat0 = x

            x = self.equi_encoder.maxpool(x)
            equi_enc_feat1 = self.equi_encoder.layer1(x)
            equi_enc_feat2 = self.equi_encoder.layer2(equi_enc_feat1)
            equi_enc_feat3 = self.equi_encoder.layer3(equi_enc_feat2)
            equi_enc_feat4 = self.equi_encoder.layer4(equi_enc_feat3)


        # cube image encoding
        cube_inputs = torch.cat(torch.split(input_cube_image, self.cube_h, dim=-1), dim=0)

        if self.num_layers < 18:
            cube_enc_feat0, cube_enc_feat1, cube_enc_feat2, cube_enc_feat3, cube_enc_feat4 \
                = self.cube_encoder(cube_inputs)
        else:

            x = self.cube_encoder.conv1(cube_inputs)
            x = self.cube_encoder.relu(self.cube_encoder.bn1(x))
            cube_enc_feat0 = x

            x = self.cube_encoder.maxpool(x)

            cube_enc_feat1 = self.cube_encoder.layer1(x)
            cube_enc_feat2 = self.cube_encoder.layer2(cube_enc_feat1)
            cube_enc_feat3 = self.cube_encoder.layer3(cube_enc_feat2)
            cube_enc_feat4 = self.cube_encoder.layer4(cube_enc_feat3)


        # euqi image decoding fused with cubemap features
        outputs = {}
        
        cube_enc_feat4 = torch.cat(torch.split(cube_enc_feat4, input_equi_image.shape[0], dim=0), dim=-1)
        c2e_enc_feat4 = self.c2e["5"](cube_enc_feat4)
        
        if not self.audio_enhanced:
            fused_feat4 = self.equi_dec_convs["fusion_5"](equi_enc_feat4, c2e_enc_feat4) # -> torch.Size([2, 512, 16, 32])
        else:
            # equi_enc_feat4.shape: torch.Size([12, 512, 16, 32])
            # c2e_enc_feat4.shape: torch.Size([12, 512, 16, 32])
            #spec_feat = self.spec_enc(spec).unsqueeze(-1).unsqueeze(-1) #.repeat(1,1,16,32)
            #ipdb.set_trace()
            """
            # ablation concatenation
            spec_feat = self.spec_enc(spec).unsqueeze(-1).unsqueeze(-1).repeat(1,1,16,32)
            assert spec_feat.shape == equi_enc_feat4.shape == c2e_enc_feat4.shape
            # spec_feat.shape: torch.Size([12, 512])
            fused_feat4 = self.equi_dec_convs["fusion_5"](equi_enc_feat4, c2e_enc_feat4, spec_feat)
            """
            
            # cross attention
            #equi_enc_feat4 = equi_enc_feat4 + spec_feat
            #fused_feat4 = self.equi_dec_convs["fusion_5"](equi_enc_feat4, c2e_enc_feat4)
            
            #ablation only sound, zero as rgb
            spec_feat = self.spec_enc(spec).unsqueeze(-1).unsqueeze(-1).repeat(1,1,16,32)
            #ipdb.set_trace()
            spec_feat = spec_feat.contiguous().view(spec_feat.shape[0], 512, -1)
            equi_enc_feat4_ = equi_enc_feat4.contiguous().view(spec_feat.shape[0], 512, -1)
            
            audio_skip = self.cross_attn_visual(inputs = equi_enc_feat4_, img_feat = spec_feat)
            visual_skip = self.cross_attn_audio(inputs = spec_feat , img_feat = equi_enc_feat4_)
            #skip = self.fusion_conv(torch.cat([visual_out, audio_out], dim=1).unsqueeze(-1)).squeeze(-1)
            #conv4 = skip #conv4_orig + skip
            equi_enc_feat4_fused = equi_enc_feat4_ + visual_skip + spec_feat + audio_skip
            equi_enc_feat4_fused_reshaped = equi_enc_feat4_fused.contiguous().view(spec_feat.shape[0], 512, 16, 32)
            
            fused_feat4 = self.equi_dec_convs["fusion_5"](equi_enc_feat4_fused_reshaped, c2e_enc_feat4)
        
            #assert spec_feat.shape == fused_feat4.shape
            #fused_feat4 = fused_feat4 + spec_feat
            
            
        
        #fused_feat4 = spec_feat.unsqueeze(-1).unsqueeze(-1) + fused_feat4
        #ipdb.set_trace()
        equi_x = upsample(self.equi_dec_convs["upconv_5"](fused_feat4))

        cube_enc_feat3 = torch.cat(torch.split(cube_enc_feat3, input_equi_image.shape[0], dim=0), dim=-1)
        c2e_enc_feat3 = self.c2e["4"](cube_enc_feat3)
        fused_feat3 = self.equi_dec_convs["fusion_4"](equi_enc_feat3, c2e_enc_feat3)
        equi_x = torch.cat([equi_x, fused_feat3], 1)
        equi_x = self.equi_dec_convs["deconv_4"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_4"](equi_x))

        cube_enc_feat2 = torch.cat(torch.split(cube_enc_feat2, input_equi_image.shape[0], dim=0), dim=-1)
        c2e_enc_feat2 = self.c2e["3"](cube_enc_feat2)
        fused_feat2 = self.equi_dec_convs["fusion_3"](equi_enc_feat2, c2e_enc_feat2)
        equi_x = torch.cat([equi_x, fused_feat2], 1)
        equi_x = self.equi_dec_convs["deconv_3"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_3"](equi_x))

        cube_enc_feat1 = torch.cat(torch.split(cube_enc_feat1, input_equi_image.shape[0], dim=0), dim=-1)
        c2e_enc_feat1 = self.c2e["2"](cube_enc_feat1)
        fused_feat1 = self.equi_dec_convs["fusion_2"](equi_enc_feat1, c2e_enc_feat1)
        equi_x = torch.cat([equi_x, fused_feat1], 1)
        equi_x = self.equi_dec_convs["deconv_2"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_2"](equi_x))

        cube_enc_feat0 = torch.cat(torch.split(cube_enc_feat0, input_equi_image.shape[0], dim=0), dim=-1)
        c2e_enc_feat0 = self.c2e["1"](cube_enc_feat0)
        fused_feat0 = self.equi_dec_convs["fusion_1"](equi_enc_feat0, c2e_enc_feat0)
        equi_x = torch.cat([equi_x, fused_feat0], 1)
        equi_x = self.equi_dec_convs["deconv_1"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_1"](equi_x))

        equi_x = self.equi_dec_convs["deconv_0"](equi_x)

        equi_depth = self.equi_dec_convs["depthconv_0"](equi_x)
        #outputs["pred_depth"] = self.max_depth * self.sigmoid(equi_depth)
        outputs["pred_depth"] = self.sigmoid(equi_depth)

        return outputs
