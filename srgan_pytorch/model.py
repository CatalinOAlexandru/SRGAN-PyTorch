# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Modified by ... Catalin Alexandru
# On ................... 2021.07.28

# ==============================================================================

import torch
import math
from torch import nn
from torch.hub import load_state_dict_from_url

__all__ = ["ResidualBlock", "SubpixelConvolutionLayer", "Discriminator",
           "Generator", "discriminator", "generator"]

model_urls = {
    "srgan": "https://github.com/Lornatang/SRGAN-PyTorch/releases/download/v0.4.0/SRGAN_DIV2K-cd696c87b61c37784d816498e535353a2f42e4307069d55bf151bc1ea5aafaf7.pth"
}


class ResidualBlock(nn.Module):
    r""" Residual convolution block.

    Args:
        channels (int): Number of channels in the input image.
    """

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, track_running_stats=False)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, track_running_stats=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += x

        return out


class SubpixelConvolutionLayer(nn.Module):
    r""" Sub-pixel upsampled convolution block.

    Args:
        channels (int): Number of channels in the input image.
    """

    def __init__(self, channels: int, up_scale):
        super(SubpixelConvolutionLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels * up_scale ** 2, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)

        return out


class Discriminator(nn.Module):

    def __init__(self,imgSize):
        # as from the original implementation it fixes some differences in sizes
        # which were initially hard coded for 96x96 image size. classfieierAdaption
        # will allow for all kind of image sizes without affecting the training.
        # we divide by 16 based on the architecture numbers and what output is always expected.
        classifierAdaption = int((imgSize/16)**2) # ** 2 is to not do 6x6 later for each nxmxd

        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            # input size. (3) x 96 x 96
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 48 x 48
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 24 x 24
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 12 x 12
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 6 x 6
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512, track_running_stats=False),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * classifierAdaption, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        # Init all model weights.
        self._initialize_weights()


    def forward(self, x):
        # print('Discriminator input',x.shape)
        out = self.features(x)
        # print('Discriminator after features',out.shape)
        out = torch.flatten(out, 1)
        # print('Discriminator after flatten',out.shape)
        out = self.classifier(out)
        # print('Discriminator after classifier',out.shape)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Generator(nn.Module):
    def __init__(self,scale_factor,num_resBlocks = 16):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        # First layer.
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.PReLU()
        )

        # 16 Residual blocks.
        trunk = []
        # used to be in range 16 as in the paper.
        # however, I am attempting to make the model deeper for testing.
        for _ in range(num_resBlocks):
            trunk += [ResidualBlock(64)]
        self.trunk = nn.Sequential(*trunk)

        # Second conv layer post residual blocks.
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=False)
        )

        # 2 Sub-pixel convolution layers.
        subpixel_conv = []
        for _ in range(upsample_block_num):
            subpixel_conv += [SubpixelConvolutionLayer(64,2)]
        self.subpixel_conv = nn.Sequential(*subpixel_conv)

        # Final output layer.
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 3, 9, 1, 4),
            nn.Tanh()
        )

        # Initializing all neural network weights.
        self._initialize_weights()

    def forward(self, x):
        # print('Generator input',x.shape)
        conv1 = self.conv1(x)
        # print('Generator conv1',conv1.shape)
        trunk = self.trunk(conv1)
        # print('Generator trunk',trunk.shape)
        conv2 = self.conv2(trunk)
        # print('Generator conv2',conv2.shape)
        out = conv1 + conv2
        out = self.subpixel_conv(out)
        # print('Generator subpixel_conv',out.shape)
        out = self.conv3(out)
        # print('Generator conv3/out',out.shape)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



# def discriminator() -> Discriminator:
#     r"""Build discriminator model from the <https://arxiv.org/abs/1609.04802>` paper.

#     Returns:
#         torch.nn.Module.
#     """
#     model = Discriminator()

#     return model


# def generator(pretrained: bool = False, progress: bool = True, up_scale: int) -> Generator:
#     r"""Build generator model from the <https://arxiv.org/abs/1609.04802>` paper.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     print('In generator upscale is set to:',up_scale)
#     model = Generator(up_scale)

#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls["srgan"],
#                                               progress=progress,
#                                               map_location=torch.device("cpu"))
#         model.load_state_dict(state_dict)

#     return model
