#
# Authors: Bowen Wen
# Contact: wenbowenxjtu@gmail.com
# Created in 2020
#
# Copyright (c) Rutgers University, 2020 All rights reserved.
#
# Wen, B., C. Mitash, B. Ren, and K. E. Bekris. "se (3)-TrackNet:
# Data-driven 6D Pose Tracking by Calibrating Image Residuals in
# Synthetic Domains." In IEEE/RSJ International Conference on Intelligent
# Robots and Systems (IROS). 2020.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the PRACSYS, Bowen Wen, Rutgers University,
#       nor the names of its contributors may be used to
#       endorse or promote products derived from this software without
#       specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


import os,sys,copy
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import torch.nn.functional as F
from torch.autograd import Function
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import cv2

class ConvBN(nn.Sequential):
		def __init__(self, C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True,dilation=1,):
				padding = (kernel_size - 1) // 2
				super(ConvBN, self).__init__(
						nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias,dilation=dilation),
						nn.BatchNorm2d(C_out),
				)



class ConvBNReLU(nn.Sequential):
		def __init__(self, C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True,dilation=1,):
				padding = (kernel_size - 1) // 2
				super(ConvBNReLU, self).__init__(
						nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias,dilation=dilation),
						nn.BatchNorm2d(C_out),
						nn.SELU(inplace=True)
				)



class ConvPadding(nn.Module):
	def __init__(self,C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True,dilation=1):
		super(ConvPadding, self).__init__()
		padding = (kernel_size - 1) // 2
		self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias,dilation=dilation)

	def forward(self,x):
		return self.conv(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


class ResnetBasicBlock(nn.Module):
	__constants__ = ['downsample']

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, bias=False):
		super().__init__()
		if norm_layer is None:
						norm_layer = nn.BatchNorm2d
		if groups != 1 or base_width != 64:
						raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		if dilation > 1:
						raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		self.conv1 = conv3x3(inplanes, planes, stride,bias=bias)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes,bias=bias)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)
		out += identity
		out = self.relu(out)

		return out


