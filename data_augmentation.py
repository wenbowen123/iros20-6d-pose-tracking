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


import scipy.signal
import scipy.stats
import torch
import sys,os,time
dir_path = os.path.dirname(os.path.realpath(__file__))
from PIL import Image
import numpy as np
import cv2
from Utils import *


class HSVJitter(object):
	def __init__(self, h_noise, s_noise, v_noise, prob=0.5):
		self.prob = prob
		self.h_noise = h_noise
		self.s_noise = s_noise
		self.v_noise = v_noise

	def __call__(self, data):
		rgbA, depthA, rgbB, depthB, maskA, maskB, poseA = data
		mask = depthB>100
		hsv = cv2.cvtColor(rgbB, cv2.COLOR_RGB2HSV).astype(np.float32)
		H = hsv.shape[0]
		W = hsv.shape[1]
		if np.random.uniform() < self.prob:
			hsv[:, :, 0] += np.random.uniform(-self.h_noise, self.h_noise)
		if np.random.uniform() < self.prob:
			hsv[:, :, 1] += np.random.uniform(-self.s_noise, self.s_noise)
		if np.random.uniform() < self.prob:
			hsv[:, :, 2] += np.random.uniform(-self.v_noise, self.v_noise)
		hsv = np.clip(hsv,0,255)
		rgbB[mask] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)[mask]
		rgbB = rgbB.astype(np.uint8)
		return rgbA, depthA, rgbB, depthB, maskA, maskB,poseA


class ChangeBright:
	def __init__(self,prob=0.5,mag=[0.5,1.5]):
		self.mag = mag

	def __call__(self,data):
		rgbA, depthA, rgbB, depthB, maskA, maskB, poseA = data
		rgbB = rgbB*random.uniform(self.mag[0],self.mag[1])
		rgbB = np.clip(rgbB,0,255).astype(np.uint8)
		return rgbA, depthA, rgbB, depthB, maskA, maskB,poseA



class GaussianNoise(object):
	def __init__(self, rgb_noise, depth_noise, prob=0.5):
		self.rgb_noise = rgb_noise
		self.depth_noise = depth_noise
		self.prob = prob

	def __call__(self, data):
		rgbA, depthA, rgbB, depthB,maskA, maskB, poseA = data
		mask = depthB>100
		if np.random.uniform() < self.prob:
			std = np.random.uniform(0, self.rgb_noise)
			noise = np.random.normal(0, std, size=rgbA.shape)
			rgbB[mask] = rgbB[mask] + noise[mask]
		if np.random.uniform() < self.prob:
			std = np.random.uniform(0, self.depth_noise)
			noise = np.random.normal(0, std, size=depthB.shape)
			depthB[mask] = depthB[mask] + noise[mask]
		return rgbA, depthA, rgbB, depthB, maskA, maskB,poseA


class GaussianBlur(object):
	def __init__(self, max_kernel_size, min_kernel_size=3, prob=0.4):
		self.prob = prob
		self.max_kernel_size = max_kernel_size
		self.min_kernel_size = 3

	def __call__(self, data):
		rgbA, depthA, rgbB, depthB, maskA, maskB,poseA = data
		if np.random.uniform() < self.prob:
			ksize = np.random.randint(1, self.max_kernel_size//2+1)
			ksize = 2*ksize+1
			rgbB = cv2.GaussianBlur(rgbB, (ksize,ksize), sigmaX=2)
		if np.random.uniform() < self.prob:
			ksize = np.random.randint(1, self.max_kernel_size//2+1)
			ksize = 2*ksize+1
			depthB = cv2.GaussianBlur(depthB, (ksize,ksize), sigmaX=2)
		return rgbA, depthA, rgbB, depthB, maskA, maskB,poseA


class OffsetDepth(object):
	def __init__(self):
		pass

	def __call__(self, data):
		rgbA, depthA, rgbB, depthB, maskA, maskB,poseA = data
		depthA = self.normalize_depth(depthA, poseA)
		depthB = self.normalize_depth(depthB, poseA)
		return rgbA.astype(np.float32), depthA, rgbB.astype(np.float32), depthB, maskA, maskB, poseA

	def normalize_depth(self, depth, pose):
		depth = depth.astype(np.float32)
		invalid_mask = np.logical_or(depth<=100, depth>=2000)
		if pose[2, 3]<0:  #gl pose
			depth += pose[2, 3] * 1000
		else:
			depth -= pose[2, 3] * 1000

		depth[invalid_mask] = 2000
		assert (depth<=2000).all()
		return depth



class NormalizeChannels(object):
	def __init__(self,mean,std):
		self.mean = mean
		self.std = std


	def __call__(self, data):
		rgbA, depthA, rgbB, depthB, maskA, maskB, poseA = data
		rgbA, depthA = self.normalize_channels(rgbA, depthA, self.mean[:4], self.std[:4])
		rgbB, depthB = self.normalize_channels(rgbB, depthB, self.mean[4:], self.std[4:])
		return rgbA, depthA, rgbB, depthB, maskA, maskB,poseA

	def normalize_channels(self, rgb, depth, mean, std):
		rgb = rgb.transpose(2,0,1)
		rgb = (rgb-mean[:3, np.newaxis, np.newaxis])/std[:3, np.newaxis, np.newaxis]
		depth = (depth-mean[3, np.newaxis, np.newaxis])/std[3, np.newaxis, np.newaxis]
		return rgb, depth


class Transpose(object):
	def __call__(self, data):
		rgbA, depthA, rgbB, depthB,maskA, maskB, poseA = data
		rgbA = rgbA.transpose(2,0,1)
		rgbB = rgbB.transpose(2,0,1)
		return rgbA, depthA, rgbB, depthB, maskA, maskB,poseA


class ToTensor(object):
	def __init__(self):
		pass

	def __call__(self, data):
		rgbA, depthA, rgbB, depthB, maskA, maskB, poseA = data
		bufferA = np.zeros((4, rgbA.shape[1], rgbA.shape[2]), dtype=np.float32)
		bufferA[0:3, :, :] = rgbA
		bufferA[3, :, :] = depthA
		bufferB = np.zeros((4, rgbA.shape[1], rgbA.shape[2]), dtype=np.float32)
		bufferB[0:3, :, :] = rgbB
		bufferB[3, :, :] = depthB
		bufferA = torch.from_numpy(bufferA)
		bufferB = torch.from_numpy(bufferB)
		return [bufferA, bufferB], maskA, maskB

	def to_tensor(self, rgb,depth):
		buffer = np.zeros((4, rgb.shape[1], rgb.shape[2]), dtype=np.float32)
		buffer[0:3, :, :] = rgb
		buffer[3, :, :] = depth
		buffer = torch.from_numpy(buffer).float()
		return buffer



class DepthMissing():
	def __init__(self,prob=0.5,missing_percent=0.5):
		self.prob = prob
		self.missing_percent = missing_percent

	def __call__(self,data):
		rgbA, depthA, rgbB, depthB,maskA, maskB, poseA = data
		W = depthB.shape[1]
		H = depthB.shape[0]
		us,vs = np.where(depthB>100)
		if np.random.uniform(0,1)<self.prob:
			missing_percent = np.random.uniform(0,self.missing_percent)
			missing_ids = np.random.choice(np.arange(0,len(us)), int(missing_percent*len(us)), replace=False)
			depthB[vs[missing_ids], us[missing_ids]] = 0
		return rgbA, depthA, rgbB, depthB, maskA, maskB,poseA


class BlackCover():
	'''Random black cover to imitate cases of object outside of image.
	'''
	def __init__(self, prob=0.3):
		self.prob = prob

	def __call__(self,data):
		rgbA, depthA, rgbB, depthB, maskA, maskB,prior = data
		rgbB_backup = rgbB.copy()
		depthB_backup = depthB.copy()
		maskB_backup = maskB.copy().astype(np.uint8)
		num_valid = np.sum(maskB_backup)
		if np.random.uniform(0,1) >= self.prob:
			return rgbA, depthA, rgbB, depthB, maskA, maskB,prior
		H = rgbB.shape[0]
		W = rgbB.shape[1]
		corner_uv = (np.random.randint(0,W),np.random.randint(0,H))
		tlbr = np.random.choice([0,1,2,3])
		i = tlbr
		while True:
			if i==0: #top left
				rgbB[:corner_uv[1],:corner_uv[0],:] = 0
				depthB[:corner_uv[1],:corner_uv[0]] = -9999
				maskB[:corner_uv[1],:corner_uv[0]] = 0
			elif i==1:  #top right
				rgbB[:corner_uv[1],corner_uv[0]:,:] = 0
				depthB[:corner_uv[1],corner_uv[0]:] = -9999
				maskB[:corner_uv[1],corner_uv[0]:] = 0
			elif i==2:  #bottom left
				rgbB[corner_uv[1]:,:corner_uv[0],:] = 0
				depthB[corner_uv[1]:,:corner_uv[0]] = -9999
				maskB[corner_uv[1]:,:corner_uv[0]] = 0
			elif i==3:
				rgbB[corner_uv[1]:,corner_uv[0]:,:] = 0
				depthB[corner_uv[1]:,corner_uv[0]:] = -9999
				maskB[corner_uv[1]:,corner_uv[0]:] = 0
			remainedB_valid = maskB==1
			if np.sum(remainedB_valid)/float(num_valid)<0.5:  # Make sure at least remain some visibility of object
				rgbB = rgbB_backup.copy()
				depthB = depthB_backup.copy()
				maskB = maskB_backup.copy()
				i += 1
				i = i%4
			else:
				break
			if i==tlbr:
				corner_uv = (np.random.randint(0,W),np.random.randint(0,H))
				tlbr = np.random.choice([0,1,2,3])
				i = tlbr

		return rgbA, depthA, rgbB, depthB, maskA, maskB,prior