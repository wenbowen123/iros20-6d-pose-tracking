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

import os, sys,time
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/../../')
import json,yaml
from torch.utils.data.dataset import Dataset
from PIL import Image
from Utils import *
from data_augmentation import *
import numpy as np
import cv2
import glob
import torch
import transformations as T

class TrackDataset(Dataset):
	def __init__(self, root,mode,images_mean, images_std, pretransforms=None, augmentations=None, posttransforms=None, dataset_info=None, trans_normalizer=0.03, rot_normalizer=5*np.pi/180):
		self.mode = mode
		self.images_mean = images_mean
		self.images_std = images_std
		self.data_pose = []
		self.data_pair = {}
		self.pretransforms = pretransforms
		self.augmentations = augmentations
		self.posttransforms = posttransforms
		self.root = root
		self.data_transforms = []
		self.dataset_info = dataset_info
		if dataset_info!=None:
			self.cam_K = cam_K_from_dict(dataset_info['camera'])
			print('self.cam_K:\n',self.cam_K)
		else:
			print('[WARN] In TrackDataset, dataset_info is None !!')
		print("making dataset... for {}".format(self.mode))

		self.rgbA_files = sorted(glob.glob(self.root+'/*rgbA.png'))

		print('#dataset:',self.__len__())

		self.trans_normalizer = trans_normalizer
		self.rot_normalizer = rot_normalizer    # 30*np.pi/180 for YCBInEOAT
		print("self.trans_normalizer={}, self.rot_normalizer={}".format(self.trans_normalizer,self.rot_normalizer))



	def __getitem__(self, index):
		"""
		@data: [rgbA, depthA, rgbB, depthB, poseA]. Depth data unit is mm (uint16).
		@target [pose_labels]
		"""
		maskB = None
		rgbB = np.array(Image.open(self.rgbA_files[index].replace('A','B')))
		depthB = cv2.imread(self.rgbA_files[index].replace('rgbA','depthB'), cv2.IMREAD_UNCHANGED)
		maskB = cv2.imread(self.rgbA_files[index].replace('rgbA','segB'), cv2.IMREAD_UNCHANGED)
		meta = np.load(self.rgbA_files[index].replace('rgbA.png','meta.npz'))
		B_in_cam = meta['B_in_cam']
		rgbA = np.array(Image.open(self.rgbA_files[index]))
		depthA = cv2.imread(self.rgbA_files[index].replace('rgbA','depthA'), cv2.IMREAD_UNCHANGED)
		A_in_cam = meta['A_in_cam']

		if rgbB.shape[0]!=self.dataset_info['resolution']:
			resolution = self.dataset_info['resolution']
			rgbA = cv2.resize(rgbA,(resolution,resolution),interpolation=cv2.INTER_NEAREST)
			rgbB = cv2.resize(rgbB,(resolution,resolution),interpolation=cv2.INTER_NEAREST)
			depthA = cv2.resize(depthA,(resolution,resolution),interpolation=cv2.INTER_NEAREST)
			depthB = cv2.resize(depthB,(resolution,resolution),interpolation=cv2.INTER_NEAREST)
			maskB = cv2.resize(maskB,(resolution,resolution),interpolation=cv2.INTER_NEAREST)

		if maskB is None:
			maskB = (depthB>100).astype(np.uint8)
		assert np.sum(maskB)>0, 'index={}'.format(index)

		data, target, rgbA, rgbB, maskA, maskB = self.processData(rgbA,depthA,A_in_cam,rgbB,depthB,B_in_cam,maskB)
		return data, target, A_in_cam, B_in_cam, rgbA, rgbB, maskA, maskB


	def __len__(self):
		return len(self.rgbA_files)


	def processData(self,rgbA,depthA,A_in_cam,rgbB,depthB,B_in_cam,maskB=None,original_size=None):
		'''
		After posttransforms,
		@sample: [dataA, dataB] where dataA and dataB are concatenated rgb with depth CHW
		@A_in_cam: 4x4 mat
		'''
		maskA = (depthA>100).astype(np.uint8)
		if maskB is None:
			maskB = (depthB>100).astype(np.uint8)
		sample = [rgbA, depthA, rgbB, depthB,maskA,maskB,A_in_cam]

		if self.pretransforms:
			sample = self.pretransforms(sample)
		rgbA_viz = rgbA.astype(np.uint8)
		rgbB_viz = rgbB.astype(np.uint8)
		if self.augmentations:
			sample = self.augmentations(sample)
			rgbA, depthA, rgbB, depthB, maskA, maskB, prior = sample
			rgbA_viz = rgbA.astype(np.uint8)
			rgbB_viz = rgbB.astype(np.uint8)
		if self.posttransforms:
			sample,maskA,maskB = self.posttransforms(sample)

		trans_label = np.zeros((3))
		rot_label = np.zeros((3))

		trans_label = B_in_cam[:3,3] - A_in_cam[:3,3]
		trans_label /= self.trans_normalizer   # Normalize

		A2B_in_cam_rot = np.eye(3)
		A2B_in_cam_rot = B_in_cam[:3,:3].dot(A_in_cam[:3,:3].T)
		A2B_in_cam_rot = normalize_rotation_matrix(A2B_in_cam_rot)

		rod = cv2.Rodrigues(A2B_in_cam_rot)[0].reshape(-1)
		rod = rod/self.rot_normalizer
		rot_label = rod

		if self.mode=='train':
			assert (trans_label<=1).all() and (trans_label>=-1).all()
			assert (rot_label>=-1).all() and (rot_label<=1).all(),'root:\n{}\nrot_label\n{}\n A2B_in_cam_rot{}\n'.format(self.root,rot_label,A2B_in_cam_rot)

		return sample, [trans_label, rot_label], rgbA_viz, rgbB_viz, maskA, maskB


	def processPredict(self,A_in_cam,predB,original_size=None):
		'''Recover the predicted pose to the true pose
		@A_in_cam: 4x4 mat
		@predB: trans, rot, ...
		return ob pose in cam frame
		'''
		B_in_cam = np.eye(4)
		trans_pred = predB[0]
		rot_pred = predB[1]

		trans_pred = trans_pred*self.trans_normalizer
		B_in_cam[:3,3] = trans_pred+A_in_cam[:3,3]

		rot_pred = rot_pred*self.rot_normalizer
		A2B_in_cam_rot = cv2.Rodrigues(rot_pred)[0].reshape(3,3)
		B_in_cam[:3,:3] = A2B_in_cam_rot.dot(A_in_cam[:3,:3])
		return B_in_cam




