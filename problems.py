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


import open3d as o3d
import os,sys,gc
import glob
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from data_augmentation import *
from network_modules import *
from Utils import *
from tensorboardX import SummaryWriter
import time
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from PIL import Image

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Problem:
	def __init__(self, model, train_data_loader, valid_data_loader, config=None, optimizer=None, scheduler=None):
		self.train_data = train_data_loader
		self.valid_data = valid_data_loader
		self.optimizer = optimizer
		self.model = model
		self.scheduler = scheduler
		self.global_step = 0

		self.model = self.model.cuda()
		self.config = config
		self.loss_weights = self.config['loss_weights']
		self.best_eval = np.inf
		self.best_val = np.inf
		self.best_train = np.inf
		self.dataset_info = train_data_loader.dataset.dataset_info
		self.K = self.train_data.dataset.cam_K


	def train(self, epoch):
		self.model.train()

		for data, target, A_in_cams, B_in_cams, rgbA, rgbB, maskA, maskB in self.train_data:
			dataA = data[0]
			dataB = data[1]
			dataA = dataA.cuda()
			dataB = dataB.cuda()
			for i in range(len(target)):
				target[i] = target[i].cuda()
			pred = self.model(dataA,dataB)
			output = self.model.loss((pred['trans'],pred['rot']), target)
			loss = output['trans']*self.loss_weights['trans']+output['rot']*self.loss_weights['rot']

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			if self.global_step%100==0:
				print("epoch={}, iter={},   loss={}".format(epoch, self.global_step, loss.data))
				print('trans_label[0] =',target[0][0].data.cpu().numpy().reshape(-1))
				print('rot_label[0] =',target[1][0].data.cpu().numpy().reshape(-1))

			self.global_step += 1
		return loss.data


	def validate(self, epoch):
		self.model.eval()
		trans_losses = []
		rot_losses = []
		trans_uncertain_losses = []
		rot_uncertain_losses = []
		seg_losses = []
		adi_losses = []

		with torch.no_grad():
			for data, target, A_in_cams, B_in_cams, rgbA, rgbB, maskA, maskB in self.valid_data:
				dataA = data[0]
				dataB = data[1]
				dataA = dataA.cuda()
				dataB = dataB.cuda()
				for i in range(len(target)):
					target[i] = target[i].cuda()
				pred = self.model(dataA,dataB)
				output = self.model.loss((pred['trans'],pred['rot']), target)
				trans_losses.append(output['trans'].cpu().item())
				rot_losses.append(output['rot'].cpu().item())

		trans_loss = np.array(trans_losses).mean()
		rot_loss = np.array(rot_losses).mean()
		total_loss = trans_loss*self.loss_weights['trans'] + rot_loss*self.loss_weights['rot']

		return total_loss


	def loop(self, total_epochs, output_path, save_all_checkpoints=False):
		for epoch in range(0, total_epochs):
			print(">>>>>>>>>>>>>> epoch {}".format(epoch))
			train_loss = self.train(epoch)
			validation_loss_average = self.validate(epoch)
			if train_loss<self.best_train:
				self.best_train = train_loss
				checkpoint_data = {'state_dict': self.model.state_dict()}
				dir = "{}/model_best_train.pth.tar".format(output_path)
				torch.save(checkpoint_data, dir)

			is_val_best = validation_loss_average < self.best_val
			self.best_val = min(validation_loss_average, self.best_val)
			if is_val_best:
				checkpoint_data = {'state_dict': self.model.state_dict()}
				dir = "{}/model_best_val.pth.tar".format(output_path)
				torch.save(checkpoint_data, dir)
			if self.scheduler:
				self.scheduler.step()



