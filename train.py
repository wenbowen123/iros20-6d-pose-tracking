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
import sys,shutil
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from datasets import *
from multiprocessing import cpu_count
import argparse
from problems import *
from network_modules import *
from se3_tracknet import *
import torch
from torch import optim
import torch.utils.data
import numpy as np
import yaml
import glob
import random
from data_augmentation import *
with open(dir_path+'/config.yml', 'r') as ff:
	config = yaml.safe_load(ff)


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True


if __name__=="__main__":
	data_path = config['data_path']
	validation_path = config['validation_path']
	dir_path = os.path.dirname(os.path.realpath(__file__))
	output_path = f"{dir_path}/train_output/"
	print('output_path',output_path)

	if not os.path.exists(output_path):
		os.makedirs(output_path)

	with open(data_path+'/../dataset_info.yml', 'r') as ff:
		dataset_info = yaml.safe_load(ff)
		print('loaded dataset info from:',data_path+'/../dataset_info.yml')
	shutil.copy(data_path+'/../dataset_info.yml',f'{output_path}/dataset_info.yml')

	hsv_noise = config['data_augmentation']['hsv_noise']
	batch_size = config['batch_size']
	n_workers = config['n_workers']

	augmentations = Compose([
							HSVJitter(hsv_noise[0],hsv_noise[1],hsv_noise[2]),
							ChangeBright(prob=0.5,mag=[config['data_augmentation']['bright_mag'][0], config['data_augmentation']['bright_mag'][1]]),
							GaussianNoise(config['data_augmentation']['gaussian_noise']['rgb'], config['data_augmentation']['gaussian_noise']['depth']),
							GaussianBlur(config['data_augmentation']['gaussian_blur_kernel']),
							BlackCover(prob=0.2),
							# DepthMissing(prob=0.5,missing_percent=config['data_augmentation']['depth_missing_percent']),
							])

	############ Compute mean std
	posttransforms = Compose([OffsetDepth(),Transpose(),ToTensor()])  #NOTE Transpose() make image CxHxW
	train_dataset = TrackDataset(data_path,mode='train',images_mean=None,images_std=None,pretransforms=None, augmentations=augmentations, posttransforms=posttransforms, dataset_info=dataset_info)
	print('len(train_dataset)=',len(train_dataset))
	train_loader = torch.utils.data.DataLoader(train_dataset,
																batch_size=batch_size,
																shuffle=False,
																num_workers=n_workers,
																pin_memory=False,
																drop_last=True,
																)

	n = 10000
	print('Computing mean std for n={}'.format(n))
	total = 0
	image_means_array = []
	for i, (data, target, A_in_cams, B_in_cams, rgbA, rgbB, maskA, maskB) in enumerate(train_loader):
		bufferA, bufferB = data
		bufferA_numpy = bufferA.cpu().numpy()[:,:4]
		bufferB_numpy = bufferB.cpu().numpy()[:,:4]
		buffer_numpy = np.concatenate((bufferA_numpy, bufferB_numpy), axis=1)  #NxCxHxW
		image_means = np.mean(buffer_numpy, axis=(0, 2, 3))
		image_means_array.append(image_means)
		total += 1
		if i * batch_size >= n:
				break

	images_mean = np.mean(image_means_array, axis=0)
	images_std = np.std(image_means_array, axis=0)

	np.save("{}/mean.npy".format(output_path), images_mean)
	np.save("{}/std.npy".format(output_path), images_std)
	print('images_mean\n',images_mean)
	print('images_std\n',images_std)


	posttransforms = Compose([OffsetDepth(),NormalizeChannels(images_mean,images_std),ToTensor()])
	train_dataset = TrackDataset(data_path,'train',images_mean,images_std,None,augmentations,posttransforms,dataset_info=dataset_info)
	valid_dataset = TrackDataset(validation_path,'val',images_mean,images_std,None,augmentations,posttransforms,dataset_info=dataset_info)

	with open(os.path.join(output_path, "config_backup.yml"), 'w') as ff:
		yaml.dump(config, ff)

	train_loader = torch.utils.data.DataLoader(train_dataset,
								   batch_size=batch_size,
								   shuffle=True,
								   num_workers=n_workers,
								   pin_memory=False,
								   drop_last=True,
								   )
	assert len(train_loader)>0
	val_loader = torch.utils.data.DataLoader(valid_dataset,
								 batch_size=200,
								 shuffle=False,
								 num_workers=n_workers,
								 pin_memory=False,
								 drop_last=False,
								 )

	model = Se3TrackNet(image_size=int(dataset_info["resolution"]))
	model = model.cuda()

	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'], betas=(0.9, 0.99), amsgrad=False)


	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300], gamma=0.1)

	problem = Problem(model, train_loader, val_loader,config=config,optimizer=optimizer, scheduler=scheduler)

	print("Training Begins:")
	problem.loop(config['epochs'], output_path, save_all_checkpoints=True)
	print("Training Complete")
