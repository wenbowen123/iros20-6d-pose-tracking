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

import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir+'/../')
import numpy as np
import glob
import scipy.io
import numpy as np
from Utils import *
from eval_ycb import VOCap
import argparse



def eval_all(args):
  res_dir = args.res_dir
  data_dir = '{}/'.format(args.YCBInEOAT_dir)
  objects = ['cracker','bleach','sugar','tomato','mustard']
  models = {}
  tmp = glob.glob('{}/CADmodels/*/points.xyz'.format(args.ycb_dir))
  for t in tmp:
    with open(t,'r') as ff:
      lines = ff.readlines()
    model_pts = []
    for i in range(len(lines)):
      line = list(map(float,lines[i].rstrip().split()))
      model_pts.append(line)
    model_pts = np.array(model_pts)
    model_pts.reshape(-1,3)
    model = U.toOpen3dCloud(model_pts,colors=np.zeros(model_pts.shape,dtype=np.float64))
    for obj in objects:
      if obj in t:
        models[obj] = model

  class_res = {}
  for obj in objects:
    class_res[obj] = {'add':[],'add-s':[]}

  folders = os.listdir(res_dir)

  for folder in folders:
    if '.tar.gz' in folder:
      continue
    print(folder)
    pred_files = sorted(glob.glob(res_dir+folder+'/*.txt'))
    obj = None
    for o in objects:
      if o in folder:
        gt_files = sorted(glob.glob(data_dir+folder+"/annotated_poses/*.txt"))
        obj = o
        break
    assert len(pred_files)==len(gt_files),'#pred_files:{}, #gt_files:{}'.format(len(pred_files),len(gt_files))
    for i in range(len(pred_files)):
      pred = np.loadtxt(pred_files[i])
      gt = np.loadtxt(gt_files[i])
      add = U.add(pred,gt,models[obj])
      adi = U.adi(pred,gt,models[obj])
      class_res[obj]['add'].append(add)
      class_res[obj]['add-s'].append(adi)

  adds = []
  adis = []
  for k in class_res.keys():
    adi = class_res[k]['add-s']
    adis += adi
    adi_auc = VOCap(adi) * 100
    add = class_res[k]['add']
    adds += add
    add_auc = VOCap(add) * 100
    print('{}: adi={} add={}'.format(k,adi_auc,add_auc))

  adi_auc = VOCap(adis) * 100
  add_auc = VOCap(adds) * 100
  print('Total pose:',len(adis))
  print('\nOverall, adi={} add={}'.format(adi_auc,add_auc))



if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--YCBInEOAT_dir', default='/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/catkin_ws/src/iros20_dataset/video_rosbag/IROS_SELECTED/FINISHED_LABEL.iros_submission_version')
  parser.add_argument('--ycb_dir', default='/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/DATASET/Tracking/YCB_Video_Dataset')
  parser.add_argument('--class_id',type=int,default=1)
  parser.add_argument('--res_dir',type=str,default='/home/bowen/debug/ycb_results')
  args = parser.parse_args()


  eval_all(args)






