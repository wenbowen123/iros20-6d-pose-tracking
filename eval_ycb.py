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
sys.path.append(code_dir)
import numpy as np
import glob
import Utils as U
import argparse


def VOCap(rec):
  rec = np.sort(np.array(rec))
  n = len(rec)
  prec = np.arange(1,n+1) / float(n)
  rec = rec.reshape(-1)
  prec = prec.reshape(-1)
  index = np.where(rec<0.1)[0]
  rec = rec[index]
  prec = prec[index]

  mrec=[0, *list(rec), 0.1]
  mpre=[0, *list(prec), prec[-1]]

  for i in range(1,len(mpre)):
    mpre[i] = max(mpre[i], mpre[i-1])
  mpre = np.array(mpre)
  mrec = np.array(mrec)
  i = np.where(mrec[1:]!=mrec[0:len(mrec)-1])[0] + 1
  ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
  return ap


def eval_one_class(args):
  pose_files = sorted(glob.glob(args.res_dir+'**/*.txt',recursive=True))
  assert len(pose_files)>0,'args.res_dir is\n{}'.format(args.res_dir)
  class_names = sorted(os.listdir('{}/CADmodels/'.format(args.ycb_dir)))
  model_files = sorted(glob.glob('{}/CADmodels/**/points.xyz'.format(args.ycb_dir),recursive=True))
  with open(model_files[args.class_id-1],'r') as ff:
    lines = ff.readlines()

  model_pts = []
  for i in range(len(lines)):
    line = list(map(float,lines[i].rstrip().split()))
    model_pts.append(line)
  model_pts = np.array(model_pts)
  model_pts.reshape(-1,3)
  model = U.toOpen3dCloud(model_pts,colors=np.zeros(model_pts.shape,dtype=np.float64))

  keyframes = []
  with open('{}/YCB_Video_toolbox/keyframe.txt'.format(args.ycb_dir),'r') as ff:
    lines = ff.readlines()
    for i in range(len(lines)):
      line = lines[i].rstrip()
      keyframes.append(line)

  adi_errs = []
  add_errs = []

  for i in range(len(pose_files)):
    pose_file = pose_files[i]
    seq_id = int(pose_file.replace(args.res_dir,'').split('/')[0].replace('seq',''))
    frame_id = int(os.path.basename(pose_file).split('.')[0])+1
    seq_frame_str = '%04d/%06d'%(seq_id,frame_id)
    if seq_frame_str not in keyframes:
      continue
    pred = np.loadtxt(pose_file)
    gt_file = '{}/data_organized/%04d/pose_gt/{}/%06d.txt'.format(args.ycb_dir,args.class_id)%(seq_id,frame_id)
    gt_pose = np.loadtxt(gt_file)
    adi_err = U.adi(pred,gt_pose,model)
    adi_errs.append(adi_err)
    add_err = U.add(pred,gt_pose,model)
    add_errs.append(add_err)

  adi_errs = np.sort(np.array(adi_errs))
  add_errs = np.sort(np.array(add_errs))


  assert len(adi_errs)>0
  add_aps = VOCap(add_errs) * 100
  print('>>>>>>>>>>>>>>>> args.class_id:',args.class_id, class_names[args.class_id-1])
  print('add:',add_aps)

  adi_aps = VOCap(adi_errs) * 100
  print('adi:',adi_aps)
  return adi_errs,add_errs

def eval_all(args):
  class_ids = np.arange(1,22)
  print(class_ids)

  root = '/home/bowen/debug/Ours/'
  class_folders = sorted(os.listdir(root))
  res_dirs = []
  for class_folder in class_folders:
    folders = os.listdir(root+class_folder)
    print(folders)
    for folder in folders:
      if os.path.isdir(root+class_folder+'/'+folder):
        res_dirs.append(root+class_folder+'/'+folder+'/')
        break

  for res_dir in res_dirs:
    print(res_dir)

  assert len(res_dirs)==len(class_ids),'len(res_dirs)={}'.format(len(res_dirs))
  adi_errs = []
  add_errs = []

  for i,class_id in enumerate(class_ids):
    args.res_dir = res_dirs[i]
    args.class_id = class_id
    res = eval_one_class(args)
    adi_errs += list(res[0])
    add_errs += list(res[1])

  adi_errs = np.array(adi_errs)
  add_errs = np.array(add_errs)

  n = len(adi_errs)
  assert(n==14025)

  add_aps = VOCap(add_errs) * 100
  print()
  print('add:',add_aps)

  adi_aps = VOCap(adi_errs) * 100
  print('adi:',adi_aps)
  print('Total res num:',n)

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--ycb_dir', default='/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/DATASET/Tracking/YCB_Video_Dataset')
  parser.add_argument('--class_id',type=int,default=1)
  parser.add_argument('--res_dir',type=str,default='/home/bowen/debug/ycb_results')
  args = parser.parse_args()

  # eval_one_class(args)
  eval_all(args)






