import open3d as o3d
import sys,shutil,pickle
import os
from scipy import spatial
import argparse
import torch
import numpy as np
import yaml
from offscreen_renderer import *
from Utils import *
import time
import numpy as np
import cv2
from PIL import Image
import copy
import glob




class ProducerPurturb:
  '''This can be used both as eval or purturb on training data to get large training set
  '''
  def __init__(self,dataset_info,check_vis=False):
    self.count = 0
    self.check_vis = check_vis
    self.dataset_info = dataset_info
    self.image_size = (self.dataset_info['resolution'],self.dataset_info['resolution'])
    model_3d_path = self.dataset_info['models'][0]['model_path']
    if '.ply' not in model_3d_path:
      model_3d_path = model_3d_path+'/geometry.ply'

    self.object_width = dataset_info['object_width']

    print('self.object_width=',self.object_width)
    self.cam_K = np.zeros((3,3)).astype(np.float32)
    self.cam_K[0,0] = self.dataset_info['camera']['focalX']
    self.cam_K[1,1] = self.dataset_info['camera']['focalY']
    self.cam_K[0,2] = self.dataset_info['camera']['centerX']
    self.cam_K[1,2] = self.dataset_info['camera']['centerY']
    self.cam_K[2,2] = 1
    print('self.cam_K:\n',self.cam_K)
    obj_path = self.dataset_info['models'][0]['obj_path']
    print('obj_path',obj_path)
    self.renderer = ModelRendererOffscreen([obj_path],self.cam_K,dataset_info['camera']['height'],dataset_info['camera']['width'])
    self.glcam_in_cvcam = np.array([[1,0,0,0],
                                    [0,-1,0,0],
                                    [0,0,-1,0],
                                    [0,0,0,1]])



  def generate(self,out_dir,B_in_cam,current_rgb,current_depth,num_sample,class_id,current_seg=None,debug=False):
    '''
    Take one real image and sample various purturbation around for evaluating the mean error
    '''
    max_trans = self.dataset_info['max_translation']
    max_rot = self.dataset_info['max_rotation']  #degree

    H = self.dataset_info['camera']['height']
    W = self.dataset_info['camera']['width']

    #========================= Check visibility =============================
    if self.check_vis:
      num_visible = np.sum(current_seg==class_id)
      if num_visible<=100:
        return
      color, depth = self.renderer.render([B_in_cam])
      visible_ratio = num_visible/float(np.sum(depth>0.1))
      if visible_ratio<0.1:
        return

    pts = []
    rot_pts = []
    for i in range(num_sample):
      # if self.count%100==0:
      #   print('>>>>>>>>>>>>>>>> processing ',self.count)
      B_in_A = random_gaussian_magnitude(max_trans, max_rot)
      A_in_cam = B_in_cam.dot(np.linalg.inv(B_in_A))
      bb_ortho = compute_2Dboundingbox(A_in_cam, self.cam_K, self.object_width, scale=(1000, -1000, 1000))
      left = np.min(bb_ortho[:, 1])
      right = np.max(bb_ortho[:, 1])
      top = np.min(bb_ortho[:, 0])
      bottom = np.max(bb_ortho[:, 0])

      projected = self.cam_K.dot(A_in_cam[:3,3].reshape(3,1)).reshape(-1)
      u = projected[0]/projected[2]
      v = projected[1]/projected[2]
      if u<0 or u>=W or v<0 or v>=H:
        continue

      bb = compute_2Dboundingbox(A_in_cam, self.cam_K, self.object_width, scale=(1000, 1000, 1000))
      if ('renderer' not in self.dataset_info) or (self.dataset_info['renderer']=='vispy'):
        self.renderer.setup_camera(self.cam_K, left, right, bottom, top)
        A_in_glcam = np.linalg.inv(self.glcam_in_cvcam).dot(A_in_cam)
        rgbA, depthA = self.renderer.render_image(A_in_glcam, fbo_index=1)
      else:
        rgb, depth = self.renderer.render([A_in_cam])
        depth = (depth*1000).astype(np.uint16)
        rgbA,depthA = normalize_scale(rgb, depth, bb, self.image_size)
      depthA = depthA.astype(np.uint16)

      if current_seg is not None:
        rgbB, depthB, segB = normalize_scale(current_rgb, current_depth, bb, self.image_size, current_seg)
      else:
        rgbB, depthB = normalize_scale(current_rgb, current_depth, bb, self.image_size)
      if np.sum(segB==class_id)<100:
        continue
      depthB = depthB.astype(np.uint16)

      Image.fromarray(rgbA).save(out_dir+'%07drgbA.png'%(self.count),optimize=True)
      Image.fromarray(rgbB).save(out_dir+'%07drgbB.png'%(self.count),optimize=True)
      cv2.imwrite(out_dir+'%07ddepthA.png'%(self.count),depthA)
      cv2.imwrite(out_dir+'%07ddepthB.png'%(self.count),depthB)
      np.savez(out_dir+'%07dmeta.npz'%(self.count),A_in_cam=A_in_cam,B_in_cam=B_in_cam)
      if current_seg is not None:
        segB = (segB==class_id).astype(np.uint8)
        cv2.imwrite(out_dir+'%07dsegB.png'%(self.count),segB)

      self.count += 1

      if debug:
        pts.append(B_in_A[:3,3].reshape(-1))
        vec = B_in_A[:3,:3].dot(np.array([0,0,1]).reshape(3,1)).reshape(-1)
        rot_pts.append(vec)

    if debug:
      from PointCloud import PointCloudClass
      pts = np.array(pts)
      rot_pts = np.array(rot_pts)
      pcd = PointCloudClass(points=pts,colors=np.zeros_like(pts),normals=np.zeros_like(pts))
      pcd.writePLY('/home/bowen/debug/trans_pts.ply')
      pcd = PointCloudClass(points=rot_pts,colors=np.zeros_like(pts),normals=np.zeros_like(pts))
      pcd.writePLY('/home/bowen/debug/rot_pts.ply')



def completeBlenderYcbDR():
  '''Domain Randomization
  '''
  class_id = 13
  data_folder = '/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/github/end_to_end_tracking/Experiments/MINE/bowl/'

  dataset_info_dir = data_folder+'dataset_info.yml'
  with open(dataset_info_dir, 'r') as ff:
    dataset_info = yaml.safe_load(ff)
  if 'object_width' not in dataset_info:
    print('Computing object width')
    model_3d_path = dataset_info['models'][0]['model_path']
    if '.ply' not in model_3d_path:
      model_3d_path = model_3d_path+'/geometry.ply'
    model_3d = o3d.io.read_point_cloud(model_3d_path)
    model_3d = np.asarray(model_3d.points).copy()
    object_max_width = compute_cloud_diameter(model_3d) * 1000
    bounding_box = dataset_info['boundingbox']
    with_add = bounding_box / 100 * object_max_width
    object_width = object_max_width + with_add
    dataset_info['object_width'] = float(object_width)
    print('object_width=',object_width)
    with open(dataset_info_dir, 'w') as ff:
      yaml.dump(dataset_info, ff)

  cam_K = np.array([[dataset_info['camera']['focalX'], 0, dataset_info['camera']['centerX']],
                    [0, dataset_info['camera']['focalY'], dataset_info['camera']['centerY']],
                    [0, 0, 1]])
  cvcam_in_blendercam = np.array([[1,0,0,0],
                                  [0,-1,0,0],
                                  [0,0,-1,0],
                                  [0,0,0,1]])


  num_val = 2000
  out_train_path = data_folder+'train_data_blender_DR/'
  out_val_path = data_folder+'validation_data_blender_DR/'
  _ = input('remove {}\nand\n{}?'.format(out_train_path,out_val_path))
  os.system('rm -rf '+out_train_path+' '+out_val_path)
  os.makedirs(out_train_path)
  os.makedirs(out_val_path)

  producer = ProducerPurturb(dataset_info)
  rgb_files = sorted(glob.glob('/media/bowen/56c8da60-8656-47c3-b940-e928a3d4ec3b/blender_syn_sequence/mydataset_DR/*rgb.png'.format(class_id)))
  assert len(rgb_files)>0
  print('len(rgb_files): ',len(rgb_files))

  for i in range(0,len(rgb_files)):
    if i%100==0:
      print('complete pair data class={}:  {}/{}'.format(class_id,i,len(rgb_files)))
    rgb_file = rgb_files[i]
    # print(rgb_file)
    meta = np.load(rgb_file.replace('rgb.png','poses_in_world.npz'))
    class_ids = meta['class_ids']
    poses_in_world = meta['poses_in_world']
    blendercam_in_world = meta['blendercam_in_world']
    pos = np.where(class_ids==class_id)
    B_in_cam = np.linalg.inv(cvcam_in_blendercam).dot(np.linalg.inv(blendercam_in_world).dot(poses_in_world[pos,:,:].reshape(4,4)))

    current_depth = cv2.imread(rgb_file.replace('rgb','depth'),cv2.IMREAD_UNCHANGED)
    current_seg = cv2.imread(rgb_file.replace('rgb','seg'), cv2.IMREAD_UNCHANGED).astype(np.uint8)

    if len(current_seg.shape)==3:
      current_seg = current_seg[:,:,0]
    if np.sum(current_seg==class_id)<100:
      continue

    current_rgb = np.array(Image.open(rgb_file))[:,:,:3]
    producer.generate(out_train_path,B_in_cam,current_rgb,current_depth,num_sample=1,class_id=class_id,current_seg=current_seg,debug=False)

  #Prepare val data
  rgbA_files = sorted(glob.glob(out_train_path+'*rgbA.png'))
  rgbA_files.reverse()

  for i in range(num_val):
    if i%1000==0:
      print('moving to val: {}/{}'.format(i,num_val))
    shutil.move(rgbA_files[i],out_val_path+'%07drgbA.png'%(i))
    shutil.move(rgbA_files[i].replace('A','B'),out_val_path+'%07drgbB.png'%(i))
    shutil.move(rgbA_files[i].replace('rgbA','depthA'),out_val_path+'%07ddepthA.png'%(i))
    shutil.move(rgbA_files[i].replace('rgbA','depthB'),out_val_path+'%07ddepthB.png'%(i))
    shutil.move(rgbA_files[i].replace('rgbA.png','meta.npz'),out_val_path+'%07dmeta.npz'%(i))
    shutil.move(rgbA_files[i].replace('rgbA','segB'),out_val_path+'%07dsegB.png'%(i))



if __name__ == '__main__':
  completeBlenderYcbDR()




