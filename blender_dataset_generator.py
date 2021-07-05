# -*- coding: future_fstrings -*-

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

import bpy
import os, sys, time,copy,string
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
import cv2
from PIL import Image, ImageDraw
import yaml
import numpy as np
import bmesh
from mathutils.bvhtree import BVHTree
import glob,subprocess
import argparse
import transformations as T
from mathutils import Vector, Matrix, Quaternion
import multiprocessing
import re



def readExr(exr_dir):
  return cv2.imread(exr_dir, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)


def matrixToNumpyArray(mat):
    new_mat = np.array([[mat[0][0],mat[0][1],mat[0][2],mat[0][3]],
                        [mat[1][0],mat[1][1],mat[1][2],mat[1][3]],
                        [mat[2][0],mat[2][1],mat[2][2],mat[2][3]],
                        [mat[3][0],mat[3][1],mat[3][2],mat[3][3]]])
    return new_mat

def numpyArrayToMatrix(array):
    mat = Matrix(((array[0,0],array[0,1], array[0,2], array[0,3]),
                (array[1,0],array[1,1], array[1,2], array[1,3]),
                (array[2,0],array[2,1], array[2,2], array[2,3]) ,
                (array[3,0],array[3,1], array[3,2], array[3,3])))
    return mat

def changeEnvironmentLight(dataset_info):
  env_light_range = dataset_info['blender']['env_light_range']
  bpy.context.scene.world.light_settings.use_environment_light = True
  bpy.context.scene.world.light_settings.environment_energy = np.random.uniform(env_light_range[0],env_light_range[1])
  env_color_r = np.random.uniform(dataset_info['blender']['env_light_color'][0][0], dataset_info['blender']['env_light_color'][0][1])
  env_color_g = np.random.uniform(dataset_info['blender']['env_light_color'][1][0], dataset_info['blender']['env_light_color'][1][1])
  env_color_b = np.random.uniform(dataset_info['blender']['env_light_color'][2][0], dataset_info['blender']['env_light_color'][2][1])
  bpy.context.scene.world.ambient_color = (env_color_r,env_color_g,env_color_b)



def reset(dataset_info):
  changeEnvironmentLight(dataset_info)
  for ob in bpy.data.objects:
    ob.select = False
  for ob in bpy.data.objects:
    if ob.type == "LAMP":
      ob.select = True
      bpy.ops.object.delete()
    else:
      ob.select = False
  for ob in bpy.data.objects:
    if 'ob' in ob.name.lower():
      ob.location[0] = 9999


def setupCamera(H,W,K):
  bpy.context.scene.render.resolution_x = W
  bpy.context.scene.render.resolution_y = H
  cam_data = bpy.data.objects['Camera'].data
  sensor_width_in_mm = cam_data.sensor_width
  cam_data.shift_x = -(K[0,2] - 0.5 * W) / W
  cam_data.shift_y = (K[1,2] - 0.5 * H) / W
  cam_data.lens = K[0,0] / W * sensor_width_in_mm
  pixel_aspect = K[1,1] / K[0,0]
  bpy.context.scene.render.pixel_aspect_x = 1.0
  bpy.context.scene.render.pixel_aspect_y = pixel_aspect
  bpy.context.scene.camera = bpy.data.objects['Camera']
  bpy.context.scene.update()


def placeObject(ob_name,pose):
  ob = bpy.data.objects[ob_name]
  pose_mat = numpyArrayToMatrix(pose)
  ob.matrix_world = pose_mat
  bpy.context.scene.update()

def addLightAndPlace(dataset_info,num):
  for _ in range(num):
    bpy.ops.object.lamp_add(type='POINT', view_align = False)

  for ob in bpy.data.objects:
    if 'Point' in ob.name:
      lamp_brightness = dataset_info['blender']['lamp_brightness']
      pos_ranges = dataset_info['blender']['lamp_pos_range']
      lx = np.random.uniform(pos_ranges[0][0],pos_ranges[0][1])
      ly = np.random.uniform(pos_ranges[1][0],pos_ranges[1][1])
      lz = np.random.uniform(pos_ranges[2][0],pos_ranges[2][1])
      strength = np.random.uniform(lamp_brightness[0], lamp_brightness[1])
      ob.location = [lx, ly, lz]
      light_color_ranges = dataset_info['blender']['lamp_colors']
      r = np.random.uniform(light_color_ranges[0][0],light_color_ranges[0][1])
      g = np.random.uniform(light_color_ranges[1][0],light_color_ranges[1][1])
      b = np.random.uniform(light_color_ranges[2][0],light_color_ranges[2][1])

      ob.data.use_specular = False
      ob.data.shadow_method = 'RAY_SHADOW'
      ob.data.energy = strength
      ob.data.color = (r, g, b)
      ob.data.shadow_ray_samples = 6
      ob.data.shadow_ray_sample_method = 'ADAPTIVE_QMC'



def loadObjectModel(file_dir,index,name):
  folder = file_dir
  if '.' not in file_dir:
    file_dir = glob.glob(file_dir+'/*.obj')[0]
  else:
    folder = os.path.dirname(file_dir)
  print('Loading object ',file_dir)

  if '.obj' in file_dir:
    bpy.ops.import_scene.obj(filepath=file_dir)
    ob = bpy.context.selected_objects[0]
    if len(ob.data.materials)==0:
      mat_name = "Material"
      mat = bpy.data.materials.new(name=mat_name)
      ob.data.materials.append(mat)
    mat = ob.data.materials[0]
    slot = mat.texture_slots.add()
  elif '.dae' in file_dir:
    bpy.ops.wm.collada_import(filepath=file_dir)
  imported = bpy.context.selected_objects[0]
  imported.pass_index = index
  imported.location[0] = 9999
  imported.name = name



def changeObjectTexture(ob_name,image_dir):
  ob=bpy.data.objects[ob_name]
  if len(ob.data.materials)==0:
      mat_name = "Material"
      mat = bpy.data.materials.new(name=mat_name)
      ob.data.materials.append(mat)
  mat = ob.data.materials[0]
  mat.use_nodes = False
  img = bpy.data.images.load(image_dir)  # img_name is the path to image
  tex_name = "Texture"
  tex = bpy.data.textures.new(tex_name, 'IMAGE')
  tex.image = img
  slot = mat.texture_slots[0]
  slot.texture = tex
  bpy.context.scene.update()
  ob.active_material.texture_slots[0].texture_coords = 'OBJECT'
  ob.active_material.texture_slots[0].scale[0] = 4
  ob.active_material.texture_slots[0].scale[1] = 4



def random_string(size):
  chars = list(string.ascii_uppercase + string.digits)
  return ''.join(np.random.choice(chars) for _ in range(size))


def render(K,id):
  '''
    return rgb, depth, id mask. object index was assigned in __init__
    id_mask: see config files
  '''
  out_dir = '/tmp/{}/'.format(random_string(size=20))
  os.system('rm -rf {} && mkdir -p {}'.format(out_dir,out_dir))
  for ob in bpy.data.objects:
    if 'ob' in ob.name:
      ob.active_material.use_nodes = False

  tree = bpy.context.scene.node_tree
  tree.render_quality = "HIGH"
  tree.edit_quality = "HIGH"
  tree.use_opencl = True

  links = tree.links

  for n in tree.nodes:
    tree.nodes.remove(n)

  #================ collect images and label ===================
  render_node = tree.nodes.new('CompositorNodeRLayers')
  rgb_node = tree.nodes.new('CompositorNodeOutputFile')   # rgb
  rgb_node.format.file_format = 'PNG'
  rgb_node.base_path = out_dir
  rgb_node.file_slots[0].path = "%07drgbB"%(id)
  links.new(render_node.outputs['Image'], rgb_node.inputs[0])

  depth_node = tree.nodes.new('CompositorNodeOutputFile')   # depth
  depth_node.format.file_format = 'OPEN_EXR'
  depth_node.base_path = out_dir
  depth_node.file_slots[0].path = "%07ddepthB"%(id)
  links.new(render_node.outputs['Depth'], depth_node.inputs[0])

  seg_node = tree.nodes.new('CompositorNodeOutputFile')   # seg
  seg_node.format.file_format = 'OPEN_EXR'
  seg_node.base_path = out_dir
  seg_node.file_slots[0].path = "%07dsegB"%(id)
  links.new(render_node.outputs['IndexOB'], seg_node.inputs[0])

  bpy.ops.render.render(write_still=False)

  index = int(re.findall(r'depthB\d{4}',glob.glob(out_dir+'*depthB*.exr')[0])[0].replace('depthB',''))
  rgbB = np.array(Image.open(out_dir+'%07drgbB%04d.png'%(id,index)))[:,:,:3]
  depth_meter = readExr(out_dir+'%07ddepthB%04d.exr'%(id,index))[:,:,0]
  depth_meter[depth_meter<0.1] = 0
  depth_meter[depth_meter>2.0] = 0
  depthB = (depth_meter*1000).astype(np.uint16)
  segB = readExr(out_dir+'%07dsegB%04d.exr'%(id,index)).astype(np.uint8)

  os.system('rm -rf {}'.format(out_dir))

  return rgbB, depthB, segB


def get_dynamic_objects():
  obs = []
  for ob in bpy.data.objects:
    if 'Camera' not in ob.name and 'Point' not in ob.name and 'box_plane' not in ob.name:
      obs.append(ob)
  return obs


def generate():
  code_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_info_dir = f"{code_dir}/dataset_info.yml"
  with open(dataset_info_dir,'r') as ff:
    dataset_info = yaml.safe_load(ff)

  num_images = int((dataset_info['train_samples']+dataset_info['val_samples'])/0.7)
  xmin = dataset_info['blender']['range_x'][0]
  xmax = dataset_info['blender']['range_x'][1]
  ymin = dataset_info['blender']['range_y'][0]
  ymax = dataset_info['blender']['range_y'][1]
  zmin = dataset_info['blender']['range_z'][0]
  zmax = dataset_info['blender']['range_z'][1]

  code_dir = os.path.dirname(os.path.realpath(__file__))
  out_dir = f'{code_dir}/generated_data/'

  print('Using: {}'.format(dataset_info_dir))
  os.system(f'rm -rf {out_dir} && mkdir -p {out_dir}')

  H = dataset_info['camera']['height']
  W = dataset_info['camera']['width']
  K = np.eye(3)
  K[0,0] = dataset_info['camera']['focalX']
  K[1,1] = dataset_info['camera']['focalY']
  K[0,2] = dataset_info['camera']['centerX']
  K[1,2] = dataset_info['camera']['centerY']
  print('K:\n',K)
  K[1,1] = K[0,0]
  setupCamera(W=dataset_info['camera']['width'],H=dataset_info['camera']['height'],K=K)

  texture_folders = dataset_info['texture_folders']
  texture_files = []
  print('Collecting texture files...')
  for folder in texture_folders:
    texture_files += glob.glob(folder,recursive=True)

  texture_files.sort()
  assert len(texture_files)>0
  print('#texture_files:',len(texture_files))

  for k in dataset_info['models'].keys():
    obj_file = dataset_info['models'][k]['model_path'].replace('.ply','.obj')
    loadObjectModel(obj_file,index=k,name=str(k))

  id2ob = {}
  obs = get_dynamic_objects()
  for ob in obs:
    print(ob.name)
    bpy.context.scene.objects.active = ob
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    bpy.ops.object.modifier_add(type = 'COLLISION')
    ob.rigid_body.mass = 10.0
    ob.rigid_body.use_margin = True
    ob.rigid_body.collision_margin = 1e-4
    ob.rigid_body.linear_damping = 0.01
    ob.rigid_body.angular_damping = 0.01
    ob.rigid_body.friction = 0.01
    ob.collision.absorption = 0.01
    ob.collision.friction_factor = 0.01
    ob.rigid_body.restitution = 0.99
    ob.data.materials[0].ambient = 0.2
    ob.layers[0] = True
    class_id = int(ob.pass_index)
    if class_id<255:
      id2ob[class_id] = ob
  class_ids = np.array(list(id2ob.keys()))
  print('class_ids',class_ids)


  count = 0
  while count<num_images:
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> {}/{}'.format(count,num_images))
    reset(dataset_info)
    light_num = np.random.randint(0,dataset_info['blender']['max_lamp_num']+1)
    print('light_num=',light_num)
    addLightAndPlace(dataset_info,light_num)

    texture_file = np.random.choice(texture_files)
    print('Using texture file\n',texture_file)
    for ob in bpy.data.objects:
      if 'box_plane' in ob.name:
        changeObjectTexture(ob.name,texture_file)

    obs = get_dynamic_objects()
    for ob in obs:
      pose = np.eye(4)
      pose[0,3] = np.random.uniform(xmin,xmax)
      pose[1,3] = np.random.uniform(ymin,ymax)
      pose[2,3] = np.random.uniform(zmin,zmax)
      pose[:3,:3] = T.random_rotation_matrix()[:3,:3]
      placeObject(ob.name,pose)

    print('start gravity simulation')
    bpy.context.scene.gravity = np.random.uniform(-2,2,size=3)

    for ii in range(1,4):
      bpy.context.scene.frame_set(ii)
    bpy.context.scene.update()

    blendercam_in_world = matrixToNumpyArray(bpy.data.objects['Camera'].matrix_world)

    rgbB, depthB, segB = render(K,count)
    if (segB>0).sum()<100:   #Target object not in the image
      print('segB too small')
      continue

    print("Saving to ",out_dir+'/%07drgb.png'%(count))
    Image.fromarray(rgbB).save(out_dir+'/%07drgb.png'%(count), optimize=True)
    cv2.imwrite(out_dir+'/%07ddepth.png'%(count),depthB.astype(np.uint16))
    cv2.imwrite(out_dir+'/%07dseg.png'%(count),segB.astype(np.uint8))

    bpy.context.scene.update()
    poses_in_world = []
    for class_id in class_ids:
      ob = id2ob[class_id]
      ob_in_world = matrixToNumpyArray(ob.matrix_world)
      poses_in_world.append(ob_in_world)
    poses_in_world = np.array(poses_in_world)
    np.savez(out_dir+'/%07dposes_in_world.npz'%(count), class_ids=class_ids, poses_in_world=poses_in_world, blendercam_in_world=blendercam_in_world,K=K)

    count += 1


  print('Finished {}'.format(out_dir))





if __name__=='__main__':
  generate()




