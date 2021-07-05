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

import os, sys, time
import open3d as o3d
import cv2
from PIL import Image
import numpy as np
sys.path.append('/../')
import math,glob,re,copy
from scipy.spatial import ConvexHull, distance_matrix
import scipy.spatial as spatial
import transformations
import random
import transformations as T
from scipy import ndimage


COLOR_MAP=np.array([[0, 0, 0], #Ignore
          [128,0,0], #Background
          [0,128,0], #Wall
          [128,128,0], #Floor
          [0,0,128], #Ceiling
          [128,0,128], #Table
          [0,128,128], #Chair
          [128,128,128], #Window
          [64,0,0], #Door
          [192,0,0], #Monitor
          [64, 128, 0],     # 11th
          [192, 0, 128],
          [64, 128, 128],
          [192, 128, 128],
          [0, 64, 0],
          [128, 64, 0],
          [0, 192, 0],
          [128, 192, 0], # defined for 18 classes currently
          ])


def add(pred,gt,model):
  """
  Average Distance of Model Points for objects with no indistinguishable views
  - by Hinterstoisser et al. (ACCV 2012).
  """
  pred_model = copy.deepcopy(model)
  gt_model = copy.deepcopy(model)
  pred_model.transform(pred)
  gt_model.transform(gt)
  e = np.linalg.norm(np.asarray(pred_model.points) - np.asarray(gt_model.points), axis=1).mean()
  return e

def adi(pred,gt,model):
  """
  @pred: 4x4 mat
  @gt:
  @model: open3d pcd model
  """
  pred_model = copy.deepcopy(model)
  gt_model = copy.deepcopy(model)
  pred_model.transform(pred)
  gt_model.transform(gt)

  nn_index = spatial.cKDTree(np.asarray(pred_model.points).copy())
  nn_dists, _ = nn_index.query(np.asarray(gt_model.points).copy(), k=1, n_jobs=10)
  e = nn_dists.mean()
  return e


def compute_cloud_diameter(points):
  hull = ConvexHull(points)
  hull_points = points[hull.vertices]
  distances = distance_matrix(hull_points, hull_points)
  return np.max(distances)


def findClassContainedVideosYcb(class_id, testset=True):
  ycb_dir = '/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/DATASET/Tracking/YCB_Video_Dataset/data_organized/'
  gt_dirs = glob.glob(ycb_dir+'**/pose_gt')
  gt_dirs.sort()
  video_indices = []
  for i in range(len(gt_dirs)):
    gt_dir = gt_dirs[i]+'/'
    video_index = re.findall(r'/[0-9]{4}/',gt_dir)[0][1:-1]
    video_index = int(video_index)
    if testset:
      if video_index<48 or video_index>59:
        continue
    indices = list(map(int,os.listdir(gt_dir)))
    if class_id in indices:
      video_indices.append(video_index)
  return video_indices

def makeCanvas(imgs, flipBR=True):
  '''
    @imgs: list of images, assume same sizes
    @flipBR: flip B and R channel for opencv
  '''
  num_imgs = len(imgs)
  H = imgs[0].shape[0]
  W = imgs[0].shape[1]
  gap = 10
  canvas = np.zeros((H, W*num_imgs+gap*(num_imgs-1),3))
  start_w = 0
  for i in range(len(imgs)):
    cur_img = imgs[i].copy()
    if flipBR:
      cur_img[:,:,0] = imgs[i][:,:,2]
      cur_img[:,:,2] = imgs[i][:,:,0]
    canvas[:,start_w:(start_w+W)] = cur_img
    start_w += W+gap
  canvas = canvas.astype(np.uint8)
  return canvas


def rgbd2PointCloud(K, depth, rgb=np.array([])):
  mask=np.logical_and(depth>0.1,depth<2)
  vs, us = np.where(mask)
  zs = depth[mask]
  xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
  ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
  pts = np.array([xs, ys, zs]).T
  if rgb != np.array([]):
    colors = rgb[vs, us, :]
  else:
    colors = None
  return pts, colors, mask


def toOpen3dCloud(points,colors):
  import open3d as o3d
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
  cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64)/255.0)
  return cloud



def hinter_sampling(min_n_pts, radius=1):
  a, b, c = 0.0, 1.0, (1.0 + math.sqrt(5.0)) / 2.0
  pts = [(-b, c, a), (b, c, a), (-b, -c, a), (b, -c, a), (a, -b, c), (a, b, c),
       (a, -b, -c), (a, b, -c), (c, a, -b), (c, a, b), (-c, a, -b), (-c, a, b)]
  faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
       (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
       (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
       (8, 6, 7), (9, 8, 1)]

  # Refinement level on which the points were created
  pts_level = [0 for _ in range(len(pts))]

  ref_level = 0
  while len(pts) < min_n_pts:
    ref_level += 1
    edge_pt_map = {}
    faces_new = []

    for face in faces:
      pt_inds = list(face)
      for i in range(3):
        edge = (face[i], face[(i + 1) % 3])
        edge = (min(edge), max(edge))
        if edge not in edge_pt_map.keys():
          pt_new_id = len(pts)
          edge_pt_map[edge] = pt_new_id
          pt_inds.append(pt_new_id)

          pt_new = 0.5 * (np.array(pts[edge[0]]) + np.array(pts[edge[1]]))
          pts.append(pt_new.tolist())
          pts_level.append(ref_level)
        else:
          pt_inds.append(edge_pt_map[edge])

      faces_new += [(pt_inds[0], pt_inds[3], pt_inds[5]),
              (pt_inds[3], pt_inds[1], pt_inds[4]),
              (pt_inds[3], pt_inds[4], pt_inds[5]),
              (pt_inds[5], pt_inds[4], pt_inds[2])]
    faces = faces_new

  pts = np.array(pts)
  pts *= np.reshape(radius / np.linalg.norm(pts, axis=1), (pts.shape[0], 1))

  pt_conns = {}
  for face in faces:
    for i in range(len(face)):
      pt_conns.setdefault(face[i], set()).add(face[(i + 1) % len(face)])
      pt_conns[face[i]].add(face[(i + 2) % len(face)])

  top_pt_id = np.argmax(pts[:, 2])
  pts_ordered = []
  pts_todo = [top_pt_id]
  pts_done = [False for _ in range(pts.shape[0])]

  def calc_azimuth(x, y):
    two_pi = 2.0 * math.pi
    return (math.atan2(y, x) + two_pi) % two_pi

  while len(pts_ordered) != pts.shape[0]:
    pts_todo = sorted(pts_todo, key=lambda i: calc_azimuth(pts[i][0], pts[i][1]))
    pts_todo_new = []
    for pt_id in pts_todo:
      pts_ordered.append(pt_id)
      pts_done[pt_id] = True
      pts_todo_new += [i for i in pt_conns[pt_id]]

    pts_todo = [i for i in set(pts_todo_new) if not pts_done[i]]

  pts = pts[np.array(pts_ordered), :]
  pts_level = [pts_level[i] for i in pts_ordered]
  pts_order = np.zeros((pts.shape[0],))
  pts_order[np.array(pts_ordered)] = np.arange(pts.shape[0])
  for face_id in range(len(faces)):
    faces[face_id] = [pts_order[i] for i in faces[face_id]]

  return pts, pts_level

def sample_views(min_n_views, radius=[1],
         azimuth_range=(0, 2 * math.pi),
         elev_range=(-0.5 * math.pi, 0.5 * math.pi)):
  pts=np.empty((0,3))
  if True:
    for r in radius:
      local_pts, pts_level = hinter_sampling(min_n_views, radius=r)
      print('radius={}, pts shape={}'.format(r,local_pts.shape))
      pts=np.vstack((pts,local_pts))

  else:
    pts = fibonacci_sampling(min_n_views + 1, radius=radius)
    pts_level = [0 for _ in range(len(pts))]


  views = []
  for pt in pts:
    azimuth = math.atan2(pt[1], pt[0])
    if azimuth < 0:
      azimuth += 2.0 * math.pi

    a = np.linalg.norm(pt)
    b = np.linalg.norm([pt[0], pt[1], 0])
    elev = math.acos(b / a)
    if pt[2] < 0:
      elev = -elev

    if not (azimuth_range[0] <= azimuth <= azimuth_range[1] and
        elev_range[0] <= elev <= elev_range[1]):
      continue

    f = -np.array(pt) # Forward direction
    f /= np.linalg.norm(f)
    u = np.array([0.0, 0.0, 1.0]) # Up direction
    s = np.cross(f, u) # Side direction
    if np.count_nonzero(s) == 0:
      # f and u are parallel, i.e. we are looking along or against Z axis
      s = np.array([1.0, 0.0, 0.0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f) # Recompute up
    R = np.array([[s[0], s[1], s[2]],
            [u[0], u[1], u[2]],
            [-f[0], -f[1], -f[2]]])

    R_yz_flip = transformations.rotation_matrix(math.pi, [1, 0, 0])[:3, :3]
    R = R_yz_flip.dot(R)

    t = -R.dot(np.array(pt).reshape((3, 1)))

    views.append({'R': R, 't': t})

  return views, pts_level


def compute_bbox(pose, K, scale_size=230, scale=(1, 1, 1)):
  obj_x = pose[0, 3] * scale[0]
  obj_y = pose[1, 3] * scale[1]
  obj_z = pose[2, 3] * scale[2]
  offset = scale_size / 2
  points = np.ndarray((4, 3), dtype=np.float)
  points[0] = [obj_x - offset, obj_y - offset, obj_z]     # top left
  points[1] = [obj_x - offset, obj_y + offset, obj_z]     # top right
  points[2] = [obj_x + offset, obj_y - offset, obj_z]     # bottom left
  points[3] = [obj_x + offset, obj_y + offset, obj_z]     # bottom right
  projected_vus = np.zeros((points.shape[0], 2))
  projected_vus[:, 1] = points[:, 0] * K[0,0] / points[:, 2] + K[0,2]
  projected_vus[:, 0] = points[:, 1] * K[1,1] / points[:, 2] + K[1,2]
  projected_vus = np.round(projected_vus).astype(np.int32)
  return projected_vus



def crop_bbox(color, depth, boundingbox, output_size=(100, 100), seg=None):
  left = np.min(boundingbox[:, 1])
  right = np.max(boundingbox[:, 1])
  top = np.min(boundingbox[:, 0])
  bottom = np.max(boundingbox[:, 0])

  h, w, c = color.shape
  crop_w = right - left
  crop_h = bottom - top
  color_crop = np.zeros((crop_h, crop_w, 3), dtype=color.dtype)
  depth_crop = np.zeros((crop_h, crop_w), dtype=np.float)
  seg_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
  top_offset = abs(min(top, 0))
  bottom_offset = min(crop_h - (bottom - h), crop_h)
  right_offset = min(crop_w - (right - w), crop_w)
  left_offset = abs(min(left, 0))

  top = max(top, 0)
  left = max(left, 0)
  bottom = min(bottom, h)
  right = min(right, w)
  color_crop[top_offset:bottom_offset, left_offset:right_offset, :] = color[top:bottom, left:right, :]
  depth_crop[top_offset:bottom_offset, left_offset:right_offset] = depth[top:bottom, left:right]
  resized_rgb = cv2.resize(color_crop, output_size, interpolation=cv2.INTER_NEAREST)
  resized_depth = cv2.resize(depth_crop, output_size, interpolation=cv2.INTER_NEAREST)

  if seg is not None:
    seg_crop[top_offset:bottom_offset, left_offset:right_offset] = seg[top:bottom, left:right]
    resized_seg = cv2.resize(seg_crop, output_size, interpolation=cv2.INTER_NEAREST)
    final_seg = resized_seg.copy()

  mask_rgb = resized_rgb != 0
  mask_depth = resized_depth != 0
  resized_depth = resized_depth.astype(np.uint16)
  final_rgb = resized_rgb * mask_rgb
  final_depth = resized_depth * mask_depth
  if seg is not None:
    return final_rgb, final_depth, final_seg
  else:
    return final_rgb, final_depth



def normalize_rotation_matrix(R):
  R[:,0] = R[:,0]/np.linalg.norm(R[:,0])
  R[:,1] = R[:,1]/np.linalg.norm(R[:,1])
  R[:,2] = R[:,2]/np.linalg.norm(R[:,2])
  return R




def random_gaussian_magnitude(max_T, max_R):
  direction_T = random_direction()
  while 1:
    magn_T = np.random.normal(0,max_T)
    if abs(magn_T)<=max_T:
      break
  T = direction_T*magn_T
  direction_R = random_direction()
  direction_R = direction_R/np.linalg.norm(direction_R)
  while 1:
    magn_R = np.random.normal(0,max_R)  #degree
    if abs(magn_R)<=max_R:
      break
  rod = direction_R*magn_R/180.0*np.pi
  R = cv2.Rodrigues(rod)[0].reshape(3,3).copy()
  pose = np.eye(4)
  pose[:3,:3] = R
  pose[:3,3] = T.copy()
  return pose



def random_direction():
  def sph2cart(phi, theta, r):
    points = np.zeros(3)
    points[0] = r * math.sin(phi) * math.cos(theta)
    points[1] = r * math.sin(phi) * math.sin(theta)
    points[2] = r * math.cos(phi)
    return points

  theta = random.uniform(0, 1) * math.pi * 2
  phi = math.acos((2 * (random.uniform(0, 1))) - 1)
  return sph2cart(phi, theta, 1)

def get_random_view_matrix(min_radius,max_radius):
  eye = random_direction()
  distance = random.uniform(0, 1) * (max_radius - min_radius) + min_radius
  eye *= distance

  def gl_look_at(eye, center, up):
    E = eye
    C = center
    U = up
    F = C - E
    F /= np.linalg.norm(F)
    S = np.cross(F, U)
    S /= np.linalg.norm(S)
    U = np.cross(S, F)
    mat = np.eye(4, dtype=np.float32)
    mat[0, :] = np.hstack([S, 0])
    mat[1, :] = np.hstack([U, 0])
    mat[2, :] = np.hstack([-F, 0])
    mat[0, 3] = -np.dot(S, E)
    mat[1, 3] = -np.dot(U, E)
    mat[2, 3] = np.dot(F, E)
    return mat

  view = gl_look_at(eye, np.zeros(3), np.array([0, 0, 1]))

  angle = random.uniform(0, 1) * math.pi * 2
  cosa = math.cos(angle)
  sina = math.sin(angle)
  rotate_around_z = np.eye(4)
  rotate_around_z[0, 0] = cosa
  rotate_around_z[1, 0] = sina
  rotate_around_z[0, 1] = -sina
  rotate_around_z[1, 1] = cosa
  cam_in_object = np.linalg.inv(view).dot(rotate_around_z)
  view = np.linalg.inv(cam_in_object)
  return view


def cam_K_from_dict(cam_cfg):
  return np.array([[cam_cfg["focalX"],0,cam_cfg['centerX']],
                  [0,cam_cfg['focalY'],cam_cfg['centerY']],
                  [0,0,1]])


def compute_obj_max_width(model_cloud):
  return compute_cloud_diameter(model_cloud) * 1000



def fill_depth(depth,max_depth=2.0,extrapolate=False,blur_type='bilateral'):
  '''
  @depth: meters
  '''
  depth = depth.astype(np.float32)
  custom_kernel = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)
  valid_pixels = (depth > 0.1)
  depth[valid_pixels] = max_depth - depth[valid_pixels]
  depth = cv2.dilate(depth, custom_kernel)

  # Hole closing
  FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
  FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
  FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
  FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
  FULL_KERNEL_31 = np.ones((31, 31), np.uint8)
  depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, FULL_KERNEL_5)

  # Fill empty spaces with dilated values
  empty_pixels = (depth < 0.1)
  dilated = cv2.dilate(depth, FULL_KERNEL_7)
  depth[empty_pixels] = dilated[empty_pixels]

  # Extend highest pixel to top of image
  if extrapolate:
    top_row_pixels = np.argmax(depth > 0.1, axis=0)
    top_pixel_values = depth[top_row_pixels, range(depth.shape[1])]

    for pixel_col_idx in range(depth.shape[1]):
      depth[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = top_pixel_values[pixel_col_idx]

    # Large Fill
    empty_pixels = depth < 0.1
    dilated = cv2.dilate(depth, FULL_KERNEL_31)
    depth[empty_pixels] = dilated[empty_pixels]

  # Median blur
  depth = cv2.medianBlur(depth, 5)

  # Bilateral or Gaussian blur
  if blur_type == 'bilateral':
    # Bilateral blur
    depth = cv2.bilateralFilter(depth, 5, 1.5, 2.0)
  elif blur_type == 'gaussian':
    # Gaussian blur
    valid_pixels = (depth > 0.1)
    blurred = cv2.GaussianBlur(depth, (5, 5), 0)
    depth[valid_pixels] = blurred[valid_pixels]

  # Invert
  valid_pixels = (depth > 0.1)
  depth[valid_pixels] = max_depth - depth[valid_pixels]
  return depth


class Compose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, img):
		for t in self.transforms:
			img = t(img)
		return img


