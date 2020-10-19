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
code_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_path)
import numpy as np
from PIL import Image
import cv2
import time
import trimesh
import pyrender


class Renderer:
	def __init__(self,model_paths, cam_K, H,W):
		if not isinstance(model_paths,list):
			print("model_paths have to be list")
			raise RuntimeError
		self.scene = pyrender.Scene(ambient_light=[1., 1., 1.],bg_color=[0,0,0])
		self.camera = pyrender.IntrinsicsCamera(fx=cam_K[0,0],fy=cam_K[1,1],cx=cam_K[0,2],cy=cam_K[1,2],znear=0.1,zfar=2.0)
		self.cam_node = self.scene.add(self.camera, pose=np.eye(4))
		self.mesh_nodes = []

		for model_path in model_paths:
			print('model_path',model_path)
			obj_mesh = trimesh.load(model_path)
			colorVisual = obj_mesh.visual.to_color()
			mesh = pyrender.Mesh.from_trimesh(obj_mesh)
			mesh_node = self.scene.add(mesh,pose=np.eye(4),parent_node=self.cam_node) # Object pose parent is cam
			self.mesh_nodes.append(mesh_node)

		self.H = H
		self.W = W

		self.r = pyrender.OffscreenRenderer(self.W, self.H)
		self.glcam_in_cvcam = np.array([[1,0,0,0],
										[0,-1,0,0],
										[0,0,-1,0],
										[0,0,0,1]])
		self.cvcam_in_glcam = np.linalg.inv(self.glcam_in_cvcam)


	def render(self,ob_in_cvcams):
		assert isinstance(ob_in_cvcams, list)
		for i,ob_in_cvcam in enumerate(ob_in_cvcams):
			ob_in_glcam = self.cvcam_in_glcam.dot(ob_in_cvcam)
			self.scene.set_pose(self.mesh_nodes[i],ob_in_glcam)
		color, depth = self.r.render(self.scene)  # depth: float
		return color, depth
