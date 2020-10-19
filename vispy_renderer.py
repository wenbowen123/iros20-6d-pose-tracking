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


import os,sys,yaml
import vispy
import OpenGL.GL as gl
from vispy import app, gloo
import numpy as np
from PIL import Image
import cv2,argparse,pickle,time
from plyfile import PlyData, PlyElement
from Utils import *

class VispyRenderer(app.Canvas):
	def __init__(self, model_path, K, H, W, backend="pyglet"):
		app.use_app(backend)
		app.Canvas.__init__(self, show=False, size=(W,H))
		self.size = (W,H)  #(W,H)
		self.K = K.copy()

		fragment_code = """
			#version 130

			varying vec3 fragpos;
			varying vec3 normal;

			varying vec3 fragmentColor;
			out vec4 color;

			varying vec3 viewPos;
			uniform vec3 light_direction;

			void main()
			{
						vec3 norm = normal;
						vec3 lightDirA = normalize(-light_direction - fragpos);
						vec3 diffuseA = vec3(0.4, 0.4, 0.4) * max(dot(norm, lightDirA), 0.0);
						vec4 colors = vec4(fragmentColor, 1);
						vec3 light_3 = diffuseA + vec3(0.65, 0.65, 0.65);
						vec4 light = vec4(light_3, 1.0f);
						color = clamp(light * colors, 0.0, 1.0);
			}
			"""

		vertex_code = """
			#version 130

			attribute vec3 a_position;
			attribute vec3 a_color;
			attribute vec3 a_normal;

			varying vec3 fragmentColor;
			varying vec3 normal;
			varying vec3 fragpos;
			varying vec3 viewPos;

			uniform mat4 view;
			uniform mat4 proj;

			void main()
			{
				fragmentColor = a_color;
				gl_Position = proj * view * vec4(a_position, 1.0);
				fragpos = a_position;
				viewPos = vec3(view);
				normal = a_normal;
			}
			"""


		print('model_path: ',model_path)
		if '.ply' in model_path:
			ply = PlyData.read(model_path)
			vertices = np.stack((ply['vertex']['x'].reshape(-1), ply['vertex']['y'].reshape(-1), ply['vertex']['z'].reshape(-1)), axis=-1)
			assert vertices.shape[1]==3
			face_indices = ply['face']['vertex_indices']
			face_indices = np.stack(face_indices, axis=0)
			assert face_indices.shape[1]==3
			vertex_color = np.stack((ply['vertex']['red'].reshape(-1), ply['vertex']['green'].reshape(-1), ply['vertex']['blue'].reshape(-1)), axis=-1)
			vertex_normal = np.stack((ply['vertex']['nx'].reshape(-1), ply['vertex']['ny'].reshape(-1), ply['vertex']['nz'].reshape(-1)), axis=-1)
			vertex_normal = vertex_normal/np.linalg.norm(vertex_normal, axis=1).reshape(-1,1)
		else:
			print('vispy model_path has to be a ply file')
			raise RuntimeError

		self.data = np.ones(vertices.shape[0], [('a_position', np.float32, 3),('a_color', np.float32, 3),('a_normal', np.float32, 3)])
		self.data['a_position'] = vertices.copy()
		self.data['a_color'] = vertex_color/255.0
		self.data['a_normal'] = vertex_normal.copy()

		self.vertex_buffer = gloo.VertexBuffer(self.data)
		self.index_buffer = gloo.IndexBuffer(face_indices.reshape(-1).astype(np.uint32))

		self.program = gloo.Program(vertex_code, fragment_code)
		self.program.bind(self.vertex_buffer)
		self.update_cam_mat(self.K, 0, self.size[0], self.size[1], 0)

		self.fbo = gloo.FrameBuffer(gloo.Texture2D(shape=(self.size[1],self.size[0],3)), gloo.RenderBuffer(self.size[::-1]))
		self.rgb = None
		self.depth = None

	def update_cam_mat(self, K, left, right, bottom, top):
		"""
		@left, right, bottom, top: bounding box corners
		"""
		self.near_plane = 0.1
		self.far_plane = 2.0
		proj = np.array([[K[0,0], 0, -K[0,2], 0],
						 [0, K[1,1], -K[1,2], 0],
						 [0, 0, self.near_plane + self.far_plane, self.near_plane * self.far_plane],
						 [0, 0, -1, 0]])
		orthographic_mat = np.array([[2. / (right - left), 0, 0, -(right + left) / (right - left)],
												[0, 2. / (top - bottom), 0, -(top + bottom) / (top - bottom)],
												[0, 0, -2 / (self.far_plane - self.near_plane), -(self.far_plane + self.near_plane) / (self.far_plane - self.near_plane)],
												[0, 0, 0, 1]]).astype(np.float32)
		self.projection_matrix = orthographic_mat.dot(proj).T
		self.program['proj'] = self.projection_matrix

	def on_draw(self, event):
		with self.fbo:
			gloo.set_state(depth_test=True)
			gloo.set_cull_face('back')
			gloo.clear(color=True, depth=True)
			gloo.set_viewport(0, 0, *self.size)
			self.program.draw('triangles', self.index_buffer)

			self.rgb = gl.glReadPixels(0, 0, self.size[0], self.size[1], gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
			self.depth = gl.glReadPixels(0, 0, self.size[0], self.size[1], gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
			self.rgb = np.frombuffer(self.rgb, dtype=np.uint8).reshape((self.size[1], self.size[0], 3))
			self.depth = self.depth.reshape(self.size[::-1])
			A = self.projection_matrix[2, 2]
			B = self.projection_matrix[3, 2]
			distance = B / (self.depth * -2.0 + 1.0 - A) * -1
			idx = distance[:, :] >= B / (A + 1)
			distance[idx] = 0
			self.depth = (distance * 1000).astype(np.uint16)

	def render_image(self, ob2cam_gl):
		light_direction = np.dot(np.linalg.inv(ob2cam_gl.T), np.array([0, 0.1, -0.9, 1]))[:3]
		self.program["light_direction"] = light_direction.astype(np.float32)
		self.program['view'] = ob2cam_gl.T

		self.update()
		self.on_draw(None)
		return self.rgb, self.depth


