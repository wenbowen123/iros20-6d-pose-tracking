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
code_dir = os.path.dirname(os.path.realpath(__file__))


if __name__=='__main__':
  if os.environ.get('BLENDER_PATH') == None:
    print("Please set BLENDER_PATH in bashrc!")
    sys.exit()
  g_blender_executable_path = os.environ['BLENDER_PATH']

  code_dir = os.path.dirname(os.path.realpath(__file__))
  blender_start_file = f'{code_dir}/1.blend'

  try:
    render_cmd = '{} {} -b -P {} '.format(g_blender_executable_path, blender_start_file, code_dir+'/blender_dataset_generator.py')
    print(render_cmd)
    os.system(render_cmd)
  except:
    print('render failed. render_cmd: %s' % (render_cmd))