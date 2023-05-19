import ros,rospy
import os,sys,cv2,glob,copy,yaml,time,argparse
sys.path.append('/root/catkin_ws/src')
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
import numpy as np
import yaml
import geometry.tf.src.tf as tf
from transformations import *
from rospy import Time
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Float64MultiArray, Float32MultiArray
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from Utils import *
from data_augmentation import *
from predict import Tracker

class TrackerRos:
  def __init__(self,tracker,pose_init):
    self.tracker = tracker
    self.color = None
    self.depth = None
    self.cur_time = None

    self.sub_depth = rospy.Subscriber(args.depth_topic, Image, self.grab_depth)
    self.sub_color = rospy.Subscriber(args.rgb_topic, Image, self.grab_color)
    self.listener = tf.listener.TransformListener()
    self.tf_pub = tf.broadcaster.TransformBroadcaster()
    self.A_in_cam = pose_init.copy()

  def reset(self,pose_init):
    self.color = None
    self.depth = None
    self.cur_time = None
    self.A_in_cam = pose_init.copy()

  def grab_depth(self,msg):
    depth = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.uint16)
    depth = fill_depth(depth/1e3,max_depth=2.0,extrapolate=False)
    self.depth = (depth*1000).astype(np.uint16)

  def grab_color(self,msg):
    self.cur_time = msg.header.stamp
    color = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
    self.color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

  def on_track(self):
    if self.color is None:
      print('color is None')
      return
    if self.depth is None:
      print('depth is None')
      return
    if self.cur_time is None:
      print('self.cur_time is None')
      return

    ob_in_cam = self.tracker.on_track(self.A_in_cam,self.color.astype(np.uint8), self.depth, gt_A_in_cam=np.eye(4),gt_B_in_cam=np.eye(4), debug=False,samples=1)
    self.A_in_cam = ob_in_cam.copy()

    trans = ob_in_cam[:3,3]
    q_wxyz = quaternion_from_matrix(ob_in_cam)
    q_xyzw = [q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]

    self.tf_pub.sendTransform(trans, q_xyzw, self.cur_time, args.object_frame_name,args.camera_frame_name)



if __name__=="__main__":
  rospy.init_node('my_node', anonymous=True)
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--artifact_id', type=int, default=772)
  parser.add_argument('--pose_init_file', type=str, default=f"{code_dir}/pose_init.txt")
  parser.add_argument('--rgb_topic', type=str, default='/camera/color/image_raw')
  parser.add_argument('--depth_topic', type=str, default='/camera/aligned_depth_to_color/image_raw')
  parser.add_argument('--artifacts_folder', type=str, default='/media/bowen/56c8da60-8656-47c3-b940-e928a3d4ec3b/artifacts_se3_tracknet')
  parser.add_argument('--camera_frame_name', type=str, default='/camera_color_optical_frame')
  parser.add_argument('--object_frame_name', type=str, default='/ob')


  args = parser.parse_args()

  artifact_dir = f'{args.artifacts_folder}/artifacts-{args.artifact_id}'
  ckpt_dir = '{}/model_best_val.pth.tar'.format(artifact_dir)
  config_path = '{}/code_backup{}/config.yml'.format(artifact_dir,args.artifact_id)
  mean_std_path = artifact_dir

  print('ckpt_dir:',ckpt_dir)

  with open(config_path, 'r') as ff:
    config = yaml.safe_load(ff)

  dataset_info_path = f"{artifact_dir}/code_backup{args.artifact_id}/dataset_info.yml"
  print('dataset_info_path',dataset_info_path)
  with open(dataset_info_path,'r') as ff:
    dataset_info = yaml.safe_load(ff)

  images_mean = np.load(os.path.join(mean_std_path, "mean.npy"))
  images_std = np.load(os.path.join(mean_std_path, "std.npy"))
  print('images_mean',images_mean)
  print('images_std',images_std)

  debug = False

  pose_init = args.pose_init_file
  print('pose_init:\n',pose_init)
  tracker = Tracker(dataset_info, images_mean, images_std,ckpt_dir,trans_normalizer=dataset_info['max_translation'],rot_normalizer=dataset_info['max_rotation'])
  ros_tracker = TrackerRos(tracker,pose_init=pose_init)

  rate = rospy.Rate(60.0)

  while not rospy.is_shutdown():
    try:
      ros_tracker.on_track()
    except Exception as e:
      print('ERROR: {}'.format(e))
    rate.sleep()

