# iros20-6d-pose-tracking

This is the official implementation of our paper "se(3)-TrackNet: Data-driven 6D Pose Tracking by Calibrating Image Residuals in Synthetic Domains" accepted in International Conference on Intelligent Robots and Systems (IROS) 2020.
[[PDF]](https://arxiv.org/abs/2007.13866)

**Abstract:** Tracking the 6D pose of objects in video sequences is important for  robot manipulation. This  task, however, introduces multiple challenges: (i) robot manipulation involves significant occlusions; (ii) data and annotations are troublesome and difficult to collect for 6D poses, which complicates machine learning solutions, and (iii) incremental error drift often accumulates in long term tracking to necessitate re-initialization of the object's pose. This work proposes a data-driven optimization approach for long-term, 6D pose tracking. It aims to identify the optimal relative pose given the current RGB-D observation and a synthetic image conditioned on the previous best estimate and the object's model. The key contribution in this context is a novel neural network architecture, which appropriately disentangles the feature encoding to help reduce domain shift, and an effective 3D orientation representation via Lie Algebra. Consequently, even when the network is trained only with synthetic data can work effectively over real images. Comprehensive experiments over benchmarks - existing ones as well as a new dataset with significant occlusions related to object manipulation - show that the proposed approach achieves consistently robust estimates and outperforms alternatives, even though they have been trained with real images. The approach is also the most computationally efficient among the alternatives and achieves a tracking frequency of 90.9Hz.


**Applications:** model-based RL, manipulation, AR/VR, human-robot-interaction, automatic 6D pose labeling.



# Bibtex
```bibtex
	@conference {wense3tracknet,
	title = {se(3)-TrackNet: Data-driven 6D Pose Tracking by Calibrating Image Residuals in Synthetic Domains},
	booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
	year = {2020},
	month = {10/2020},
	address = {Las Vegas, NV},
	url = {http://arxiv.org/abs/2007.13866},
	author = {Wen, B. and Mitash, C. and Ren, B. and Bekris, K. E.}
}
```


# Supplementary Video:
Click to watch

[<img src="./media/youtube_thumbnail.jpg" width="480">](https://www.youtube.com/watch?v=dhqM0hZmGR4)


#  Results on YCB

<img src="./media/occlusion.gif" width="480">

<img src="./media/curves.jpg" width="480">

<img src="./media/ycb_results.jpg">



# About YCBInEOAT Dataset

<img src="./media/eoat1.jpg" width="1200">

<img src="./media/eoat2.jpg" width="1200">


Due to the lack of suitable dataset about RGBD-based 6D pose tracking in robotic manipulation, a novel dataset is developed in this work. It has these key attributes:

* Real manipulation tasks

* 3 kinds of end-effectors

* 5 YCB objects

* 9 videos for evaluation, 7449 RGBD in total

* Ground-truth poses annotated for each frame

* Forward-kinematics recorded

* Camera extrinsic parameters calibrated

Link to download this dataset is provided below under 'Data Preparation'.
Example manipulation sequence:

<img src="./media/manipulation1.gif" width="480">


Current benchmark:

<img src="./media/ycbineoat_benchmark.jpg" width="480">

More details are in the paper and supplementary video.



# Data Download
1. [YCB_Video dataset  ](https://rse-lab.cs.washington.edu/projects/posecnn/)
1. [data_organized](https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCB_Video_data_organized/) (15G). It is the reorganized YCB_Video data for convenience. Then extract it under your YCB_Video dataset directory, e.g. YCB_Video_Dataset/data_organized/0048/
1. [YCBInEOAT dataset](https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCBInEOAT/) (22G)
1. Our [pretrained weights on YCB_Video](https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCB_weights.zip) and [pretrained weights on YCBInEOAT](https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCBInEOAT_weights.zip)
1. Our generated [synthetic YCB_Video training data](https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCB_traindata/) (~15G for each object) and  [synthetic YCBInEOAT trainnig data](https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCBInEOAT_traindata/) (~15G for each object)


<img src="./media/syndata_gen.gif" width="480" style="position:relative;left:5%">

6. [se(3)-TrackNet's output pose estimations of YCB_Video](https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/Ours_YCB_results.tar.gz) and [se(3)-TrackNet's output pose estimations of YCBInEOAT](https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCBInEOAT_results/)






# Prediction on YCB_Video and YCBInEOAT
Please refer to `predict.py` and `predict.sh`

# Benchmarking
Please refer to `eval_ycb.py` and `eval_ycbineoat.py`


# Training
1. Edit the config.yml. Make sure the paths are correct. Other settings need not be changed in most cases.
1. Then  `python train.py`


# To Appear
- code for synthetic training data generation for your own use case.

