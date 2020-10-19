CUDA_VISIBLE_DEVICES=0 \
python /media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/github/iros20-6d-pose-tracking/predict.py \
  --train_data_path /media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/github/end_to_end_tracking/Experiments/MINE/bleach_cleanser/train_data_blender_DR \
  --ckpt_dir /home/bowen/debug/bleach_cleanser/model_best_val.pth.tar \
  --mean_std_path /home/bowen/debug/bleach_cleanser \
  --class_id 12 \
