# yolo11_trt

model : Yolo11n_custom

dependencies : wsl2-ubuntu24.04, ROS2 Jazzy, TensorRT 10.14.1, Cuda 12.9.1, cudnn 8.9.7

TensorRT engine file : yolo11n.engine

yolo11_node : inference node using single image file

pub : publish a topic with image captured from video file(mp4)

sub : inference node using image topic received from pub node

engine inference time < about 8msec on RTX 4070 Ti super

total callback processing time < about 12msec on RTX 4070 Ti super
