# yolo11_trt

model : Yolo11n_custom

class : 2(small robot, big robot)

model input size : 3x640x640

model output size : 6x8400

dependencies : wsl2-ubuntu24.04, ROS2 Jazzy, TensorRT 10.14.1, Cuda 12.9.1, cudnn 8.9.7

TensorRT engine file : yolo11n.engine(FP32 precision)

# node info

yolo11_node : perform an inference using a single image file(test.jpg)

pub : publish a topic with image captured from a video file(robot_video.mp4)

sub : perform an inference using an image topic received from pub node

engine inference time(enqueueV3) < about 2msec on RTX 4070 Ti super

total callback processing time < about 10msec on RTX 4070 Ti super
