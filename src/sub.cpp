#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "opencv2/opencv.hpp"
#include <memory>
#include <functional>
#include <iostream>
#include "yolo11_trt/Yolo11TRT.hpp"
#include <chrono>
using std::placeholders::_1;

std::string enginePath = "/home/linux/ros2_ws/src/yolo11_trt/src/yolo11n.engine";
std::string imagePath = "/home/linux/ros2_ws/src/yolo11_trt/src/test.jpg";

class Sub : public rclcpp::Node
{
private:
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_;
    std::shared_ptr<YOLO11TensorRT> detector_;
    void sub_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) const;
public:
    Sub();
};

Sub::Sub() : Node("imagesub")
{
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));
    sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>("image/compressed", qos_profile, std::bind(&Sub::sub_callback, this, _1));
    detector_ = std::make_shared<YOLO11TensorRT>(enginePath);
}

void Sub::sub_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) const
{
    cv::Mat frame = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    RCLCPP_INFO(this->get_logger(), "Received Image : %s,%d,%d", msg->format.c_str(),frame.rows,frame.cols);
    
    std::cout << "Image loaded: " << frame.cols << "x" << frame.rows << std::endl;
       
    // Run inference
    std::cout << "Running inference..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<Detection> detections = detector_->inference(frame, 0.01f, 0.45f);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Inference completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Detected " << detections.size() << " objects:" << std::endl;
    
    // Draw detections on image
    detector_->drawDetections(frame, detections);
            
    // Display result (optional)
    cv::namedWindow("YOLO11 Detection", cv::WINDOW_NORMAL);
    cv::imshow("YOLO11 Detection", frame);        
    cv::waitKey(1);

    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start);
    std::cout << "callback completed in " << duration2.count() << " ms" << std::endl;
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Sub>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
