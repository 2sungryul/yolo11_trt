#ifndef YOLO11_TENSORRT_H
#define YOLO11_TENSORRT_H

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <chrono>

// Detection result structure
struct Detection {
    cv::Rect bbox;
    float confidence;
    int class_id;
};

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class YOLO11TensorRT {
public:
    YOLO11TensorRT(const std::string& engine_path);
    ~YOLO11TensorRT();
    
    std::vector<Detection> inference(const cv::Mat& image, 
                                     float conf_threshold = 0.25f,
                                     float iou_threshold = 0.45f);
    void drawDetections(cv::Mat& image, const std::vector<Detection>& detections);
    std::vector<cv::Scalar> generateColors(int num_classes);
private:
    bool loadEngine(const std::string& engine_path);
    void preprocessImage(const cv::Mat& image, float* input_data);
    std::vector<Detection> postprocess(float* output, 
                                       const cv::Size& original_size,
                                       float conf_threshold,
                                       float iou_threshold);
    float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);
    void nms(std::vector<Detection>& detections, float iou_threshold);
    void warmup();
    Logger logger_;
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    
    void* buffers_[2];  // input and output buffers
    cudaStream_t stream_;
    
    int input_width_;
    int input_height_;
    int output_size_;
    bool warmed_up;
    
    //const int NUM_CLASSES = 80;
    const int NUM_CLASSES = 2;
    const int INPUT_CHANNELS = 3;
};

#endif // YOLO11_TENSORRT_H
