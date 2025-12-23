#include "yolo11_trt/Yolo11TRT.hpp"

// COCO class names
/*const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};*/

const std::vector<std::string> COCO_CLASSES = {
    "big robot", "small robot"
};

// Generate random colors for each class
std::vector<cv::Scalar> YOLO11TensorRT::generateColors(int num_classes) {
    std::vector<cv::Scalar> colors;
    std::srand(42);  // Fixed seed for consistency

    for (int i = 0; i < num_classes; ++i) {
        colors.push_back(cv::Scalar(std::rand() % 256,
            std::rand() % 256,
            std::rand() % 256));
    }
    return colors;
}

void YOLO11TensorRT::drawDetections(cv::Mat& image, const std::vector<Detection>& detections) {
    // Print detection results
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        std::cout << "  [" << i << "] "
            << COCO_CLASSES[det.class_id]
            << " - confidence: " << det.confidence
            << " - bbox: (" << det.bbox.x << ", " << det.bbox.y << ", "
            << det.bbox.width << ", " << det.bbox.height << ")" << std::endl;
    }

    // Draw detections on image
    static auto colors = generateColors(COCO_CLASSES.size());

    for (const auto& det : detections) {
        // Draw bounding box
        cv::rectangle(image, det.bbox, colors[det.class_id], 2);

        // Prepare label text
        std::string label = COCO_CLASSES[det.class_id] + ": " +
            std::to_string(static_cast<int>(det.confidence * 100)) + "%";

        // Calculate text size for background
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
            0.5, 1, &baseline);

        // Draw label background
        cv::Point label_pos(det.bbox.x, det.bbox.y - 5);
        cv::Rect label_bg(label_pos.x, label_pos.y - text_size.height - 5,
            text_size.width + 5, text_size.height + 10);
        cv::rectangle(image, label_bg, colors[det.class_id], -1);

        // Draw label text
        cv::putText(image, label,
            cv::Point(label_pos.x + 2, label_pos.y - 2),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

YOLO11TensorRT::YOLO11TensorRT(const std::string& engine_path) 
    : runtime_(nullptr), engine_(nullptr), context_(nullptr) {
    
    input_width_ = 640;
    input_height_ = 640;
    output_size_ = 8400;
    //input_width_ = 416;
    //input_height_ = 416;
    //output_size_ = 3549;

    warmed_up = false;

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s\n", prop.name);
    
    if (!loadEngine(engine_path)) {
        throw std::runtime_error("Failed to load TensorRT engine");
    }
    
    // Create CUDA stream
    cudaStreamCreate(&stream_);
    
    // Allocate buffers
    size_t input_size = INPUT_CHANNELS * input_height_ * input_width_ * sizeof(float);
    size_t output_size = (4 + NUM_CLASSES) * output_size_ * sizeof(float);
    
    cudaMalloc(&buffers_[0], input_size);
    cudaMalloc(&buffers_[1], output_size);

    if (!warmed_up) warmup();
}

YOLO11TensorRT::~YOLO11TensorRT() {
    cudaStreamDestroy(stream_);
    cudaFree(buffers_[0]);
    cudaFree(buffers_[1]);
    
    if (context_) delete context_;
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
}

void YOLO11TensorRT::warmup() {
    if (warmed_up) return;

    std::cout << "Warming up TensorRT engine..." << std::endl;

    // ���� �Է� ������ ����
    size_t input_size = 1 * INPUT_CHANNELS * input_width_ * input_height_ * sizeof(float);
    std::vector<float> dummy_input(1 * INPUT_CHANNELS * input_width_ * input_height_, 0.5f);

    // Set input tensor address
    const char* input_name = engine_->getIOTensorName(0);
    context_->setTensorAddress(input_name, buffers_[0]);
    // Set output tensor address
    const char* output_name = engine_->getIOTensorName(1);
    context_->setTensorAddress(output_name, buffers_[1]);
    
    // ù ����� (����)
    
    cudaMemcpyAsync(buffers_[0], dummy_input.data(), input_size,cudaMemcpyHostToDevice, stream_);

    bool success = context_->enqueueV3(stream_);
    if (!success) {
        std::cerr << "Error: Warmup Inference failed" << std::endl;
        return;
    }

    cudaStreamSynchronize(stream_);    

    warmed_up = true;
    std::cout << "Warmup complete!" << std::endl;
}


bool YOLO11TensorRT::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Cannot open engine file: " << engine_path << std::endl;
        return false;
    }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();
    
    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
        std::cerr << "Error: Failed to create runtime" << std::endl;
        return false;
    }
    
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) {
        std::cerr << "Error: Failed to deserialize engine" << std::endl;
        return false;
    }
    
    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "Error: Failed to create execution context" << std::endl;
        return false;
    }
    
    // Set input shape for enqueueV3
    const char* input_name = engine_->getIOTensorName(0);
    nvinfer1::Dims input_dims = engine_->getTensorShape(input_name);
    
    // If dynamic shapes, set the actual input shape
    if (input_dims.d[0] == -1) {
        input_dims.d[0] = 1;  // batch size
        input_dims.d[1] = INPUT_CHANNELS;
        input_dims.d[2] = input_height_;
        input_dims.d[3] = input_width_;
        context_->setInputShape(input_name, input_dims);
    }
    
    std::cout << "TensorRT engine loaded successfully" << std::endl;
    std::cout << "Input tensor name: " << input_name << std::endl;
    std::cout << "Output tensor name: " << engine_->getIOTensorName(1) << std::endl;
    
    return true;
}

void YOLO11TensorRT::preprocessImage(const cv::Mat& image, float* input_data) {
    // Resize image with letterbox padding
    int width = image.cols;
    int height = image.rows;
    
    float scale = std::min(static_cast<float>(input_width_) / width, 
                           static_cast<float>(input_height_) / height);
    
    int new_width = static_cast<int>(width * scale);
    int new_height = static_cast<int>(height * scale);
    
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height));
    
    // Create padded image
    cv::Mat padded(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
    int top = (input_height_ - new_height) / 2;
    int left = (input_width_ - new_width) / 2;
    resized.copyTo(padded(cv::Rect(left, top, new_width, new_height)));
    
    // Convert to float and normalize to [0, 1]
    cv::Mat float_image;
    padded.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
    
    // Convert HWC to CHW format
    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);
    
    std::cout << "padded size:"<<input_height_ << ',' << input_width_ << std::endl;

    int channel_size = input_height_ * input_width_;
    for (int c = 0; c < 3; ++c) {
        memcpy(input_data + c * channel_size, 
               channels[c].data, 
               channel_size * sizeof(float));
    }
}

std::vector<Detection> YOLO11TensorRT::inference(const cv::Mat& image, 
                                                  float conf_threshold,
                                                  float iou_threshold) {
    // Preprocess image
    std::vector<float> input_data(INPUT_CHANNELS * input_height_ * input_width_);
    preprocessImage(image, input_data.data());
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Copy input to device
    cudaMemcpyAsync(buffers_[0], input_data.data(), 
                    input_data.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    
    // Set input tensor address
    const char* input_name = engine_->getIOTensorName(0);
    context_->setTensorAddress(input_name, buffers_[0]);
    
    // Set output tensor address
    const char* output_name = engine_->getIOTensorName(1);
    context_->setTensorAddress(output_name, buffers_[1]);
    
    //auto start = std::chrono::high_resolution_clock::now();
    
    // Run inference using enqueueV3
   
    bool success = context_->enqueueV3(stream_);
    if (!success) {
        std::cerr << "Error: Inference failed" << std::endl;
        return {};
    }
      
    //auto end = std::chrono::high_resolution_clock::now();
         
    // Copy output to host
    size_t output_buffer_size = (4 + NUM_CLASSES) * output_size_;
    std::vector<float> output_data(output_buffer_size);
    cudaMemcpyAsync(output_data.data(), buffers_[1],
                    output_buffer_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    
    cudaStreamSynchronize(stream_);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Inference core completed in " << duration.count() << " ms" << std::endl;
    
    // Postprocess
    return postprocess(output_data.data(), image.size(), conf_threshold, iou_threshold);
}

std::vector<Detection> YOLO11TensorRT::postprocess(float* output,
                                                    const cv::Size& original_size,
                                                    float conf_threshold,
                                                    float iou_threshold) {
    std::vector<Detection> detections;
    
    // Calculate scaling factors
    float scale = std::min(static_cast<float>(input_width_) / original_size.width,
                          static_cast<float>(input_height_) / original_size.height);
    
    int new_width = static_cast<int>(original_size.width * scale);
    int new_height = static_cast<int>(original_size.height * scale);
    int pad_left = (input_width_ - new_width) / 2;
    int pad_top = (input_height_ - new_height) / 2;
    
    // Output format: [1, 84, 8400] transposed to [8400, 84]
    for (int i = 0; i < output_size_; ++i) {
        // Get bounding box (first 4 elements)
        float x_center = output[i];  // Note: output is in format [84, 8400]
        float y_center = output[output_size_ + i];
        float width = output[2 * output_size_ + i];
        float height = output[3 * output_size_ + i];
        
        // Get class scores (next 80 elements)
        float max_score = 0.0f;
        int class_id = -1;
        
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float score = output[(4 + c) * output_size_ + i];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }
        
        // Filter by confidence threshold
        if (max_score < conf_threshold) continue;
        
        // Convert to original image coordinates
        float x1 = (x_center - width / 2.0f - pad_left) / scale;
        float y1 = (y_center - height / 2.0f - pad_top) / scale;
        float x2 = (x_center + width / 2.0f - pad_left) / scale;
        float y2 = (y_center + height / 2.0f - pad_top) / scale;
        
        // Clamp to image boundaries
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_size.width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_size.height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(original_size.width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(original_size.height)));
        
        Detection det;
        det.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
        det.confidence = max_score;
        det.class_id = class_id;
        
        detections.push_back(det);
    }
    
    // Apply NMS
    nms(detections, iou_threshold);
    
    return detections;
}

float YOLO11TensorRT::calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    int intersection_area = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int box1_area = box1.width * box1.height;
    int box2_area = box2.width * box2.height;
    int union_area = box1_area + box2_area - intersection_area;
    
    return static_cast<float>(intersection_area) / union_area;
}

void YOLO11TensorRT::nms(std::vector<Detection>& detections, float iou_threshold) {
    // Sort by confidence
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            
            // Only apply NMS for same class
            if (detections[i].class_id != detections[j].class_id) continue;
            
            float iou = calculateIoU(detections[i].bbox, detections[j].bbox);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    // Remove suppressed detections
    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
                      [&suppressed, &detections](const Detection& det) {
                          return suppressed[&det - &detections[0]];
                      }),
        detections.end()
    );
}
