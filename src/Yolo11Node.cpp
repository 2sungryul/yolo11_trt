#include "yolo11_trt/Yolo11TRT.hpp"

int main(int argc, char** argv) {
    /*if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <image_file>" << std::endl;
        std::cerr << "Example: " << argv[0] << " yolo11.engine input.jpg" << std::endl;
        return -1;
    }    
    std::string engine_path = argv[1];
    std::string image_path = argv[2];*/
    
    std::string engine_path = "/home/linux/ros2_ws/src/yolo11_trt/src/yolo11n.engine"; 
    std::string image_path = "/home/linux/ros2_ws/src/yolo11_trt/src/test.jpg";
        
    try {
        // Initialize YOLO11 detector
        std::cout << "Loading TensorRT engine..." << std::endl;
        YOLO11TensorRT detector(engine_path);
        
        //auto start = std::chrono::high_resolution_clock::now();

        // Load image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Cannot load image: " << image_path << std::endl;
            return -1;
        }

        std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;
        
        // Run inference
        std::cout << "Running inference..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<Detection> detections = detector.inference(image, 0.05f, 0.45f);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Inference completed in " << duration.count() << " ms" << std::endl;
        std::cout << "Detected " << detections.size() << " objects:" << std::endl;
        
        // Draw detections on image
        detector.drawDetections(image, detections);
        
        // Save result
        std::string output_path = "/home/linux/ros2_ws/src/yolo11_trt/src/output.jpg";
        cv::imwrite(output_path, image);
        std::cout << "Result saved to: " << output_path << std::endl;
        
        // Display result (optional)
        cv::namedWindow("YOLO11 Detection", cv::WINDOW_NORMAL);
        cv::imshow("YOLO11 Detection", image);
        std::cout << "Press any key to exit..." << std::endl;
        cv::waitKey(0);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
