#include "openvino/openvino.hpp"
#include "opencv2/opencv.hpp"
#include <filesystem>
#include <iostream>
#include <string>
#include <chrono>

using namespace std;

class Upscaler_OV {
private:    
    ov::CompiledModel compiled_model;
    ov::Output<const ov::Node> output_tensor;
    ov::InferRequest infer_request;

    cv::Mat resize_image(const cv::Mat& image, int max_size = 1280) {
        int original_height = image.rows;
        int original_width = image.cols;
        
        if (max(original_height, original_width) <= 1280) {
            return image;
        }

        int new_width, new_height;
        if (original_width > original_height) {
            new_width = max_size;
            new_height = static_cast<int>((static_cast<float>(new_width) / original_width) * original_height);
        } else {
            new_height = max_size;
            new_width = static_cast<int>((static_cast<float>(new_height) / original_height) * original_width);
        }

        cout << "resizing" << endl;
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(new_width, new_height));        
        return resized_image;
    }

public:
    Upscaler_OV() {
        
        string model_all_local_path = "model/esr_dynamic.xml";        
        
        ov::Core core;
        filesystem::create_directories("compile_cache/");
        core.set_property(ov::cache_dir("compile_cache"));        
        
        vector<string> device_list = core.get_available_devices();
        cout << "Available devices: ";
        for (const auto& device : device_list) {
            cout << device << " ";
        }
        cout << endl;
        
        string selected_device = "CPU";
        for (const auto& device : device_list) {
            if (device.find("GPU") != string::npos) {
                selected_device = device;
                cout << "Selected device: " << selected_device << endl;
                break;
            }
        }
        if (selected_device == "CPU") {
            cout << "No GPU device found, using CPU." << endl;
        }
        
        compiled_model = core.compile_model(model_all_local_path, selected_device);
        output_tensor = compiled_model.output();
        infer_request = compiled_model.create_infer_request();
    }

    cv::Mat run(const cv::Mat& image) {
        auto t1 = chrono::high_resolution_clock::now();
        
        if (image.empty()) {
            throw runtime_error("Can't open the image");
        }

        cout << "Input shape: " << image.size() << endl;
        auto image_resized = resize_image(image);

        cv::Mat blob = cv::dnn::blobFromImage(image_resized,    
                                     1.0/255.0,     
                                     cv::Size(image_resized.size()), 
                                     cv::Scalar(0,0,0),   
                                     true,    
                                     false);  
        

         
        const ov::element::Type& input_type = ov::element::f32;
        ov::Shape input_shape = {1, 3, static_cast<size_t>(image_resized.rows), static_cast<size_t>(image_resized.cols)};        
        ov::Tensor input_tensor(input_type, input_shape, blob.data);

        cout << "Start inferring" << endl;
        
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        auto output = infer_request.get_output_tensor();

        auto t2 = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1);
        cout << "Real-ESRGAN execution time: " << duration.count() / 1000.0 << " seconds" << endl;

        auto shape_out = output.get_shape();
        vector<int> sizes = {1,(int)shape_out[1],(int)shape_out[2],(int)shape_out[3]};
        cv::Mat blob1(4, sizes.data(), CV_32F, output.data<float>());
        vector<cv::Mat> images;
        cv::dnn::imagesFromBlob(blob1, images);       
        cv::Mat image_u8;
        ((cv::Mat)(images[0] * 255.0)).convertTo(image_u8, CV_8U);
        cv::Mat bgr_image;
        cv::cvtColor(image_u8, bgr_image, cv::COLOR_RGB2BGR);
        cout << "Output shape: " << bgr_image.size() << endl;
        return bgr_image;        
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " -i <input_image> [-o <output_image>]" << endl;
        return 0;
    }

    string input_path;
    string output_path;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    cv::Mat org = cv::imread(input_path);
    if (org.empty()) {
        cout << "Failed to open image" << endl;
        return 1;
    }

    try {
        Upscaler_OV esr;
        cv::Mat img = esr.run(org);

        if (output_path.empty()) {
            filesystem::path input_file(input_path);
            string filename = input_file.stem().string() + "_x4.png";
            output_path = (input_file.parent_path() / filename).string();
        }
        
        cout << "out: " << output_path << endl;
        cv::imwrite(output_path, img);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}