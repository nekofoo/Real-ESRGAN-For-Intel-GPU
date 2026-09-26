#include "openvino/openvino.hpp"
#include "opencv2/opencv.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

// OpenVINO names a real device either after its plugin ("CPU", "GPU", "NPU") or,
// when a plugin exposes several devices, with an index ("GPU.0", "GPU.1").
// Inference modes append a candidate list after a colon ("AUTO:GPU,CPU"), which
// is why matching on a substring is unsafe: "AUTO:GPU,CPU" contains "GPU" but
// describes a scheduler that may still place the model on the CPU.
static string plugin_of(const string& device) {
    const size_t colon = device.find(':');
    return colon == string::npos ? device : device.substr(0, colon);
}

static bool is_gpu_device(const string& device) {
    const string plugin = plugin_of(device);
    if (plugin == "GPU") {
        return true;
    }
    if (plugin.compare(0, 4, "GPU.") != 0) {
        return false;
    }
    // Only a plain numeric index counts, so names like "GPU.foo" are rejected.
    return plugin.find_first_not_of("0123456789", 4) == string::npos;
}

static int gpu_index(const string& device) {
    const string plugin = plugin_of(device);
    if (plugin == "GPU") {
        return 0;
    }
    return is_gpu_device(device) ? atoi(plugin.c_str() + 4) : -1;
}

// Prefer the bare "GPU" plugin, which targets the default GPU, then the lowest
// indexed "GPU.N". Inference modes are never selected automatically.
static string pick_gpu_device(const vector<string>& devices) {
    string best;
    int best_index = -1;
    for (const auto& device : devices) {
        if (!is_gpu_device(device)) {
            continue;
        }
        const int index = gpu_index(device);
        if (best.empty() || (best != "GPU" && index < best_index)) {
            best = device;
            best_index = index;
        }
    }
    return best;
}

class Upscaler_OV {
private:
    ov::Core core;
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
    // An empty requested_device keeps the previous behaviour of auto-selecting a
    // GPU and falling back to the CPU.
    explicit Upscaler_OV(const string& requested_device = "") {
        const string model_all_local_path = "model/esr_dynamic.xml";

        filesystem::create_directories("compile_cache/");
        core.set_property(ov::cache_dir("compile_cache"));

        const vector<string> device_list = core.get_available_devices();

        cout << "Available devices: ";
        for (const auto& device : device_list) {
            cout << device << " ";
        }
        cout << endl;

        string selected_device;
        if (!requested_device.empty()) {
            // Accept only a device OpenVINO actually reports, so a typo fails
            // loudly instead of silently compiling somewhere unexpected.
            if (find(device_list.begin(), device_list.end(), requested_device) == device_list.end()) {
                string available;
                for (const auto& device : device_list) {
                    if (!available.empty()) {
                        available += ", ";
                    }
                    available += device;
                }
                throw runtime_error("Requested device '" + requested_device +
                                    "' is not available. Available devices: " + available);
            }
            selected_device = requested_device;
        } else {
            selected_device = pick_gpu_device(device_list);
            if (selected_device.empty()) {
                cout << "No GPU device found, using CPU." << endl;
                selected_device = "CPU";
            }
        }

        cout << "Selected device: " << selected_device << endl;

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

        // swapRB converts the BGR data from cv::imread to the RGB order the
        // Real-ESRGAN model expects, and run() converts back to BGR for imwrite.
        cv::Mat blob = cv::dnn::blobFromImage(image_resized,
                                     1.0/255.0,
                                     cv::Size(image_resized.size()),
                                     cv::Scalar(0,0,0),
                                     true,   // swapRB: BGR -> RGB
                                     false); // crop

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
        cout << "Usage: " << argv[0] << " -i <input_image> [-o <output_image>] [-d <device>]" << endl;
        return 0;
    }

    string input_path;
    string output_path;
    string device;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "-d" && i + 1 < argc) {
            device = argv[++i];
        } else {
            cerr << "Ignoring unknown or incomplete argument: " << arg << endl;
        }
    }

    cv::Mat org = cv::imread(input_path);
    if (org.empty()) {
        cout << "Failed to open image" << endl;
        return 1;
    }

    try {
        Upscaler_OV esr(device);
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
