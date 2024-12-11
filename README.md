# Real-ESRGAN For Intel GPU
The [OpenVINO](https://github.com/openvinotoolkit/openvino) implementation of [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), optimized for Intel platform.
## Features
Up to 5x faster than the [Real-ESRGAN ncnn Vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) implementation.
## Limitation
Input image's width/height is limited to 1280 pixels to reduce memory usage. large images will be resized.

eg. max output resolution of a 16:9 image will be 5120 * 2880
## Usage
```bash
realesrgan-ov.exe -i <input_image> [-o <output_image>]

If -o is not specified, the image will be saved to the input directory as {input_image_filename_no_extention}_x4.png
```
