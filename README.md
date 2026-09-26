# Real-ESRGAN For Intel GPU
The project implements [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) inference using [OpenVINO](https://github.com/openvinotoolkit/openvino) which is optimized for Intel platform.
## Features
Up to 5x faster than the [Real-ESRGAN ncnn Vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) implementation.
## Limitation
Input image's width/height is limited to 1280 pixels to reduce memory usage. large images will be resized.

eg. max output resolution of a 16:9 image will be 5120 * 2880
## Usage
```bash
realesrgan-ov.exe -i <input_image> [-o <output_image>] [-d <device>]

If -o is not specified, the image will be saved to the input directory as {input_image_filename_no_extention}_x4.png
```

### Device selection
By default the first available GPU is used, falling back to `CPU` when no GPU is present. The available devices are printed on startup, for example:

```
Available devices: CPU GPU.0 GPU.1 NPU
Selected device: GPU.0
```

Only real GPU devices are auto-selected, meaning `GPU` and indexed names such as `GPU.0` or `GPU.1`. Inference modes are never chosen automatically, because `AUTO:GPU,CPU` and friends are schedulers that may still place the model on the CPU.

Use `-d` to override the choice:

```bash
realesrgan-ov.exe -i input.png -d GPU.1
realesrgan-ov.exe -i input.png -d NPU
realesrgan-ov.exe -i input.png -d CPU
```

`-d` accepts any device reported by OpenVINO, including the inference modes (`AUTO`, `HETERO:...`, `MULTI:...`, `BATCH:...`). A device that is not available is rejected with an error listing the valid names.
