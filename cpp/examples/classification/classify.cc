#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "edgetpu_c.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace {
constexpr size_t kBmpFileHeaderSize = 14;
constexpr size_t kBmpInfoHeaderSize = 40;
constexpr size_t kBmpHeaderSize = kBmpFileHeaderSize + kBmpInfoHeaderSize;

int32_t ToInt32(const char p[4]) {
  return (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0];
}

std::vector<uint8_t> ReadBmpImage(const char* filename,
                                  int* out_width = nullptr,
                                  int* out_height = nullptr,
                                  int* out_channels = nullptr) {
  assert(filename);

  std::ifstream file(filename, std::ios::binary);
  if (!file) return {};  // Open failed.

  char header[kBmpHeaderSize];
  if (!file.read(header, sizeof(header))) return {};  // Read failed.

  const char* file_header = header;
  const char* info_header = header + kBmpFileHeaderSize;

  if (file_header[0] != 'B' || file_header[1] != 'M')
    return {};  // Invalid file type.

  const int channels = info_header[14] / 8;
  if (channels != 1 && channels != 3) return {};  // Unsupported bits per pixel.

  if (ToInt32(&info_header[16]) != 0) return {};  // Unsupported compression.

  const uint32_t offset = ToInt32(&file_header[10]);
  if (offset > kBmpHeaderSize &&
      !file.seekg(offset - kBmpHeaderSize, std::ios::cur))
    return {};  // Seek failed.

  int width = ToInt32(&info_header[4]);
  if (width < 0) return {};  // Invalid width.

  int height = ToInt32(&info_header[8]);
  const bool top_down = height < 0;
  if (top_down) height = -height;

  const int line_bytes = width * channels;
  const int line_padding_bytes =
      4 * ((8 * channels * width + 31) / 32) - line_bytes;
  std::vector<uint8_t> image(line_bytes * height);
  for (int i = 0; i < height; ++i) {
    uint8_t* line = &image[(top_down ? i : (height - 1 - i)) * line_bytes];
    if (!file.read(reinterpret_cast<char*>(line), line_bytes))
      return {};  // Read failed.
    if (!file.seekg(line_padding_bytes, std::ios::cur))
      return {};  // Seek failed.
    if (channels == 3) {
      for (int j = 0; j < width; ++j) std::swap(line[3 * j], line[3 * j + 2]);
    }
  }

  if (out_width) *out_width = width;
  if (out_height) *out_height = height;
  if (out_channels) *out_channels = channels;
  return image;
}

std::vector<std::string> ReadLabels(const std::string& filename) {
  std::ifstream file(filename);
  if (!file) return {};  // Open failed.

  std::vector<std::string> lines;
  for (std::string line; std::getline(file, line);) lines.emplace_back(line);
  return lines;
}

std::string GetLabel(const std::vector<std::string>& labels, int label) {
  if (label >= 0 && label < labels.size()) return labels[label];
  return std::to_string(label);
}

std::vector<float> Dequantize(const TfLiteTensor& tensor) {
  const auto* data = reinterpret_cast<const uint8_t*>(tensor.data.data);
  std::vector<float> result(tensor.bytes);
  for (int i = 0; i < tensor.bytes; ++i)
    result[i] = tensor.params.scale * (data[i] - tensor.params.zero_point);
  return result;
}

std::vector<std::pair<int, float>> Sort(const std::vector<float>& scores,
                                        float threshold) {
  std::vector<const float*> ptrs(scores.size());
  std::iota(ptrs.begin(), ptrs.end(), scores.data());
  auto end = std::partition(ptrs.begin(), ptrs.end(),
                            [=](const float* v) { return *v >= threshold; });
  std::sort(ptrs.begin(), end,
            [](const float* a, const float* b) { return *a > *b; });

  std::vector<std::pair<int, float>> result;
  result.reserve(end - ptrs.begin());
  for (auto it = ptrs.begin(); it != end; ++it)
    result.emplace_back(*it - scores.data(), **it);
  return result;
}
}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr << argv[0]
              << " <model_file> <label_file> <image_file> <threshold>"
              << std::endl;
    return 1;
  }

  const std::string model_file = argv[1];
  const std::string label_file = argv[2];
  const std::string image_file = argv[3];
  const float threshold = std::stof(argv[4]);

  // Find TPU device.
  size_t num_devices;
  std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
      edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

  if (num_devices == 0) {
    std::cerr << "No connected TPU found" << std::endl;
    return 1;
  }
  const auto& device = devices.get()[0];

  // Load labels.
  auto labels = ReadLabels(label_file);
  if (labels.empty()) {
    std::cerr << "Cannot read labels from " << label_file << std::endl;
    return 1;
  }

  // Load image.
  int image_bpp, image_width, image_height;
  auto image =
      ReadBmpImage(image_file.c_str(), &image_width, &image_height, &image_bpp);
  if (image.empty()) {
    std::cerr << "Cannot read image from " << image_file << std::endl;
    return 1;
  }

  // Load model.
  auto model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
  if (!model) {
    std::cerr << "Cannot read model from " << model_file << std::endl;
    return 1;
  }

  // Create interpreter.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
    std::cerr << "Cannot create interpreter" << std::endl;
    return 1;
  }

  auto* delegate =
      edgetpu_create_delegate(device.type, device.path, nullptr, 0);
  interpreter->ModifyGraphWithDelegate({delegate, edgetpu_free_delegate});

  // Allocate tensors.
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Cannot allocate interpreter tensors" << std::endl;
    return 1;
  }

  // Set interpreter input.
  const auto* input_tensor = interpreter->input_tensor(0);
  if (input_tensor->type != kTfLiteUInt8 ||           //
      input_tensor->dims->data[0] != 1 ||             //
      input_tensor->dims->data[1] != image_height ||  //
      input_tensor->dims->data[2] != image_width ||   //
      input_tensor->dims->data[3] != image_bpp) {
    std::cerr << "Input tensor shape does not match input image" << std::endl;
    return 1;
  }

  std::copy(image.begin(), image.end(),
            interpreter->typed_input_tensor<uint8_t>(0));

  // Run inference.
  if (interpreter->Invoke() != kTfLiteOk) {
    std::cerr << "Cannot invoke interpreter" << std::endl;
    return 1;
  }

  // Get interpreter output.
  auto results = Sort(Dequantize(*interpreter->output_tensor(0)), threshold);
  for (auto& result : results)
    std::cout << std::setw(7) << std::fixed << std::setprecision(5)
              << result.second << GetLabel(labels, result.first) << std::endl;

  return 0;
}
