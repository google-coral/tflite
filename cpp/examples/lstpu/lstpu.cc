// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <iostream>
#include <memory>

#include "edgetpu_c.h"

std::string ToString(edgetpu_device_type type) {
  switch (type) {
    case EDGETPU_APEX_PCI:
      return "PCI";
    case EDGETPU_APEX_USB:
      return "USB";
  }
  return "Unknown";
}

int main(int argc, char* argv[]) {
  size_t num_devices;
  std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
      edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

  for (size_t i = 0; i < num_devices; ++i) {
    const auto& device = devices.get()[i];
    std::cout << i << " " << ToString(device.type) << " " << device.path
              << std::endl;
  }

  return 0;
}
