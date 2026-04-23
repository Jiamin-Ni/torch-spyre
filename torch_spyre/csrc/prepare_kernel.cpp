/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "prepare_kernel.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "common/json.hpp"
#include "flex/allocator/alloc_address.hpp"
#include "flex/allocator/flex_allocator.hpp"
#include "flex/runtime_stream/runtime_operation.hpp"
#include "flex/runtime_stream/runtime_stream.hpp"
#include "host_compute_step.h"
#include "job_plan_step.h"
#include "spyre_allocator.h"

// Forward declarations
namespace flex {
class RuntimeStream;
class RuntimeEntry {
 public:
  RuntimeStream* getDefaultStream(int device_id) const;
};
}  // namespace flex

namespace spyre {

// External linkage to GlobalRuntime accessor from module.cpp.
std::shared_ptr<flex::RuntimeEntry>& getGlobalRuntimeInstance();

namespace detail {

// Helper to get the global runtime
static std::shared_ptr<flex::RuntimeEntry> getGlobalRuntime() {
  return getGlobalRuntimeInstance();
}

// TODO: check CompositeAddress updates and verify
// Helper to extract CompositeAddress from c10::DataPtr
static flex::CompositeAddress ExtractCompositeAddress(const c10::DataPtr& data_ptr) {
  auto* ctx = static_cast<SharedOwnerCtx*>(data_ptr.get_context());
  if (!ctx || !ctx->owner) {
    throw std::runtime_error("Failed to get allocation context from DataPtr");
  }
  
  uint64_t dmpa = ctx->owner->DmpaAsBytes();
  size_t alloc_size = ctx->owner->SizeAsBytes();
  
  // Create CompositeAddress from the DMPA
  // region_id=0 for now, domain_id=0 for single-domain allocation
  flex::LogicalAddress logical_addr(0, dmpa);
  flex::Chunk chunk(logical_addr, alloc_size, 0);
  return flex::CompositeAddress(chunk);
}

// TODO: check CompositeAddress updates and verify
// Helper to compute CompositeAddress with offset from device_addr for program
static flex::CompositeAddress ComputeOffsetAddress(
    const flex::CompositeAddress& program_address,
    uint64_t device_addr) {
  // Calculate offset: device_addr is relative to 112GB base
  constexpr uint64_t BASE_ADDR = 112ULL * 1024 * 1024 * 1024;
  uint64_t offset = device_addr - BASE_ADDR;
  
  // Create CompositeAddress using program_address with offset
  if (program_address.chunks().empty()) {
    throw std::runtime_error("program_address has no chunks");
  }
  
  // Get the first chunk and add offset to its address
  const auto& base_chunk = program_address.chunks()[0];
  flex::LogicalAddress offset_addr(
      base_chunk.addr.region_id,
      base_chunk.addr.offset + offset);
  flex::Chunk offset_chunk(offset_addr, 0, base_chunk.domain_id);
  return flex::CompositeAddress(offset_chunk);
}

// Helper to check if a file exists
bool FileExists(const std::filesystem::path& path) {
  return std::filesystem::exists(path) &&
         std::filesystem::is_regular_file(path);
}

// Helper to read entire file into string
std::string ReadFileToString(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + path.string());
  }

  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

// Helper to parse metadata JSON if present
std::vector<std::vector<int64_t>> ParseExpectedInputShapes(
    const std::filesystem::path& metadata_path) {
  std::vector<std::vector<int64_t>> shapes;

  if (!FileExists(metadata_path)) {
    return shapes;  // Return empty if no metadata
  }

  try {
    std::string json_str = ReadFileToString(metadata_path);
    nlohmann::json metadata = nlohmann::json::parse(json_str);

    if (metadata.contains("expected_input_shapes") &&
        metadata["expected_input_shapes"].is_array()) {
      for (const auto& shape_array : metadata["expected_input_shapes"]) {
        std::vector<int64_t> shape;
        for (const auto& dim : shape_array) {
          shape.push_back(dim.get<int64_t>());
        }
        shapes.push_back(std::move(shape));
      }
    }
  }
  catch (const std::exception& e) {
    // Log warning but don't fail - metadata is optional
    // In production, you might want to use a proper logging framework
  }

  return shapes;
}

// Helper to parse a SpyreCode JSON command and create a JobPlanStep
JobPlanStep ParseSpyreCodeCommand(
    const nlohmann::json& command,
    const flex::CompositeAddress& program_address) {
  if (!command.contains("command") || !command["command"].is_string()) {
    throw std::runtime_error("SpyreCode command missing 'command' field");
  }

  std::string command_type = command["command"].get<std::string>();
  const auto& properties =
      command.contains("properties") ? command["properties"] : nlohmann::json();

  if (command_type == "ComputeOnDevice") {
    // Create RuntimeOperationCompute with the allocated program address
    auto compute_op =
        std::make_shared<flex::RuntimeOperationCompute>(program_address);

    return JobPlanStep(compute_op);

  } else if (command_type == "ComputeOnHost") {
    // Create a host compute step
    // For now, create an empty HostComputeStep as placeholder
    // metadata to be defined by deeptools
    HostComputeStep host_step;
    return JobPlanStep(std::move(host_step));

  } else if (command_type == "DataTransfer") {
    // Extract direction: 0 = H2D, 1 = D2H
    if (!properties.contains("direction")) {
      throw std::runtime_error(
          "DataTransfer command missing 'direction' property");
    }

    int direction = properties["direction"].get<int>();

    if (direction == 0) {
      // Host-to-Device transfer
      // Extract host and device addresses
      if (!properties.contains("dev_ptr")) {
        throw std::runtime_error(
            "DataTransfer H2D missing 'dev_ptr' property");
      }

      std::string dev_ptr_str = properties["dev_ptr"].get<std::string>();

      // TODO: indicate which HostCompute the input is from
      void* host_addr = nullptr;
      uint64_t device_addr = std::stoull(dev_ptr_str);

      // Compute CompositeAddress with offset from device_addr
      flex::CompositeAddress comp_addr = ComputeOffsetAddress(program_address, device_addr);

      auto h2d_op =
          std::make_shared<flex::RuntimeOperationH2D>(host_addr, comp_addr);
      return JobPlanStep(h2d_op);

    } else if (direction == 1) {
      // TODO: check if this is required
      // Device-to-Host transfer
      throw std::runtime_error("Invalid DataTransfer direction: " +
                               std::to_string(direction));

    } else {
      throw std::runtime_error("Invalid DataTransfer direction: " +
                               std::to_string(direction));
    }

  } else {
    throw std::runtime_error("Unknown SpyreCode command type: " + command_type);
  }
}

// Helper to execute Job Preparation Plan
c10::DataPtr ExecuteJobPreparationPlan(
    const nlohmann::json& job_prep_plan,
    const std::filesystem::path& spyrecode_dir) {
  if (!job_prep_plan.is_array()) {
    throw std::runtime_error("JobPreparationPlan must be an array");
  }

  c10::DataPtr allocated_ptr;
  bool found_allocate = false;
  bool found_init_transfer = false;

  for (const auto& command : job_prep_plan) {
    if (!command.contains("command") || !command["command"].is_string()) {
      throw std::runtime_error(
          "JobPreparationPlan command missing 'command' field");
    }

    std::string command_type = command["command"].get<std::string>();
    const auto& properties = command.contains("properties")
                                 ? command["properties"]
                                 : nlohmann::json();

    if (command_type == "Allocate") {
      if (found_allocate) {
        throw std::runtime_error(
            "Multiple Allocate commands found in JobPreparationPlan");
      }

      // Extract size from properties
      if (!properties.contains("size")) {
        throw std::runtime_error("Allocate command missing 'size' property");
      }

      // Parse size (stored as string in JSON)
      std::string size_str = properties["size"].get<std::string>();
      size_t size = std::stoull(size_str);

      // Call SpyreAllocator to allocate memory
      auto& allocator = SpyreAllocator::instance();
      allocated_ptr = allocator.allocate(size);

      found_allocate = true;

    } else if (command_type == "InitTransfer") {
      if (found_init_transfer) {
        throw std::runtime_error(
            "Multiple InitTransfer commands found in JobPreparationPlan");
      }

      if (!found_allocate) {
        throw std::runtime_error(
            "InitTransfer command must come after Allocate command");
      }

      // Extract binary file path from properties
      if (!properties.contains("file_path")) {
        throw std::runtime_error(
            "InitTransfer command missing 'binary_file' property");
      }

      std::string binary_file = properties["file_path"].get<std::string>();
      std::filesystem::path binary_path(binary_file);

      // Load binary from file
      if (!FileExists(binary_path)) {
        throw std::runtime_error("Binary file not found: " +
                                 binary_path.string());
      }

      std::string binary_data = ReadFileToString(binary_path);

      // Extract device address from properties
      if (!properties.contains("dev_ptr")) {
        throw std::runtime_error(
            "InitTransfer command missing 'dev_ptr' property");
      }

      std::string dev_ptr_str = properties["dev_ptr"].get<std::string>();
      uint64_t device_addr = std::stoull(dev_ptr_str);

      // Extract CompositeAddress from allocated_ptr and compute offset
      flex::CompositeAddress program_address = ExtractCompositeAddress(allocated_ptr);
      flex::CompositeAddress comp_addr = ComputeOffsetAddress(program_address, device_addr);

      // TODO: update RuntimeOperationH2D
      // TODO: extend lifetime?
      // Create RuntimeOperationH2D to transfer binary to device
      auto h2d_op = std::make_shared<flex::RuntimeOperationH2D>(
          const_cast<void*>(static_cast<const void*>(binary_data.data())),
          comp_addr);

      // Get the default stream and launch the operation
      auto runtime = getGlobalRuntime();
      if (!runtime) {
        throw std::runtime_error("GlobalRuntime not initialized");
      }
      flex::RuntimeStream* stream = runtime->getDefaultStream(0);
      stream->launchOperation(*h2d_op);

      // Synchronize to ensure transfer completes before returning
      stream->synchronize();

      found_init_transfer = true;

    } else {
      throw std::runtime_error("Unknown JobPreparationPlan command type: " +
                               command_type);
    }
  }

  if (!allocated_ptr) {
    throw std::runtime_error(
        "JobPreparationPlan did not allocate program memory");
  }

  return allocated_ptr;
}

// Helper to translate Job Execution Plan to JobPlan
std::unique_ptr<JobPlan> TranslateJobExecPlan(
    const nlohmann::json& job_exec_plan,
    const flex::CompositeAddress& program_address) {
  if (!job_exec_plan.is_array()) {
    throw std::runtime_error("JobExecPlan must be an array");
  }

  // Parse each command in the JobExecPlan and create JobPlanSteps
  std::vector<JobPlanStep> steps;
  for (const auto& command : job_exec_plan) {
    try {
      steps.push_back(ParseSpyreCodeCommand(command, program_address));
    }
    catch (const std::exception& e) {
      throw std::runtime_error("Failed to parse SpyreCode command: " +
                               std::string(e.what()));
    }
  }

  // Create and return the JobPlan
  return std::make_unique<JobPlan>(std::move(steps));
}

}  // namespace detail

std::unique_ptr<JobPlan> PrepareKernel(
    const std::filesystem::path& spyrecode_dir_path) {
  if (!std::filesystem::exists(spyrecode_dir_path)) {
    throw std::runtime_error("SpyreCode directory does not exist: " +
                             spyrecode_dir_path.string());
  }

  if (!std::filesystem::is_directory(spyrecode_dir_path)) {
    throw std::runtime_error("Path is not a directory: " +
                             spyrecode_dir_path.string());
  }

  auto spyrecode_json_path = spyrecode_dir_path / "spyrecode.json";
  if (!detail::FileExists(spyrecode_json_path)) {
    throw std::runtime_error(
        "Required file spyrecode.json not found in directory: " +
        spyrecode_dir_path.string());
  }

  std::string json_str = detail::ReadFileToString(spyrecode_json_path);
  nlohmann::json spyrecode_json;

  try {
    spyrecode_json = nlohmann::json::parse(json_str);
  }
  catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse spyrecode.json: " +
                             std::string(e.what()));
  }

  if (!spyrecode_json.contains("JobPreparationPlan") ||
      !spyrecode_json["JobPreparationPlan"].is_array()) {
    throw std::runtime_error(
        "SpyreCode JSON missing 'JobPreparationPlan' array");
  }

  c10::DataPtr allocated_memory = detail::ExecuteJobPreparationPlan(
      spyrecode_json["JobPreparationPlan"], spyrecode_dir_path);

  // Extract the CompositeAddress from the allocated DataPtr
  flex::CompositeAddress program_address = detail::ExtractCompositeAddress(allocated_memory);

  if (!spyrecode_json.contains("JobExecPlan") ||
      !spyrecode_json["JobExecPlan"].is_array()) {
    throw std::runtime_error("SpyreCode JSON missing 'JobExecPlan' array");
  }

  auto job_plan = detail::TranslateJobExecPlan(spyrecode_json["JobExecPlan"],
                                               program_address);

  // Store the program memory in JobPlan to keep it alive for the lifetime
  // of the plan. This prevents premature deallocation of the device memory
  // that contains the compiled program.
  job_plan->setProgramMemory(std::move(allocated_memory));

  auto metadata_path = spyrecode_dir_path / "metadata.json";
  auto expected_shapes = detail::ParseExpectedInputShapes(metadata_path);

  if (!expected_shapes.empty()) {
    job_plan->setExpectedInputShapes(std::move(expected_shapes));
  }

  return job_plan;
}

bool ValidateSpyreCodeDir(const std::filesystem::path& spyrecode_dir_path,
                          bool strict) {
  // Check if directory exists
  if (!std::filesystem::exists(spyrecode_dir_path)) {
    return false;
  }

  if (!std::filesystem::is_directory(spyrecode_dir_path)) {
    return false;
  }

  // Check for required spyrecode.json file
  auto spyrecode_json_path = spyrecode_dir_path / "spyrecode.json";
  if (!detail::FileExists(spyrecode_json_path)) {
    return false;
  }

  // If strict validation is enabled, check file size
  if (strict) {
    auto file_size = std::filesystem::file_size(spyrecode_json_path);
    if (file_size == 0) {
      return false;
    }
  }

  return true;
}

}  // namespace spyre
