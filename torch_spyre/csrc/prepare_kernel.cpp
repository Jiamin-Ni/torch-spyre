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

#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "job_plan.h"
#include "module.h"
#include "spyre_allocator.h"

namespace spyre {

namespace detail {

static uint64_t job_allocation_ptr_start = 120259084288;

/**
 * @brief Helper to compute CompositeAddress with offset from device_addr for
 * program
 */
static flex::CompositeAddress ComputeOffsetAddress(
    const flex::CompositeAddress& job_allocation, uint64_t dev_ptr,
    size_t size = 0) {
  // Create CompositeAddress using program_address with offset
  TORCH_CHECK(job_allocation.chunks().size() == 1,
              "job_allocation must have 1 chunk");

  // Calculate offset
  uint64_t offset = dev_ptr - job_allocation_ptr_start;
  if (size == 0) {
    size = job_allocation.total_size() - offset;
  }

  // Get the first chunk and add offset to its address
  const auto& base_chunk = job_allocation.chunks()[0];
  flex::LogicalAddress offset_addr(base_chunk.addr.region_id,
                                   base_chunk.addr.offset + offset);
  flex::Chunk offset_chunk(offset_addr, size, base_chunk.domain_id);
  return flex::CompositeAddress(offset_chunk);
}

/**
 * @brief Helper to check if a file exists
 */
bool FileExists(const std::filesystem::path& path) {
  return std::filesystem::exists(path) &&
         std::filesystem::is_regular_file(path);
}

/**
 * @brief Helper to read entire file into string
 */
std::string ReadFileToString(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  TORCH_CHECK(file, "Failed to open file: ", path.string());

  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

/**
 * @brief Helper to parse a SpyreCode JSON command and create a JobPlanStep
 * @param command The JSON command to parse
 * @param program_address The composite address allocated for the job_allocation
 */
std::unique_ptr<JobPlanStep> ParseSpyreCodeCommand(
    const nlohmann::json& command,
    const flex::CompositeAddress& job_allocation) {
  TORCH_CHECK(command.contains("command") && command["command"].is_string(),
              "SpyreCode command missing 'command' field");

  std::string command_type = command["command"].get<std::string>();
  const auto& properties =
      command.contains("properties") ? command["properties"] : nlohmann::json();

  if (command_type == "ComputeOnDevice") {
    TORCH_CHECK(properties.contains("job_bin_ptr"),
                command_type + " command missing 'job_bin_ptr' property");

    std::string job_bin_ptr_str = properties["job_bin_ptr"].get<std::string>();
    uint64_t job_bin_ptr = std::stoull(job_bin_ptr_str);

    auto job_bin_addr = ComputeOffsetAddress(job_allocation, job_bin_ptr);
    // Create RuntimeOperationCompute with the allocated program address
    auto compute_op =
        std::make_unique<JobPlanStepComputeSpecialize>(std::move(job_bin_addr));

    return compute_op;

  } else if (command_type == "ComputeOnHost") {
    // TODO(jni): create JobPlanStepHostCompute
    TORCH_CHECK(
        false,
        "ComputeOnHost not yet implemented - waiting for deeptools PR to "
        "be merged");
  } else if (command_type == "DataTransfer") {
    // TODO(jni): create JobPlanStepH2D or JobPlanStepD2H
    TORCH_CHECK(
        false,
        "ComputeOnHost not yet implemented - waiting for deeptools PR to "
        "be merged");

    // Extract direction: 0 = H2D, 1 = D2H
    TORCH_CHECK(properties.contains("dirn"),
                "DataTransfer command missing 'dirn' property");

    std::string dirn_str = properties["dirn"].get<std::string>();

    if (dirn_str == "false") {
      // Host-to-Device transfer
      // Extract host and device addresses
      TORCH_CHECK(properties.contains("dev_ptr"),
                  "DataTransfer H2D missing 'dev_ptr' property");

      TORCH_CHECK(properties.contains("size"),
                  "DataTransfer H2D missing 'size' property");

      std::string dev_ptr_str = properties["dev_ptr"].get<std::string>();
      std::string size_str = properties["size"].get<std::string>();

      // TODO(jni): host_handle should contain info about the host buffer to be
      // copied, figure out how and connect host_addr
      TORCH_CHECK(properties.contains("host_handle"),
                  "DataTransfer H2D missing 'host_handle' property");
      std::string host_handle_str =
          properties["host_handle"].get<std::string>();
      void* host_addr = nullptr;
      uint64_t device_ptr = std::stoull(dev_ptr_str);
      size_t transfer_size = std::stoull(size_str);

      // Compute CompositeAddress with offset
      flex::CompositeAddress comp_addr =
          ComputeOffsetAddress(job_allocation, device_ptr, transfer_size);

      auto h2d_op =
          std::make_unique<JobPlanStepH2D>(host_addr, std::move(comp_addr));
      return h2d_op;

    } else if (dirn_str == "true") {
      // Device-to-Host transfer
      // Extract host and device addresses
      TORCH_CHECK(properties.contains("dev_ptr"),
                  "DataTransfer D2H missing 'dev_ptr' property");

      TORCH_CHECK(properties.contains("size"),
                  "DataTransfer D2H missing 'size' property");

      std::string dev_ptr_str = properties["dev_ptr"].get<std::string>();
      std::string size_str = properties["size"].get<std::string>();

      // TODO(jni): host_handle should contain info about the host buffer to be
      // copied to, figure out how and connect host_addr
      TORCH_CHECK(properties.contains("host_handle"),
                  "DataTransfer H2D missing 'host_handle' property");
      std::string host_handle_str =
          properties["host_handle"].get<std::string>();
      void* host_addr = nullptr;
      uint64_t device_ptr = std::stoull(dev_ptr_str);
      size_t transfer_size = std::stoull(size_str);

      // Compute CompositeAddress with offset from device_addr
      flex::CompositeAddress comp_addr =
          ComputeOffsetAddress(job_allocation, device_ptr, transfer_size);

      auto d2h_op =
          std::make_unique<JobPlanStepD2H>(std::move(comp_addr), host_addr);
      return d2h_op;

    } else {
      TORCH_CHECK(false, "Invalid DataTransfer direction: ", dirn_str);
    }

  } else {
    TORCH_CHECK(false, "Unknown SpyreCode command type: ", command_type);
  }
}

/**
 * @brief Helper to execute Job Preparation Plan
 * @return CompositeAddress allocated for the job_allocation during preparation
 */
flex::CompositeAddress ExecuteJobPreparationPlan(
    const nlohmann::json& job_prep_plan,
    const std::filesystem::path& spyrecode_dir) {
  TORCH_CHECK(job_prep_plan.is_array() && job_prep_plan.size() == 2,
              "JobPreparationPlan must be an array with exactly 2 commands "
              "(Allocate and InitTransfer)");

  // Process Allocate command (first item)
  const auto& allocate_cmd = job_prep_plan[0];
  TORCH_CHECK(
      allocate_cmd.contains("command") && allocate_cmd["command"].is_string(),
      "JobPreparationPlan command missing 'command' field");

  std::string allocate_type = allocate_cmd["command"].get<std::string>();
  TORCH_CHECK(allocate_type == "Allocate",
              "First command must be 'Allocate', got: " + allocate_type);

  const auto& allocate_props = allocate_cmd.contains("properties")
                                   ? allocate_cmd["properties"]
                                   : nlohmann::json();

  TORCH_CHECK(allocate_props.contains("size"),
              "Allocate command missing 'size' property");

  std::string size_str = allocate_props["size"].get<std::string>();
  size_t size = std::stoull(size_str);

  auto& allocator = SpyreAllocator::instance();
  c10::DataPtr allocated_ptr = allocator.allocate(size);

  flex::CompositeAddress job_allocation =
      std::move(static_cast<SharedOwnerCtx*>(allocated_ptr.get_context())
                    ->composite_addr);

  // Process InitTransfer command (second item)
  const auto& init_cmd = job_prep_plan[1];
  TORCH_CHECK(init_cmd.contains("command") && init_cmd["command"].is_string(),
              "JobPreparationPlan command missing 'command' field");

  std::string init_type = init_cmd["command"].get<std::string>();
  TORCH_CHECK(init_type == "InitTransfer",
              "Second command must be 'InitTransfer', got: " + init_type);

  const auto& init_props = init_cmd.contains("properties")
                               ? init_cmd["properties"]
                               : nlohmann::json();

  TORCH_CHECK(init_props.contains("file_path"),
              "InitTransfer command missing 'file_path' property");

  std::string binary_file = init_props["file_path"].get<std::string>();
  std::filesystem::path binary_path(binary_file);

  std::string binary_data = ReadFileToString(binary_path);

  TORCH_CHECK(init_props.contains("dev_ptr"),
              "InitTransfer command missing 'dev_ptr' property");

  std::string dev_ptr_str = init_props["dev_ptr"].get<std::string>();
  uint64_t dev_ptr = std::stoull(dev_ptr_str);

  TORCH_CHECK(allocate_props.contains("size"),
              "InitTransfer command missing 'size' property");

  std::string init_size_str = allocate_props["size"].get<std::string>();
  size_t init_size = std::stoull(size_str);

  auto device_addr = ComputeOffsetAddress(job_allocation, dev_ptr, init_size);

  // Create RuntimeOperationH2D to transfer binary to device
  auto h2d_op = flex::RuntimeOperationH2D(
      const_cast<void*>(static_cast<const void*>(binary_data.data())),
      &device_addr);

  // Get the default stream and launch the operation
  // TODO(jni): launch through SpyreStream
  auto runtime = GlobalRuntime::get();
  flex::RuntimeStream* stream = runtime->getDefaultStream();
  stream->launchOperation(h2d_op);

  return job_allocation;
}

/**
 * @brief Helper to translate Job Execution Plan to JobPlan
 * @param job_exec_plan The JSON job execution plan
 * @param job_allocation The composite address allocated for the job_allocation
 * during preparation (moved into JobPlan)
 */
std::unique_ptr<JobPlan> TranslateJobExecPlan(
    const nlohmann::json& job_exec_plan,
    flex::CompositeAddress job_allocation) {
  TORCH_CHECK(job_exec_plan.is_array(), "JobExecPlan must be an array");

  // Parse each command in the JobExecPlan and create JobPlanSteps
  std::vector<std::unique_ptr<JobPlanStep>> steps;
  for (const auto& command : job_exec_plan) {
    try {
      steps.push_back(ParseSpyreCodeCommand(command, job_allocation));
    }
    catch (const std::exception& e) {
      TORCH_CHECK(false, "Failed to parse SpyreCode command: ", e.what());
    }
  }

  // TODO(jni): expected_input_shapes to be added once provided in SpyreCode
  // TODO(jni): pinned buffer to be added as std::map once HostCompute provided
  // in SpyreCode Create and return the JobPlan Use brace initialization to
  // construct JobPlan with moved members
  return std::make_unique<JobPlan>(JobPlan{
      std::move(steps),           // steps
      std::move(job_allocation),  // job_allocation
      {},                         // expected_input_shapes
      {}                          // pinned_buffers
  });
}

}  // namespace detail

std::unique_ptr<JobPlan> PrepareKernel(const std::string& spyrecode_dir) {
  std::filesystem::path spyrecode_dir_path(spyrecode_dir);

  TORCH_CHECK(std::filesystem::exists(spyrecode_dir_path),
              "SpyreCode directory does not exist: ", spyrecode_dir_path);

  TORCH_CHECK(std::filesystem::is_directory(spyrecode_dir_path),
              "Path is not a directory: ", spyrecode_dir_path);

  auto spyrecode_json_path = spyrecode_dir_path / "spyrecode.json";
  TORCH_CHECK(detail::FileExists(spyrecode_json_path),
              "Required file spyrecode.json not found in directory: ",
              spyrecode_dir_path);

  std::string json_str = detail::ReadFileToString(spyrecode_json_path);
  nlohmann::json spyrecode_json;

  try {
    spyrecode_json = nlohmann::json::parse(json_str);
  }
  catch (const std::exception& e) {
    TORCH_CHECK(false, "Failed to parse spyrecode.json: ", e.what());
  }

  TORCH_CHECK(spyrecode_json.contains("JobPreparationPlan") &&
                  spyrecode_json["JobPreparationPlan"].is_array(),
              "SpyreCode JSON missing 'JobPreparationPlan' array");

  flex::CompositeAddress job_allocation = detail::ExecuteJobPreparationPlan(
      spyrecode_json["JobPreparationPlan"], spyrecode_dir_path);

  TORCH_CHECK(spyrecode_json.contains("JobExecPlan") &&
                  spyrecode_json["JobExecPlan"].is_array(),
              "SpyreCode JSON missing 'JobExecPlan' array");

  auto job_plan = detail::TranslateJobExecPlan(spyrecode_json["JobExecPlan"],
                                               std::move(job_allocation));

  return job_plan;
}

}  // namespace spyre
