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

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace spyre {

/**
 * @brief Base class for host compute operation metadata
 *
 * This polymorphic base class allows different host compute operations
 * to define their own metadata structures while maintaining type safety
 * and avoiding JSON parsing overhead.
 */
struct HostComputeMetadata {
  virtual ~HostComputeMetadata() = default;
};

/**
 * @brief Function type for host-side computation operations
 *
 * This callable type represents host-side operations such as program
 * correction, collectives, and other host computations that need to be
 * executed as part of a job plan.
 *
 * @param metadata Reference to operation-specific metadata (contains buffer sizes)
 * @param input_buffer Pointer to input buffer containing source data
 * @param output_buffer Pointer to output buffer for results
 */
using HostComputeFunction = std::function<void(
    const HostComputeMetadata& metadata, const void* input_buffer,
    void* output_buffer)>;

/**
 * @brief Represents host-side computation metadata within a JobPlanStep
 *
 * This type encapsulates all information needed to execute host-side
 * computations as part of a job plan, including the function to execute,
 * operation-specific metadata, and output buffer requirements.
 */
class HostComputeStep {
 public:
  /**
   * @brief Default constructor
   *
   * Creates an empty HostComputeStep with no function, empty metadata,
   * and zero output buffer size.
   */
  HostComputeStep();

  /**
   * @brief Construct a HostComputeStep with all parameters
   *
   * @param function The host computation function to execute
   * @param metadata Unique pointer to operation-specific metadata
   * @param output_buffer_size Size of output buffer required (in bytes)
   */
  HostComputeStep(HostComputeFunction function,
                  std::shared_ptr<HostComputeMetadata> metadata,
                  size_t output_buffer_size);

  /**
   * @brief Copy constructor
   */
  HostComputeStep(const HostComputeStep& other);

  /**
   * @brief Move constructor
   */
  HostComputeStep(HostComputeStep&& other) noexcept;

  /**
   * @brief Copy assignment operator
   */
  HostComputeStep& operator=(const HostComputeStep& other);

  /**
   * @brief Move assignment operator
   */
  HostComputeStep& operator=(HostComputeStep&& other) noexcept;

  /**
   * @brief Destructor
   */
  ~HostComputeStep();

  /**
   * @brief Get the host compute function
   * @return Reference to the function callable
   */
  const HostComputeFunction& getFunction() const;

  /**
   * @brief Set the host compute function
   * @param function The function to set
   */
  void setFunction(HostComputeFunction function);

  /**
   * @brief Get the metadata object
   * @return Const reference to the metadata
   */
  const HostComputeMetadata& getMetadata() const;

  /**
   * @brief Get a shared pointer to the metadata object
   * @return Shared pointer to the metadata
   */
  std::shared_ptr<HostComputeMetadata> getMetadataPtr() const;

  /**
   * @brief Set the metadata object
   * @param metadata Shared pointer to the metadata to set
   */
  void setMetadata(std::shared_ptr<HostComputeMetadata> metadata);

  /**
   * @brief Get the output buffer size
   * @return Size of output buffer in bytes
   */
  size_t getOutputBufferSize() const;

  /**
   * @brief Set the output buffer size
   * @param size Size of output buffer in bytes
   */
  void setOutputBufferSize(size_t size);

  /**
   * @brief Check if the HostComputeStep has a valid function
   * @return true if function is set, false otherwise
   */
  bool hasFunction() const;

 private:
  HostComputeFunction function_;
  std::shared_ptr<HostComputeMetadata> metadata_;
  size_t output_buffer_size_;
};

}  // namespace spyre
