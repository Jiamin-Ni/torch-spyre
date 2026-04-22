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

#include "host_compute_step.h"

#include <stdexcept>
#include <utility>

namespace spyre {

HostComputeStep::HostComputeStep()
    : function_(nullptr), metadata_(nullptr), output_buffer_size_(0) {}

HostComputeStep::HostComputeStep(HostComputeFunction function,
                                 std::shared_ptr<HostComputeMetadata> metadata,
                                 size_t output_buffer_size)
    : function_(std::move(function)),
      metadata_(std::move(metadata)),
      output_buffer_size_(output_buffer_size) {}

HostComputeStep::HostComputeStep(const HostComputeStep& other)
    : function_(other.function_),
      metadata_(other.metadata_),
      output_buffer_size_(other.output_buffer_size_) {}

HostComputeStep::HostComputeStep(HostComputeStep&& other) noexcept
    : function_(std::move(other.function_)),
      metadata_(std::move(other.metadata_)),
      output_buffer_size_(other.output_buffer_size_) {
  other.output_buffer_size_ = 0;
}

HostComputeStep& HostComputeStep::operator=(const HostComputeStep& other) {
  if (this != &other) {
    function_ = other.function_;
    metadata_ = other.metadata_;
    output_buffer_size_ = other.output_buffer_size_;
  }
  return *this;
}

HostComputeStep& HostComputeStep::operator=(HostComputeStep&& other) noexcept {
  if (this != &other) {
    function_ = std::move(other.function_);
    metadata_ = std::move(other.metadata_);
    output_buffer_size_ = other.output_buffer_size_;
    other.output_buffer_size_ = 0;
  }
  return *this;
}

HostComputeStep::~HostComputeStep() = default;

const HostComputeFunction& HostComputeStep::getFunction() const {
  return function_;
}

void HostComputeStep::setFunction(HostComputeFunction function) {
  function_ = std::move(function);
}

const HostComputeMetadata& HostComputeStep::getMetadata() const {
  if (!metadata_) {
    throw std::runtime_error(
        "HostComputeStep::getMetadata: No metadata is set");
  }
  return *metadata_;
}

std::shared_ptr<HostComputeMetadata> HostComputeStep::getMetadataPtr() const {
  return metadata_;
}

void HostComputeStep::setMetadata(std::shared_ptr<HostComputeMetadata> metadata) {
  metadata_ = std::move(metadata);
}

size_t HostComputeStep::getOutputBufferSize() const {
  return output_buffer_size_;
}

void HostComputeStep::setOutputBufferSize(size_t size) {
  output_buffer_size_ = size;
}

bool HostComputeStep::hasFunction() const {
  return static_cast<bool>(function_);
}

}  // namespace spyre
