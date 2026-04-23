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

#include "job_plan_step.h"

#include <stdexcept>
#include <utility>

#include "flex/runtime_stream/runtime_operation.hpp"

namespace spyre {

JobPlanStep::JobPlanStep() : operation_(nullptr), host_compute_(std::nullopt) {}

JobPlanStep::JobPlanStep(std::shared_ptr<flex::RuntimeOperation> operation)
    : operation_(std::move(operation)), host_compute_(std::nullopt) {}

JobPlanStep::JobPlanStep(const HostComputeStep& host_compute)
    : operation_(nullptr), host_compute_(host_compute) {}

JobPlanStep::JobPlanStep(HostComputeStep&& host_compute)
    : operation_(nullptr), host_compute_(std::move(host_compute)) {}

JobPlanStep::JobPlanStep(const JobPlanStep& other)
    : operation_(other.operation_), host_compute_(other.host_compute_) {}

JobPlanStep::JobPlanStep(JobPlanStep&& other) noexcept
    : operation_(std::move(other.operation_)),
      host_compute_(std::move(other.host_compute_)) {}

JobPlanStep& JobPlanStep::operator=(const JobPlanStep& other) {
  if (this != &other) {
    operation_ = other.operation_;
    host_compute_ = other.host_compute_;
  }
  return *this;
}

JobPlanStep& JobPlanStep::operator=(JobPlanStep&& other) noexcept {
  if (this != &other) {
    operation_ = std::move(other.operation_);
    host_compute_ = std::move(other.host_compute_);
  }
  return *this;
}

JobPlanStep::~JobPlanStep() = default;

std::shared_ptr<flex::RuntimeOperation> JobPlanStep::getOperation() const {
  return operation_;
}

void JobPlanStep::setOperation(
    std::shared_ptr<flex::RuntimeOperation> operation) {
  operation_ = std::move(operation);
  host_compute_ = std::nullopt;
}

const std::optional<HostComputeStep>& JobPlanStep::getHostCompute() const {
  return host_compute_;
}

void JobPlanStep::setHostCompute(const HostComputeStep& host_compute) {
  operation_ = nullptr;
  host_compute_ = host_compute;
}

void JobPlanStep::setHostCompute(HostComputeStep&& host_compute) {
  operation_ = nullptr;
  host_compute_ = std::move(host_compute);
}

void JobPlanStep::clearHostCompute() {
  host_compute_ = std::nullopt;
}

bool JobPlanStep::hasOperation() const {
  return operation_ != nullptr;
}

bool JobPlanStep::hasHostCompute() const {
  return host_compute_.has_value();
}

}  // namespace spyre
