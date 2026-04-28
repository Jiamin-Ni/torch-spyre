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

#include "job_plan.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "flex/runtime_stream/runtime_operation.hpp"

namespace spyre {

std::unique_ptr<flex::RuntimeOperation> JobPlanStepH2D::construct(
    LaunchContext&) const {
  auto op = std::make_unique<flex::RuntimeOperationH2D>(host_address_,
                                                        &device_address_);
  op->setPipelineBarrier(pipeline_barrier_);
  return op;
}

std::unique_ptr<flex::RuntimeOperation> JobPlanStepD2H::construct(
    LaunchContext&) const {
  auto op = std::make_unique<flex::RuntimeOperationD2H>(&device_address_,
                                                        host_address_);
  op->setPipelineBarrier(pipeline_barrier_);
  return op;
}

std::unique_ptr<flex::RuntimeOperation> JobPlanStepCompute::construct(
    LaunchContext&) const {
  auto op = std::make_unique<flex::RuntimeOperationCompute>(&binary_address_);
  op->setPipelineBarrier(pipeline_barrier_);
  return op;
}

std::unique_ptr<flex::RuntimeOperation> JobPlanStepHostCompute::construct(
    LaunchContext& ctx) const {
  auto callback = [this, composite_addresses = ctx.composite_addresses](void*) {
    function_(metadata_, composite_addresses.data(), output_buffer_);
  };

  auto op = std::make_unique<flex::RuntimeOperationHostCallback>(
      pipeline_barrier_, std::move(callback), nullptr);

  return op;
}

std::unique_ptr<flex::RuntimeOperation> JobPlanStepComputeSpecialize::construct(
    LaunchContext& ctx) const {
  // TODO(jni): to be added once flex PR merged
  throw std::runtime_error(
      "JobPlanStepComputeSpecialize::construct is not implemented: "
      "flex runtime does not provide RuntimeOperationComputeSpecializeResident "
      "in the current API");
}

}  // namespace spyre
