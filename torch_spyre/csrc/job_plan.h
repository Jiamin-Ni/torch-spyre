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

#include <torch/types.h>

#include <cstdint>
#include <flex/flex.hpp>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "util/spyrecode.h"

// Forward declarations for flex types
namespace flex {
class RuntimeOperation;
}

namespace spyre {

/**
 * @brief Context passed to JobPlanStep::construct() at launch time
 *
 * Carries runtime data available at LaunchKernel time that was not available
 * during PrepareKernel.
 */
struct LaunchContext {
  /**
   * @brief at::Tensor list of inputs and outputs
   *
   */
  const std::vector<at::Tensor>& inputs_outputs;
};

/**
 * @brief Polymorphic base class for JobPlan steps
 *
 * Each concrete subclass holds metadata resolved during PrepareKernel and
 * implements construct() to produce a RuntimeOperation at LaunchKernel time.
 * This factory method pattern eliminates special-case branching in
 * SpyreStream::Launch.
 *
 * All RuntimeOperation objects are transient: constructed inside construct(),
 * ownership transferred to RuntimeStream via launchOperation(), and destroyed
 * when the stream completes the operation. No RuntimeOperation is cached in
 * the JobPlan.
 */
class JobPlanStep {
 public:
  virtual ~JobPlanStep() = default;

  /**
   * @brief Construct a RuntimeOperation for this step
   *
   * Called by SpyreStream during LaunchKernel. Produces a fully-populated
   * RuntimeOperation using metadata stored during PrepareKernel and runtime
   * data from the LaunchContext.
   *
   * @param ctx Launch context containing composite addresses
   * @return Unique pointer to the constructed RuntimeOperation
   */
  virtual std::unique_ptr<flex::RuntimeOperation> construct(
      LaunchContext& ctx) const = 0;

  /**
   * @brief Enable or disable pipeline barrier for this step
   *
   * Pipeline barriers control operation ordering within a stream. When enabled,
   * the operation waits for all prior operations to complete before starting.
   *
   * @param enable True to enable pipeline barrier, false to disable
   */
  void setPipelineBarrier(bool enable) {
    pipeline_barrier_ = enable;
  }

  /**
   * @brief Get the pipeline barrier setting for this step
   *
   * @return True if pipeline barrier is enabled, false otherwise
   */
  bool getPipelineBarrier() const {
    return pipeline_barrier_;
  }

 protected:
  bool pipeline_barrier_ = false;
};

/**
 * @brief Host-to-device transfer step
 *
 * All fields resolved during PrepareKernel. construct() produces a
 * RuntimeOperationH2D.
 *
 * When used for correction tensor DMA, the host_address points into a pinned
 * host buffer allocated during PrepareKernel and shared with the
 * JobPlanStepHostCompute that writes into it. The buffer is allocated once and
 * reused across launches — FIFO ordering within a stream guarantees the
 * HostCompute callback writes the buffer before the H2D reads it.
 */
class JobPlanStepH2D final : public JobPlanStep {
 public:
  /**
   * @brief Construct H2D step with raw host pointer
   *
   * @param host_address Host memory address (lifetime managed by JobPlan)
   * @param device_address Device memory address
   */
  JobPlanStepH2D(void* host_address, flex::CompositeAddress device_address)
      : host_address_(host_address),
        device_address_(std::move(device_address)) {}

  std::unique_ptr<flex::RuntimeOperation> construct(
      LaunchContext& ctx) const override;

 private:
  void* host_address_;  // Non-owning pointer (JobPlan owns the buffer)
  flex::CompositeAddress device_address_;
};

/**
 * @brief Device-to-host transfer step
 *
 * All fields resolved during PrepareKernel. construct() produces a
 * RuntimeOperationD2H.
 */
class JobPlanStepD2H final : public JobPlanStep {
 public:
  /**
   * @brief Construct D2H step
   *
   * @param device_address Device memory address
   * @param host_address Host memory address (caller manages lifetime)
   */
  JobPlanStepD2H(flex::CompositeAddress device_address, void* host_address)
      : device_address_(std::move(device_address)),
        host_address_(host_address) {}

  std::unique_ptr<flex::RuntimeOperation> construct(
      LaunchContext& ctx) const override;

 private:
  flex::CompositeAddress device_address_;
  void* host_address_;
};

/**
 * @brief Device compute launch step
 *
 * All fields resolved during PrepareKernel. construct() produces a
 * RuntimeOperationCompute.
 */
class JobPlanStepCompute final : public JobPlanStep {
 public:
  /**
   * @brief Construct compute step
   *
   * @param binary_address Address of the program binary on device
   * @param bind_io_addresses Whether to bind the compute operation
   * with inputs and outputs addresses
   */
  explicit JobPlanStepCompute(flex::CompositeAddress binary_address,
                              bool bind_io_addresses)
      : binary_address_(std::move(binary_address)),
        bind_io_addresses_(bind_io_addresses) {}

  std::unique_ptr<flex::RuntimeOperation> construct(
      LaunchContext& ctx) const override;

 private:
  flex::CompositeAddress binary_address_;
  bool bind_io_addresses_;
};

/**
 * @brief Host-side computation step (e.g., program correction)
 *
 * Stores compiler metadata (Hcm) and a shared output buffer during
 * PrepareKernel. The host computation uses
 * deeptools::processComputeOnHostCommand which takes Hcm metadata and performs
 * program correction or other host-side operations.
 *
 * The output buffer is a pointer to pinned host memory, shared
 * with the subsequent JobPlanStepH2D that transfers it to device. construct()
 * builds a closure capturing the metadata, composite addresses, and
 * the buffer, and produces a RuntimeOperationHostCallback.
 *
 * The shared buffer is allocated once during PrepareKernel and reused across
 * launches. For tiled execution, the same buffer is reused across iterations —
 * FIFO ordering guarantees each iteration's H2D consumes the buffer before the
 * next iteration's HostCompute overwrites it.
 */
class JobPlanStepHostCompute final : public JobPlanStep {
 public:
  /**
   * @brief Construct host compute step
   *
   * @param hcm Compiler-provided metadata from deeptools (contains vdci and
   *            senConstants describing how symbolic values must be interpreted)
   * @param output_buffer Pinned host buffer (lifetime managed by JobPlan)
   */
  JobPlanStepHostCompute(std::unique_ptr<Hcm> hcm, void* output_buffer)
      : hcm_(std::move(hcm)), output_buffer_(output_buffer) {}

  std::unique_ptr<flex::RuntimeOperation> construct(
      LaunchContext& ctx) const override;

 private:
  std::unique_ptr<Hcm> hcm_;
  void* output_buffer_;  // Non-owning pointer (JobPlan owns the buffer)
};

/**
 * @brief A torch-spyre internal container for executing a unit of work
 *
 * A JobPlan bundles everything needed to execute a unit of work on a stream.
 * It is produced by translating a SpyreCode's Job Execution Plan after the Job
 * Preparation Plan has been executed. flex never sees a JobPlan — SpyreStream
 * extracts the operations and submits them to RuntimeStream.launchOperation()
 * as a vector<RuntimeOperation>.
 *
 * A JobPlan is self-contained: if a compute requires program correction, the
 * correction callback, the correction tensor DMA, and the device compute are
 * all separate steps in the same JobPlan. For pure data movement (e.g., tensor
 * .to(device) or binary loading), a JobPlan with only DMA steps is used.
 *
 * Producers:
 * - Backend compiler (deeptools) via torch-spyre: Deeptools produces a
 *   SpyreCode JSON per SDSC. torch-spyre translates the SpyreCode into a
 *   JobPlan — executing the Job Preparation Plan (allocations, binary loading)
 *   and translating the Job Execution Plan into JobPlanStep entries with
 *   resolved CompositeAddress values. A single torch.compile call may produce
 *   multiple SDSCs, resulting in multiple JobPlans.
 * - Communications libraries: Create JobPlans for inter-device data transfers,
 *   collective operations, or other multi-step communication patterns.
 * - torch-spyre: Assembles JobPlans for tensor .to(device) moves (single
 *   RuntimeOperationH2D step), tensor .to("cpu") readbacks (single
 *   RuntimeOperationD2H step), or any other sequence of operations it needs to
 *   containerize.
 */
struct JobPlan {
  /**
   * @brief Ordered sequence of steps
   *
   * During LaunchKernel, SpyreStream calls construct(ctx) on each step in
   * order, collecting the resulting RuntimeOperations, then submits them to
   * RuntimeStream.
   */
  std::vector<std::unique_ptr<JobPlanStep>> steps;

  /**
   * @brief Owning CompositeAddress of the program binary, and conditionally
   * program correction data and spillover tensor data
   *
   * The JobPlan owns this address and is responsible for its lifetime. When the
   * JobPlan is destroyed, the memory is freed.
   *
   * Set during PrepareKernel when it's loaded to device memory. Empty for pure
   * DMA JobPlans (e.g., tensor .to(device)) that don't involve compute
   * operations.
   */
  flex::CompositeAddress job_allocation;

  /**
   * @brief Compiled tile dimensions from SpyreCode
   *
   * One entry per kernel input tensor. Used by SpyreStream for tiling
   * detection. Empty for pure DMA JobPlans (e.g., tensor .to(device)).
   */
  std::vector<std::vector<int64_t>> expected_input_shapes;

  /**
   * @brief Pinned host buffers owned by this JobPlan
   *
   * Stores pinned memory buffers (e.g., for correction tensors) that must
   * remain alive for the lifetime of the JobPlan. Steps reference these
   * buffers via raw pointers. Buffers are automatically freed when JobPlan
   * is destroyed.
   *
   * Allocated using torch::empty() with .pinned_memory(true) during
   * PrepareKernel. The pinned memory ensures efficient DMA transfers and
   * prevents OS from swapping pages to disk.
   */
  std::vector<torch::Tensor> pinned_buffers;
};

}  // namespace spyre
