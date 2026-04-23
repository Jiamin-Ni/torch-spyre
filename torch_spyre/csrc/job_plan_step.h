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

#include <memory>
#include <optional>

#include "host_compute_step.h"

// Forward declaration of flex::RuntimeOperation
namespace flex {
class RuntimeOperation;
class RuntimeOperationHostCallback;
}  // namespace flex

namespace spyre {

/**
 * @brief Represents a single step in a job plan
 *
 * A JobPlanStep pairs a runtime operation with optional host-side computation.
 * When host compute is present, the operation must be a
 * RuntimeOperationHostCallback (or null, to be constructed at launch time).
 *
 * This type is used to build job plans that interleave device operations
 * (compute, data transfers) with host-side operations (program correction,
 * collectives, etc.).
 */
class JobPlanStep {
 public:
  /**
   * @brief Default constructor
   *
   * Creates an empty JobPlanStep with no operation and no host compute.
   */
  JobPlanStep();

  /**
   * @brief Construct a JobPlanStep with only a runtime operation
   *
   * @param operation Shared pointer to the runtime operation
   */
  explicit JobPlanStep(std::shared_ptr<flex::RuntimeOperation> operation);

  /**
   * @brief Construct a JobPlanStep with only host compute (lvalue overload)
   *
   * Creates a host compute step with no runtime operation.
   * A JobPlanStep is either a runtime operation OR a host compute step,
   * never both.
   *
   * @param host_compute Host computation step to execute
   */
  explicit JobPlanStep(const HostComputeStep& host_compute);

  /**
   * @brief Construct a JobPlanStep with only host compute (rvalue overload)
   *
   * Creates a host compute step with no runtime operation.
   * A JobPlanStep is either a runtime operation OR a host compute step,
   * never both.
   *
   * @param host_compute Host computation step to execute
   */
  explicit JobPlanStep(HostComputeStep&& host_compute);

  /**
   * @brief Copy constructor
   */
  JobPlanStep(const JobPlanStep& other);

  /**
   * @brief Move constructor
   */
  JobPlanStep(JobPlanStep&& other) noexcept;

  /**
   * @brief Copy assignment operator
   */
  JobPlanStep& operator=(const JobPlanStep& other);

  /**
   * @brief Move assignment operator
   */
  JobPlanStep& operator=(JobPlanStep&& other) noexcept;

  /**
   * @brief Destructor
   */
  ~JobPlanStep();

  /**
   * @brief Get the runtime operation
   * @return Shared pointer to the runtime operation (may be nullptr)
   */
  std::shared_ptr<flex::RuntimeOperation> getOperation() const;

  /**
   * @brief Set the runtime operation
   *
   * Setting an operation clears any existing host compute step, as a
   * JobPlanStep is either a runtime operation OR a host compute step.
   *
   * @param operation Shared pointer to the runtime operation
   */
  void setOperation(std::shared_ptr<flex::RuntimeOperation> operation);

  /**
   * @brief Get the host compute step
   * @return Optional containing the host compute step if present
   */
  const std::optional<HostComputeStep>& getHostCompute() const;

  /**
   * @brief Set the host compute step (lvalue overload)
   *
   * Setting host compute clears any existing runtime operation, as a
   * JobPlanStep is either a runtime operation OR a host compute step.
   *
   * @param host_compute Host computation step to execute
   */
  void setHostCompute(const HostComputeStep& host_compute);

  /**
   * @brief Set the host compute step (rvalue overload)
   *
   * Setting host compute clears any existing runtime operation, as a
   * JobPlanStep is either a runtime operation OR a host compute step.
   *
   * @param host_compute Host computation step to execute
   */
  void setHostCompute(HostComputeStep&& host_compute);

  /**
   * @brief Clear the host compute step
   *
   * Removes the host compute step from this JobPlanStep.
   */
  void clearHostCompute();

  /**
   * @brief Check if this step has a runtime operation
   * @return true if operation is set, false otherwise
   */
  bool hasOperation() const;

  /**
   * @brief Check if this step has host compute
   * @return true if host compute is present, false otherwise
   */
  bool hasHostCompute() const;

 private:
  std::shared_ptr<flex::RuntimeOperation> operation_;
  std::optional<HostComputeStep> host_compute_;
};

}  // namespace spyre
