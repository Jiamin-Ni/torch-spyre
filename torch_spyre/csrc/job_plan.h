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

#include <c10/core/Allocator.h>

#include <cstddef>
#include <vector>

#include "job_plan_step.h"

namespace spyre {

/**
 * @brief Represents a complete execution plan for a job
 *
 * A JobPlan defines the top-level execution plan with ordered steps and
 * expected input shapes. It contains a sequence of JobPlanStep objects that
 * will be executed in order, along with metadata about the expected input
 * tensor shapes for each kernel input.
 *
 * Empty plans are supported for DMA-only operations that don't require
 * computation steps.
 */
class JobPlan {
 public:
  /**
   * @brief Default constructor
   *
   * Creates an empty JobPlan with no steps and no expected input shapes.
   */
  JobPlan();

  /**
   * @brief Construct a JobPlan with steps (lvalue overload)
   *
   * @param steps Vector of JobPlanStep objects defining the execution sequence
   */
  explicit JobPlan(const std::vector<JobPlanStep>& steps);

  /**
   * @brief Construct a JobPlan with steps (rvalue overload)
   *
   * @param steps Vector of JobPlanStep objects defining the execution sequence
   */
  explicit JobPlan(std::vector<JobPlanStep>&& steps);

  /**
   * @brief Construct a JobPlan with steps and expected input shapes (lvalue
   * overload)
   *
   * @param steps Vector of JobPlanStep objects defining the execution sequence
   * @param expected_input_shapes Vector of shape vectors, one per kernel input
   */
  JobPlan(const std::vector<JobPlanStep>& steps,
          const std::vector<std::vector<int64_t>>& expected_input_shapes);

  /**
   * @brief Construct a JobPlan with steps and expected input shapes (rvalue
   * overload)
   *
   * @param steps Vector of JobPlanStep objects defining the execution sequence
   * @param expected_input_shapes Vector of shape vectors, one per kernel input
   */
  JobPlan(std::vector<JobPlanStep>&& steps,
          std::vector<std::vector<int64_t>>&& expected_input_shapes);

  /**
   * @brief Copy constructor
   */
  JobPlan(const JobPlan& other);

  /**
   * @brief Move constructor
   */
  JobPlan(JobPlan&& other) noexcept;

  /**
   * @brief Copy assignment operator
   */
  JobPlan& operator=(const JobPlan& other);

  /**
   * @brief Move assignment operator
   */
  JobPlan& operator=(JobPlan&& other) noexcept;

  /**
   * @brief Destructor
   */
  ~JobPlan();

  /**
   * @brief Get the vector of steps
   * @return Const reference to the steps vector
   */
  const std::vector<JobPlanStep>& getSteps() const;

  /**
   * @brief Set the vector of steps (lvalue overload)
   * @param steps Vector of JobPlanStep objects
   */
  void setSteps(const std::vector<JobPlanStep>& steps);

  /**
   * @brief Set the vector of steps (rvalue overload)
   * @param steps Vector of JobPlanStep objects
   */
  void setSteps(std::vector<JobPlanStep>&& steps);

  /**
   * @brief Add a step to the end of the plan
   * @param step The JobPlanStep to add
   */
  void addStep(JobPlanStep step);

  /**
   * @brief Get the expected input shapes
   * @return Const reference to the expected input shapes vector
   */
  const std::vector<std::vector<int64_t>>& getExpectedInputShapes() const;

  /**
   * @brief Set the expected input shapes (lvalue overload)
   * @param shapes Vector of shape vectors, one per kernel input
   */
  void setExpectedInputShapes(const std::vector<std::vector<int64_t>>& shapes);

  /**
   * @brief Set the expected input shapes (rvalue overload)
   * @param shapes Vector of shape vectors, one per kernel input
   */
  void setExpectedInputShapes(std::vector<std::vector<int64_t>>&& shapes);

  /**
   * @brief Get the number of steps in the plan
   * @return Number of steps
   */
  size_t getStepCount() const;

  /**
   * @brief Check if the plan is empty (has no steps)
   * @return true if plan has no steps, false otherwise
   */
  bool isEmpty() const;

  /**
   * @brief Get a step by index
   *
   * @param index Index of the step to retrieve
   * @return Const reference to the JobPlanStep at the given index
   * @throws std::out_of_range if index is out of bounds
   */
  const JobPlanStep& getStep(size_t index) const;

  /**
   * @brief Get a mutable step by index
   *
   * @param index Index of the step to retrieve
   * @return Reference to the JobPlanStep at the given index
   * @throws std::out_of_range if index is out of bounds
   */
  JobPlanStep& getStep(size_t index);

  /**
   * @brief Clear all steps from the plan
   */
  void clearSteps();

  /**
   * @brief Clear the expected input shapes
   */
  void clearExpectedInputShapes();

  /**
   * @brief Iterator support - begin
   * @return Iterator to the first step
   */
  std::vector<JobPlanStep>::iterator begin();

  /**
   * @brief Iterator support - begin (const)
   * @return Const iterator to the first step
   */
  std::vector<JobPlanStep>::const_iterator begin() const;

  /**
   * @brief Iterator support - end
   * @return Iterator to one past the last step
   */
  std::vector<JobPlanStep>::iterator end();

  /**
   * @brief Iterator support - end (const)
   * @return Const iterator to one past the last step
   */
  std::vector<JobPlanStep>::const_iterator end() const;

  /**
   * @brief Iterator support - cbegin
   * @return Const iterator to the first step
   */
  std::vector<JobPlanStep>::const_iterator cbegin() const;

  /**
   * @brief Iterator support - cend
   * @return Const iterator to one past the last step
   */
  std::vector<JobPlanStep>::const_iterator cend() const;

  /**
   * @brief Set the program memory allocation
   *
   * Stores the DataPtr holding the compiled program on device memory.
   * This keeps the program memory alive for the lifetime of the JobPlan.
   *
   * @param program_memory DataPtr holding the compiled program on device
   */
  void setProgramMemory(c10::DataPtr program_memory);

  /**
   * @brief Get the program memory allocation
   * @return Const reference to the program memory DataPtr
   */
  const c10::DataPtr& getProgramMemory() const;

 private:
  std::vector<JobPlanStep> steps_;
  std::vector<std::vector<int64_t>> expected_input_shapes_;
  c10::DataPtr program_memory_;
};

}  // namespace spyre
