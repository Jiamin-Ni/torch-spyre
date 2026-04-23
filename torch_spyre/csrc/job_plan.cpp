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

#include <stdexcept>
#include <utility>

namespace spyre {

JobPlan::JobPlan() : steps_(), expected_input_shapes_(), program_memory_() {}

JobPlan::JobPlan(const std::vector<JobPlanStep>& steps)
    : steps_(steps), expected_input_shapes_(), program_memory_() {}

JobPlan::JobPlan(std::vector<JobPlanStep>&& steps)
    : steps_(std::move(steps)), expected_input_shapes_(), program_memory_() {}

JobPlan::JobPlan(const std::vector<JobPlanStep>& steps,
                 const std::vector<std::vector<int64_t>>& expected_input_shapes)
    : steps_(steps),
      expected_input_shapes_(expected_input_shapes),
      program_memory_() {}

JobPlan::JobPlan(std::vector<JobPlanStep>&& steps,
                 std::vector<std::vector<int64_t>>&& expected_input_shapes)
    : steps_(std::move(steps)),
      expected_input_shapes_(std::move(expected_input_shapes)),
      program_memory_() {}

JobPlan::JobPlan(const JobPlan& other)
    : steps_(other.steps_),
      expected_input_shapes_(other.expected_input_shapes_),
      program_memory_() {
  // Note: program_memory_ is not copied as c10::DataPtr is move-only
  // Each JobPlan instance should manage its own program memory
}

JobPlan::JobPlan(JobPlan&& other) noexcept
    : steps_(std::move(other.steps_)),
      expected_input_shapes_(std::move(other.expected_input_shapes_)),
      program_memory_(std::move(other.program_memory_)) {}

JobPlan& JobPlan::operator=(const JobPlan& other) {
  if (this != &other) {
    steps_ = other.steps_;
    expected_input_shapes_ = other.expected_input_shapes_;
    // Note: program_memory_ is not copied as c10::DataPtr is move-only
    // The existing program_memory_ is retained
  }
  return *this;
}

JobPlan& JobPlan::operator=(JobPlan&& other) noexcept {
  if (this != &other) {
    steps_ = std::move(other.steps_);
    expected_input_shapes_ = std::move(other.expected_input_shapes_);
    program_memory_ = std::move(other.program_memory_);
  }
  return *this;
}

JobPlan::~JobPlan() = default;

const std::vector<JobPlanStep>& JobPlan::getSteps() const {
  return steps_;
}

void JobPlan::setSteps(const std::vector<JobPlanStep>& steps) {
  steps_ = steps;
}

void JobPlan::setSteps(std::vector<JobPlanStep>&& steps) {
  steps_ = std::move(steps);
}

void JobPlan::addStep(JobPlanStep step) {
  steps_.push_back(std::move(step));
}

const std::vector<std::vector<int64_t>>& JobPlan::getExpectedInputShapes()
    const {
  return expected_input_shapes_;
}

void JobPlan::setExpectedInputShapes(
    const std::vector<std::vector<int64_t>>& shapes) {
  expected_input_shapes_ = shapes;
}

void JobPlan::setExpectedInputShapes(
    std::vector<std::vector<int64_t>>&& shapes) {
  expected_input_shapes_ = std::move(shapes);
}

size_t JobPlan::getStepCount() const {
  return steps_.size();
}

bool JobPlan::isEmpty() const {
  return steps_.empty();
}

const JobPlanStep& JobPlan::getStep(size_t index) const {
  if (index >= steps_.size()) {
    throw std::out_of_range("JobPlan::getStep: index out of range");
  }
  return steps_[index];
}

JobPlanStep& JobPlan::getStep(size_t index) {
  if (index >= steps_.size()) {
    throw std::out_of_range("JobPlan::getStep: index out of range");
  }
  return steps_[index];
}

void JobPlan::clearSteps() {
  steps_.clear();
}

void JobPlan::clearExpectedInputShapes() {
  expected_input_shapes_.clear();
}

std::vector<JobPlanStep>::iterator JobPlan::begin() {
  return steps_.begin();
}

std::vector<JobPlanStep>::const_iterator JobPlan::begin() const {
  return steps_.begin();
}

std::vector<JobPlanStep>::iterator JobPlan::end() {
  return steps_.end();
}

std::vector<JobPlanStep>::const_iterator JobPlan::end() const {
  return steps_.end();
}

std::vector<JobPlanStep>::const_iterator JobPlan::cbegin() const {
  return steps_.cbegin();
}

std::vector<JobPlanStep>::const_iterator JobPlan::cend() const {
  return steps_.cend();
}

void JobPlan::setProgramMemory(c10::DataPtr program_memory) {
  program_memory_ = std::move(program_memory);
}

const c10::DataPtr& JobPlan::getProgramMemory() const {
  return program_memory_;
}

}  // namespace spyre
