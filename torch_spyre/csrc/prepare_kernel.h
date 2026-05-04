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
#include <string>

namespace spyre {

// Forward declarations
class JobPlan;

/**
 * @brief Prepare a kernel from a SpyreCode directory
 *
 * Loads and validates SpyreCode artifacts, executes the job preparation plan,
 * translates the execution plan into a JobPlan, and optionally loads expected
 * input shapes from metadata.
 *
 * @param spyrecode_dir Path to the SpyreCode directory
 * @return Prepared JobPlan
 */
std::unique_ptr<JobPlan> PrepareKernel(const std::string& spyrecode_dir);

}  // namespace spyre
