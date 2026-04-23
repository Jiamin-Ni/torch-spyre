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

#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <vector>

#include "common/json.hpp"
#include "job_plan.h"

// Forward declarations for flex types
namespace flex {
class CompositeAddress;
}

namespace spyre {

// Forward declarations
class JobPlanStep;

namespace detail {

/**
 * @brief Helper to check if a file exists
 */
bool FileExists(const std::filesystem::path& path);

/**
 * @brief Helper to read entire file into string
 */
std::string ReadFileToString(const std::filesystem::path& path);

/**
 * @brief Helper to parse metadata JSON if present
 */
std::vector<std::vector<int64_t>> ParseExpectedInputShapes(
    const std::filesystem::path& metadata_path);

/**
 * @brief Helper to parse a SpyreCode JSON command and create a JobPlanStep
 * @param command The JSON command to parse
 * @param program_address The composite address allocated for the program (used
 * for ComputeOnDevice)
 */
JobPlanStep ParseSpyreCodeCommand(
    const nlohmann::json& command,
    const flex::CompositeAddress& program_address);

/**
 * @brief Helper to execute Job Preparation Plan
 * @return Allocated DataPtr for program memory ownership
 */
c10::DataPtr ExecuteJobPreparationPlan(
    const nlohmann::json& job_prep_plan,
    const std::filesystem::path& spyrecode_dir);

/**
 * @brief Helper to translate Job Execution Plan to JobPlan
 * @param job_exec_plan The JSON job execution plan
 * @param program_address The composite address allocated for the program during
 * preparation
 */
std::unique_ptr<JobPlan> TranslateJobExecPlan(
    const nlohmann::json& job_exec_plan,
    const flex::CompositeAddress& program_address);

}  // namespace detail

/**
 * @brief Validate that a directory contains required SpyreCode files
 *
 * @param spyrecode_dir_path Path to the SpyreCode directory
 * @param strict If true, performs additional validation (e.g., file size
 * checks)
 * @return true if directory is valid, false otherwise
 */
bool ValidateSpyreCodeDir(const std::filesystem::path& spyrecode_dir_path,
                          bool strict = false);

/**
 * @brief Prepare a kernel from a SpyreCode directory
 *
 * Loads and validates SpyreCode artifacts, executes the job preparation plan,
 * translates the execution plan into a JobPlan, and optionally loads expected
 * input shapes from metadata.
 *
 * @param spyrecode_dir_path Path to the SpyreCode directory
 * @return Prepared JobPlan
 */
std::unique_ptr<JobPlan> PrepareKernel(
    const std::filesystem::path& spyrecode_dir_path);

}  // namespace spyre
