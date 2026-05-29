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

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "spyre_allocator.h"
#include "util/processSpyreCodeArtifacts.h"

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
    LaunchContext& ctx) const {
  if (bind_io_addresses_) {
    std::vector<const flex::CompositeAddress*> inp;
    for (auto& tensor : ctx.inputs_outputs) {
      flex::CompositeAddress* address =
          &(static_cast<SharedOwnerCtx*>(
                tensor.storage().data_ptr().get_context())
                ->composite_addr);
      inp.push_back(address);
    }

    auto op = std::make_unique<flex::RuntimeOperationCompute>(
        &binary_address_, inp, "", bootstrap_addr_);
    op->setPipelineBarrier(pipeline_barrier_);
    return op;
  }
  auto op = std::make_unique<flex::RuntimeOperationCompute>(&binary_address_);
  op->setPipelineBarrier(pipeline_barrier_);
  return op;
}

// convert CompositeAddress to address that host compute function expects
int64_t convert_address(flex::CompositeAddress& composite_address) {
  size_t num_chunks = composite_address.chunks().size();
  TORCH_CHECK(num_chunks == 1, "Interleaved not supported yet");

  // TODO(jni): update once resolved on flex support
  // const auto& addr = composite_address.chunks().at(0).addr;
  // int64_t address = addr.segment_id * flex::SEGMENT_SIZE + addr.offset;

  TORCH_CHECK(false,
              "convert_address not yet implemented - waiting for flex support");
  return 0;
}

std::unique_ptr<flex::RuntimeOperation> JobPlanStepHostCompute::construct(
    LaunchContext& ctx) const {
  // fake symbols
  if (ishape_.size() == 1 && ishape_[0] == 0) {
    auto callback = [this](void*) {
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer_, nullptr);
    };
    auto op = std::make_unique<flex::RuntimeOperationHostCallback>(
        pipeline_barrier_, std::move(callback), nullptr);

    return op;
  }

  std::vector<int64_t> addresses(ctx.inputs_outputs.size());
  int addr_idx = 0;
  for (auto& tensor : ctx.inputs_outputs) {
    int64_t addr = convert_address(
        (static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context())
             ->composite_addr));
    addresses[addr_idx++] = addr;
  }
  auto callback = [this, addresses](void*) {
    deeptools::processComputeOnHostCommand(*hcm_, output_buffer_, &addresses);
  };

  auto op = std::make_unique<flex::RuntimeOperationHostCallback>(
      pipeline_barrier_, std::move(callback), nullptr);

  return op;
}

namespace {

// TODO(jni): move to flex
// Helper function to format CompositeAddress for debugging
std::string format_composite_address(const flex::CompositeAddress& addr) {
  std::ostringstream oss;
  const auto& chunks = addr.chunks();
  if (chunks.empty()) {
    oss << "<empty>";
    return oss.str();
  }

  oss << "[";
  for (size_t i = 0; i < chunks.size(); ++i) {
    if (i > 0) oss << ", ";
    const auto& chunk = chunks[i];
    oss << "region=" << chunk.addr.region_id << " off=0x" << std::hex
        << chunk.addr.offset << std::dec << " size=" << chunk.size
        << " domain=" << chunk.domain_id;
  }
  oss << "]";
  return oss.str();
}

}  // namespace

void JobPlanStepH2D::dump(std::ostream& os) const {
  os << "  H2D (Host-to-Device)\n";
  os << "    Host address: " << host_address_ << "\n";
  os << "    Device address: " << format_composite_address(device_address_)
     << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepD2H::dump(std::ostream& os) const {
  os << "  D2H (Device-to-Host)\n";
  os << "    Device address: " << format_composite_address(device_address_)
     << "\n";
  os << "    Host address: " << host_address_ << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepCompute::dump(std::ostream& os) const {
  os << "  Device Compute\n";
  os << "    Binary address: " << format_composite_address(binary_address_)
     << "\n";
  os << "    Bind I/O addresses: " << (bind_io_addresses_ ? "yes" : "no")
     << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepHostCompute::dump(std::ostream& os) const {
  os << "  Host Compute\n";
  os << "    Output buffer: " << output_buffer_ << "\n";
  os << "    HCM metadata: " << (hcm_ ? "present" : "null") << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

std::string JobPlan::toString() const {
  std::ostringstream os;
  os << "============ JobPlan =============\n";
  os << "Total steps: " << steps.size() << "\n";

  // Job allocation
  if (!job_allocation.chunks().empty()) {
    os << "Job allocation: " << format_composite_address(job_allocation)
       << "\n";
  } else {
    os << "Job allocation: <none>\n";
  }

  // Expected input shapes
  if (!expected_input_shapes.empty()) {
    os << "Expected input shapes (" << expected_input_shapes.size()
       << " tensors):\n";
    for (size_t i = 0; i < expected_input_shapes.size(); ++i) {
      os << "  Input " << i << ": [";
      for (size_t j = 0; j < expected_input_shapes[i].size(); ++j) {
        if (j > 0) os << ", ";
        os << expected_input_shapes[i][j];
      }
      os << "]\n";
    }
  }

  // Pinned buffers
  os << "Pinned buffers: " << pinned_buffers.size() << "\n";
  for (size_t i = 0; i < pinned_buffers.size(); ++i) {
    const auto& buf = pinned_buffers[i];
    os << "  Buffer " << i << ": ptr=" << buf.data_ptr()
       << ", size=" << buf.nbytes() << " bytes\n";
    os << buf << std::endl;
  }

  // Detailed step information
  os << "\nDetailed Steps:\n";
  for (size_t i = 0; i < steps.size(); ++i) {
    os << "Step " << i << ": ";
    steps[i]->dump(os);
  }

  os << "==================================\n";
  return os.str();
}

void JobPlan::dump(std::ostream& os) const {
  os << toString();
}

}  // namespace spyre
