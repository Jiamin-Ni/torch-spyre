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
#include <utility>
#include <variant>
#include <vector>

#include "spyre_allocator.h"
#include "spyre_stream.h"
#include "util/processSpyreCodeArtifacts.h"

namespace spyre {

void JobPlanStepH2D::construct(LaunchContext& ctx,
                               const SpyreStream& stream) const {
  // Resolve the host source per launch: ring-backed sources pick this launch's
  // slot from ctx.slot_index
  void* host_address = resolveHostAddress(ctx);
  auto* params =
      flex::createDmaParams(host_address, device_address_.total_size(),
                            /*to_device=*/true, &device_address_);
  params->pipeline_barrier = pipeline_barrier_;
  stream.launchH2D(params);
  flex::destroyDmaParams(params);
}

void JobPlanStepH2D::write(std::ostream& os) const {
  os << "  H2D (Host-to-Device)\n";
  os << "    Host source: pinned buffer ring (" << host_ring_->slots.size()
     << " slot(s), resolved at launch)\n";
  os << "    Device CompositeAddress: " << device_address_ << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepD2H::construct(LaunchContext& ctx,
                               const SpyreStream& stream) const {
  // Resolve the host source per launch: ring-backed sources pick this launch's
  // slot from ctx.slot_index
  void* host_address = resolveHostAddress(ctx);
  if (std::holds_alternative<flex::CompositeAddress>(device_address_)) {
    const auto& device_address =
        std::get<flex::CompositeAddress>(device_address_);
    auto* params =
        flex::createDmaParams(host_address, device_address.total_size(),
                              /*to_device=*/false, &device_address);
    params->pipeline_barrier = pipeline_barrier_;
    stream.launchD2H(params);
    flex::destroyDmaParams(params);
  } else {
    const uint64_t dmva = std::get<Dmva>(device_address_).value;
    auto segment_id = flex::dmvaToSegmentId(dmva);
    TORCH_CHECK(segment_id < ctx.inputs_outputs.size(),
                "D2H tensor-segment lookup out of range: segment ", segment_id,
                " but only ", ctx.inputs_outputs.size(),
                " launch args were provided");
    const auto& tensor = ctx.inputs_outputs.at(segment_id);
    const auto& tensor_address =
        static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context())
            ->composite_addr;
    TORCH_CHECK(tensor_address.chunks().size() == 1,
                "Tensor address must have 1 chunk");
    const auto& base_chunk = tensor_address.chunks()[0];
    uint64_t segment_offset = dmva - (segment_id << flex::SEGMENT_SIZE_BITS);
    TORCH_CHECK(segment_offset + size_ <= tensor_address.total_size(),
                "D2H transfer out of bounds: offset ", segment_offset,
                " + size ", size_, " exceeds tensor allocation size ",
                tensor_address.total_size());
    flex::LogicalAddress offset_addr(base_chunk.addr.region_id,
                                     base_chunk.addr.offset + segment_offset);
    flex::Chunk offset_chunk(offset_addr, size_, base_chunk.domain_id);

    // Create shared_ptr to manage lifetime - will be kept alive by callback
    auto device_address =
        std::make_shared<flex::CompositeAddress>(offset_chunk);

    auto* params =
        flex::createDmaParams(host_address, device_address->total_size(),
                              /*to_device=*/false, device_address.get());
    params->pipeline_barrier = pipeline_barrier_;
    params->callback = [device_address](void*) {};
    stream.launchD2H(params);
    flex::destroyDmaParams(params);
  }
}

void JobPlanStepD2H::write(std::ostream& os) const {
  os << "  D2H (Device-to-Host)\n";
  if (std::holds_alternative<flex::CompositeAddress>(device_address_)) {
    os << "    Device CompositeAddress: "
       << std::get<flex::CompositeAddress>(device_address_) << "\n";
  } else {
    os << "    Device dmva: " << std::get<Dmva>(device_address_).value << "\n";
  }
  os << "    Host dest: pinned buffer ring (" << host_ring_->slots.size()
     << " slot(s), resolved at launch)\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepCompute::construct(LaunchContext& ctx,
                                   const SpyreStream& stream) const {
  std::vector<const flex::CompositeAddress*> tensor_allocs;
  if (bind_io_addresses_) {
    for (auto& tensor : ctx.inputs_outputs) {
      flex::CompositeAddress* address =
          &(static_cast<SharedOwnerCtx*>(
                tensor.storage().data_ptr().get_context())
                ->composite_addr);
      tensor_allocs.push_back(address);
    }
  }
  auto* params = flex::createComputeParams(
      &program_address_, std::move(tensor_allocs), name_, bootstrap_offset_);
  params->pipeline_barrier = pipeline_barrier_;
  stream.launchCompute(params);
  flex::destroyComputeParams(params);
}

void JobPlanStepCompute::write(std::ostream& os) const {
  os << "  Device Compute\n";
  os << "    Name: " << (name_.empty() ? "(unnamed)" : name_) << "\n";
  os << "    Program CompositeAddress: " << program_address_ << "\n";
  os << "    Bind I/O addresses: " << (bind_io_addresses_ ? "yes" : "no")
     << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepHostCompute::construct(LaunchContext& ctx,
                                       const SpyreStream& stream) const {
  // Helper lambda to build HostCallbackParams and launch on the stream.
  // flex::RuntimeStream::launchOperationHostCallback() invokes the callback
  // synchronously in the calling thread, so exceptions propagate directly
  // through launchHostCallback() to the caller
  auto launch_host_callback = [this, &stream](auto&& callback) {
    auto* params = flex::createHostCallbackParams(
        std::forward<decltype(callback)>(callback), nullptr, pipeline_barrier_);
    // Use a scope-exit guard so params is freed even if launchHostCallback
    // throws (which it does when the synchronous host callback raises).
    struct Guard {
      flex::HostCallbackParams* p;
      ~Guard() {
        flex::destroyHostCallbackParams(p);
      }
    } guard{params};
    stream.launchHostCallback(params);
  };

  void* output_buffer = output_ring_->slotAt(ctx.slot_index).data();
  // Case 1: input_ring_ is provided
  if (input_ring_ != nullptr) {
    const void* input_buffer = input_ring_->slotAt(ctx.slot_index).data();
    launch_host_callback([this, output_buffer, input_buffer](void*) {
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer,
                                             input_buffer);
    });
    return;
  }

  // Case 2: fake symbols (ishape_ is {0})
  // Further discussion is required on "ishape". For now, it's vector<int64_t>,
  // and it's {0}, it's for fake symbols
  if (ishape_.size() == 1 && ishape_[0] == 0) {
    launch_host_callback([this, output_buffer](void*) {
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer, nullptr);
    });
    return;
  }

  // Case 3: extract addresses from context tensors
  std::vector<int64_t> addresses(ctx.inputs_outputs.size());
  int addr_idx = 0;
  auto& allocator = SpyreAllocator::instance();
  for (auto& tensor : ctx.inputs_outputs) {
    int64_t addr = allocator.compositeAddressToDmva(
        (static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context())
             ->composite_addr));
    addresses[addr_idx++] = addr;
  }

  launch_host_callback([this, output_buffer, addresses](void*) {
    deeptools::processComputeOnHostCommand(*hcm_, output_buffer, &addresses);
  });
}

void JobPlanStepHostCompute::write(std::ostream& os) const {
  os << "  Host Compute\n";
  os << "    Output ring: " << output_ring_->slots.size()
     << " slot(s) (resolved at launch)\n";
  os << "    HCM metadata: " << (hcm_ ? "present" : "null") << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

std::ostream& operator<<(std::ostream& os, const JobPlan& plan) {
  os << "============ JobPlan =============\n";
  os << "Total steps: " << plan.steps.size() << "\n";

  // Job allocation
  size_t addr_idx = 0;
  for (const auto& addr : plan.job_allocation) {
    if (addr_idx == 0) {
      os << "Job allocation: " << addr << "\n";
    } else {
      os << "Program " << addr_idx - 1 << ": " << addr << "\n";
    }
    ++addr_idx;
  }

  // Expected input shapes
  if (!plan.expected_input_shapes.empty()) {
    os << "Expected input shapes (" << plan.expected_input_shapes.size()
       << " tensors):\n";
    for (size_t i = 0; i < plan.expected_input_shapes.size(); ++i) {
      os << "  Input " << i << ": [";
      for (size_t j = 0; j < plan.expected_input_shapes[i].size(); ++j) {
        if (j > 0) os << ", ";
        os << plan.expected_input_shapes[i][j];
      }
      os << "]\n";
    }
  }

  // Pinned buffer rings (keyed by handle)
  os << "Pinned buffer rings: " << plan.pinned_buffers.size() << "\n";
  for (const auto& [handle, ring] : plan.pinned_buffers) {
    os << "  Handle '" << handle << "': " << ring.slots.size() << " slot(s)\n";
    for (size_t s = 0; s < ring.slots.size(); ++s) {
      const auto& buf = ring.slots[s];
      os << "    Slot " << s << ": ptr=" << buf.data()
         << ", size=" << buf.size() << " bytes\n";
    }
  }

  // Detailed step information
  os << "\nDetailed Steps:\n";
  for (size_t i = 0; i < plan.steps.size(); ++i) {
    os << "Step " << i << ": ";
    os << *plan.steps[i];
  }

  os << "==================================\n";
  return os;
}

}  // namespace spyre
