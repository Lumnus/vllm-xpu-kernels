// SPDX-License-Identifier: Apache-2.0
//
// Mailbox all-reduce for tensor parallelism on 2 or 4 Xe2 cards over PCIe
// peer-to-peer: a standalone op, one launch per call, in place on a 16-bit
// tensor. See mailbox_all_reduce.cpp for the wire format, the summation
// contract and the failure behaviour.
#pragma once

#include <torch/all.h>

#include <cstdint>
#include <vector>

namespace vllm::mailbox_ar {

// x (bf16 or fp16, contiguous) is replaced by the sum over the ranks.
// inbox[s]: this rank's mailbox for stage s (written by the stage-s partner);
// outbox[s]: the stage-s partner's mailbox, mapped into this process (Level
// Zero IPC). Each mailbox holds two slots of slot_bytes. state: 2 device
// words (epoch, done counter), zeroed before the first call. host: the host
// words from mailbox_ar_host_alloc. timeout_cycles: sub-group clock cycles a
// wait may last before the call fails (0: wait forever).
void mailbox_all_reduce(
    torch::Tensor& x,
    std::vector<int64_t> inbox,
    std::vector<int64_t> outbox,
    int64_t state,
    int64_t host,
    int64_t slot_bytes,
    bool fp32_partials,
    int64_t timeout_cycles);

// Mailbox bytes one slot needs for numel elements.
int64_t mailbox_ar_slot_bytes(int64_t numel, bool fp32_partials);

// Host words the kernel reports a failed wait into (host USM, readable by the
// host at any time without a device sync).
int64_t mailbox_ar_host_alloc();
void mailbox_ar_host_free(int64_t host);
std::vector<int64_t> mailbox_ar_host_read(int64_t host);

}  // namespace vllm::mailbox_ar
