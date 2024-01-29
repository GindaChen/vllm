#pragma once

#include <torch/extension.h>

#include <map>
#include <vector>

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping);

void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping);

void reshape_and_cache(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);

void gather_cached_kv(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);

std::vector<int64_t> get_ipc_mem_handle(torch::Tensor tensor);

bool register_ipc_mem_handle(
	std::vector<int64_t> k_cache_handle_vec,
	std::vector<int64_t> v_cache_handle_vec,
    int64_t layer_id,
	int64_t num_layers,
	int64_t num_heads,
	const std::vector<int64_t> &context_parallel_config,
	const std::vector<int64_t> &decoding_parallel_config
);

void migrate_blocks(
	const int64_t context_pp_size,
	const int64_t context_tp_size,

	const std::vector<int64_t> &context_block_indexes,

	const int64_t decoding_pp_size,
	const int64_t decoding_tp_size,

	const int64_t decoding_pp_rank,
	const int64_t decoding_tp_rank,

	const std::vector<int64_t> &decoding_block_indexes,

	std::vector<torch::Tensor>& decoding_worker_k_caches,
	std::vector<torch::Tensor>& decoding_worker_v_caches
);

void migrate_blocks__block_contiguous(
	const int64_t context_pp_size,
	const int64_t context_tp_size,

	const std::vector<int64_t> &context_block_indexes,

	const int64_t decoding_pp_size,
	const int64_t decoding_tp_size,

	const int64_t decoding_pp_rank,
	const int64_t decoding_tp_rank,

	const std::vector<int64_t> &decoding_block_indexes,

	torch::Tensor decoding_worker_k_cache,
	torch::Tensor decoding_worker_v_cache
);