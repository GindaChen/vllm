#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#include <cstdio>
#include <cuda_runtime_api.h>
#include <cassert>

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    TORCH_CHECK(
      src_device.index() == dst_device.index(),
      "src and dst must be on the same GPU");
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  char *src_ptr = static_cast<char*>(src.data_ptr());
  char *dst_ptr = static_cast<char*>(dst.data_ptr());

  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  const at::cuda::OptionalCUDAGuard device_guard(src_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // NOTE(woosuk): This can be slow if the number of blocks is large.
  for (const auto& pair : block_mapping) {
    int64_t src_block_number = pair.first;
    int64_t dst_block_number = pair.second;
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    cudaMemcpyAsync(
      dst_ptr + dst_offset,
      src_ptr + src_offset,
      block_size_in_bytes,
      memcpy_type,
      stream);
  }
}

namespace vllm {

// Grid: (num_layers, num_pairs)
template<typename scalar_t>
__global__ void copy_blocks_kernel(
  int64_t* key_cache_ptrs,
  int64_t* value_cache_ptrs,
  const int64_t* __restrict__ block_mapping,
  const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

} // namespace vllm

void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping) {
  int num_layers = key_caches.size();
  TORCH_CHECK(num_layers == value_caches.size());
  if (num_layers == 0) {
    return;
  }
  torch::Device cache_device = key_caches[0].device();
  TORCH_CHECK(cache_device.is_cuda());

  // Create data structures for the kernel.
  // Create an array of pointers to the key and value caches.
  int64_t key_cache_ptrs[num_layers];
  int64_t value_cache_ptrs[num_layers];
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    key_cache_ptrs[layer_idx] = reinterpret_cast<int64_t>(key_caches[layer_idx].data_ptr());
    value_cache_ptrs[layer_idx] = reinterpret_cast<int64_t>(value_caches[layer_idx].data_ptr());
  }
  // Create block mapping array.
  std::vector<int64_t> block_mapping_vec;
  for (const auto& pair : block_mapping) {
    int64_t src_block_number = pair.first;
    for (int64_t dst_block_number : pair.second) {
      block_mapping_vec.push_back(src_block_number);
      block_mapping_vec.push_back(dst_block_number);
    }
  }
  int64_t* block_mapping_array = block_mapping_vec.data();
  int num_pairs = block_mapping_vec.size() / 2;

  // Move the data structures to the GPU.
  // NOTE: This synchronizes the CPU and GPU.
  torch::Tensor key_cache_ptrs_tensor = torch::from_blob(
    key_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);
  torch::Tensor value_cache_ptrs_tensor = torch::from_blob(
    value_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);
  torch::Tensor block_mapping_tensor = torch::from_blob(
    block_mapping_array, {2 * num_pairs}, torch::kInt64).to(cache_device);

  // Launch the kernel.
  const int numel_per_block = key_caches[0][0].numel();
  dim3 grid(num_layers, num_pairs);
  dim3 block(std::min(1024, numel_per_block));
  const at::cuda::OptionalCUDAGuard device_guard(cache_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    key_caches[0].scalar_type(), "copy_blocks_kernel", ([&] {
      vllm::copy_blocks_kernel<scalar_t><<<grid, block, 0, stream>>>(
        key_cache_ptrs_tensor.data_ptr<int64_t>(),
        value_cache_ptrs_tensor.data_ptr<int64_t>(),
        block_mapping_tensor.data_ptr<int64_t>(),
        numel_per_block);
    }));
}

namespace vllm {

template<typename scalar_t>
__global__ void reshape_and_cache_kernel(
  const scalar_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
  scalar_t* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                                + head_idx * (head_size / x) * block_size * x
                                + x_idx * block_size * x
                                + block_offset * x
                                + x_offset;
    const int64_t tgt_value_idx = block_idx * num_heads * head_size * block_size
                                  + head_idx * head_size * block_size
                                  + head_offset * block_size
                                  + block_offset;
    key_cache[tgt_key_idx] = key[src_key_idx];
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}

} // namespace vllm

void reshape_and_cache(
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,   // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& slot_mapping)  // [num_tokens]
{
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    key.scalar_type(),
    "reshape_and_cache_kernel",
    [&] {
      vllm::reshape_and_cache_kernel<scalar_t><<<grid, block, 0, stream>>>(
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int64_t>(),
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        x);
    });
}

namespace vllm {

// Grid: (num_blocks, block_size).
template<typename scalar_t>
__global__ void gather_cached_kv_kernel(
  scalar_t* __restrict__ key,             // [num_tokens, [stride], num_heads, head_size]
  scalar_t* __restrict__ value,           // [num_tokens, [stride], num_heads, head_size]
  const scalar_t* __restrict__ key_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
  const scalar_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size, block_size]
  const int* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
    const int token_idx = blockIdx.x;
    const int slot_idx = slot_mapping[token_idx];
    const int block_idx = slot_idx / block_size;
    const int block_offset = slot_idx % block_size;

    const int num_tokens = num_heads * head_size;
    for (int i = threadIdx.x; i < num_tokens; i += blockDim.x) {
      const int tgt_key_idx = token_idx * key_stride + i;
      const int tgt_value_idx = token_idx * value_stride + i;
  
      const int head_idx = i / head_size;
      const int head_offset = i % head_size;
      const int x_idx = head_offset / x;  // the offset of the [head_size/x] dimension
      const int x_offset = head_offset % x;
  
      const int src_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                              + head_idx * (head_size / x) * block_size * x
                              + x_idx * block_size * x
                              + block_offset * x
                              + x_offset;
      const int src_value_idx = block_idx * num_heads * head_size * block_size
                                + head_idx * head_size * block_size
                                + head_offset * block_size
                                + block_offset;

      key[tgt_key_idx] = VLLM_LDG(&key_cache[src_key_idx]);
      value[tgt_value_idx] = VLLM_LDG(&value_cache[src_value_idx]);
    }
}

template <typename scalar_t>
__global__ void gather_cached_kv_kernel_optimized(
    scalar_t *__restrict__ key,             // [num_tokens, [stride], num_heads, head_size]
    scalar_t *__restrict__ value,           // [num_tokens, [stride], num_heads, head_size]
    const scalar_t *__restrict__ key_cache, // [num_blocks, num_heads, head_size/x, block_size, x]
    const scalar_t *__restrict__ value_cache, // [num_blocks, num_heads, head_size, block_size]
    const int *__restrict__ slot_mapping,   // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x)
{
    const int token_idx = blockIdx.x;
    const int slot_idx = slot_mapping[token_idx];
    const int block_idx = slot_idx / block_size;
    const int block_offset = slot_idx % block_size;

    const int dim = num_heads * head_size;
    assert(dim % 4 == 0);  // this is true for known use cases
    const int unroll_factor = 4;
    const int unrolled_dim = dim / unroll_factor;

    for (int i = threadIdx.x; i < unrolled_dim; i += blockDim.x)
    {
        int tgt_key_indices[unroll_factor];
        int tgt_value_indices[unroll_factor];
        int src_key_indices[unroll_factor];
        int src_value_indices[unroll_factor];
        scalar_t keys_to_store[unroll_factor];
        scalar_t values_to_store[unroll_factor];

        #pragma unroll
        for (int j = 0; j < unroll_factor; ++j)
        {
            int index = i + j * unrolled_dim;

            const int tgt_key_idx = token_idx * key_stride + index;
            const int tgt_value_idx = token_idx * value_stride + index;

            const int head_idx = index / head_size;
            const int head_offset = index % head_size;
            const int x_idx = head_offset / x;
            const int x_offset = head_offset % x;

            const int src_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                                    + head_idx * (head_size / x) * block_size * x
                                    + x_idx * block_size * x
                                    + block_offset * x
                                    + x_offset;
            const int src_value_idx = block_idx * num_heads * head_size * block_size
                                      + head_idx * head_size * block_size
                                      + head_offset * block_size
                                      + block_offset;

            tgt_key_indices[j] = tgt_key_idx;
            tgt_value_indices[j] = tgt_value_idx;
            src_key_indices[j] = src_key_idx;
            src_value_indices[j] = src_value_idx;

            keys_to_store[j] = VLLM_LDG(&key_cache[src_key_idx]);
            values_to_store[j] = VLLM_LDG(&value_cache[src_value_idx]);
        }

        #pragma unroll
        for (int j = 0; j < unroll_factor; ++j)
        {
            key[tgt_key_indices[j]] = keys_to_store[j];
            value[tgt_value_indices[j]] = values_to_store[j];
        }
    }
}

} // namespace vllm

void gather_cached_kv(
  torch::Tensor& key,           // [out] [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [out] [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [in]  [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,   // [in]  [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& slot_mapping)  // [in]  [num_tokens]
{
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    key.scalar_type(),
    "gather_cached_kv_kernel_optimized",
    [&] {
      vllm::gather_cached_kv_kernel_optimized<scalar_t><<<grid, block, 0, stream>>>(
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int>(),
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        x);
    });
}


// Block Migration Code

/*
assert_whenever: assertion which ignore whether NDEBUG is set

In C++, assert() is evaluated only when NDEBUG is not set. This is
inconvenient when we want to check the assertion even in release mode.
This macro is a workaround for this problem.
*/

extern "C" {
// Copied from assert.h
extern void __assert_fail (const char *__assertion, const char *__file,
			   unsigned int __line, const char *__function)
     __THROW __attribute__ ((__noreturn__));

#define __ASSERT_FUNCTION	__extension__ __PRETTY_FUNCTION__
#  define assert_whenever(expr)							\
     (static_cast <bool> (expr)						\
      ? void (0)							\
      : __assert_fail (#expr, __FILE__, __LINE__, __ASSERT_FUNCTION))
}

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

#define INDEX_5D(dim1, dim2, dim3, dim4, dim5, index1, index2, index3, index4, index5) \
    (((int64_t)index1) * (dim2) * (dim3) * (dim4) * (dim5) + ((int64_t)index2) * (dim3) * (dim4) * (dim5) + ((int64_t)index3) * (dim4) * (dim5) + (index4) * (dim5) + (index5))




static std::vector<int64_t> cudaIpcMemHandle2Bytes(const cudaIpcMemHandle_t &handle) {
	std::vector<int64_t> result;
	for (size_t i = 0; i < sizeof(handle); ++i) {
		result.push_back(((uint8_t*) &handle)[i]);
	}
	return result;
}

static cudaIpcMemHandle_t bytes2CudaIpcMemHandle(const std::vector<int64_t> &bytes) {
	assert_whenever(bytes.size() == sizeof(cudaIpcMemHandle_t));
	cudaIpcMemHandle_t result;
	for (size_t i = 0; i < sizeof(result); ++i) {
		((uint8_t*) &result)[i] = bytes[i];
	}
	return result;
}




/*
get_ipc_mem_handle: Get the IPC memory handle of a tensor
The returned handle can be used to open the tensor in another process.
*/
std::vector<int64_t> get_ipc_mem_handle(torch::Tensor tensor) {
	cudaIpcMemHandle_t handle;
	CUDA_CHECK(cudaIpcGetMemHandle(&handle, tensor.data_ptr()));
	return cudaIpcMemHandle2Bytes(handle);
}


/*
register_ipc_mem_handle: Register an IPC memory handle

This function receives a IPC memory handle and the context worker's info
(context_pp_rank and context_tp_rank) that the handle belongs to, then
it checks whether it needs to register the handle (i.e. whether the k/v range
it needs overlaps with the k/v range that the context worker calculates). If
the answer is YES, register it and note down its local address.

Return true if the handle is registered, false otherwise.
*/
static constexpr int64_t MAX_PARALLEL_HASH = 4096;	// Assume there are at most 64 pp stages and 64 tp stages
static void* context_worker_k_cache_addr[MAX_PARALLEL_HASH];
static void* context_worker_v_cache_addr[MAX_PARALLEL_HASH];
bool register_ipc_mem_handle(
	std::vector<int64_t> k_cache_handle_vec,
	std::vector<int64_t> v_cache_handle_vec,
	int64_t num_layers,
	int64_t num_heads,
	const std::vector<int64_t> &context_parallel_config,	// Generated via ParallelConfig.to_list()
	const std::vector<int64_t> &decoding_parallel_config
) {
	// Convert the handles to cudaIpcMemHandle_t
	const cudaIpcMemHandle_t k_cache_handle = bytes2CudaIpcMemHandle(k_cache_handle_vec);
	const cudaIpcMemHandle_t v_cache_handle = bytes2CudaIpcMemHandle(v_cache_handle_vec);

	// First we check whether the two k/v cache area overlaps
	const int64_t context_tp_size = context_parallel_config[0];
	const int64_t context_tp_rank = context_parallel_config[1];
	const int64_t context_pp_size = context_parallel_config[2];
	const int64_t context_pp_rank = context_parallel_config[3];
	const int64_t decoding_tp_size = decoding_parallel_config[0];
	const int64_t decoding_tp_rank = decoding_parallel_config[1];
	const int64_t decoding_pp_size = decoding_parallel_config[2];
	const int64_t decoding_pp_rank = decoding_parallel_config[3];

	const int64_t layers_per_context_worker = num_layers / context_pp_size;
	const int64_t heads_per_context_worker = num_heads / context_tp_size;
	const int64_t layers_per_decoding_worker = num_layers / decoding_pp_size;
	const int64_t heads_per_decoding_worker = num_heads / decoding_tp_size;

	const int64_t context_start_layer = context_pp_rank * layers_per_context_worker;
	const int64_t context_end_layer = context_start_layer + layers_per_context_worker;
	const int64_t context_start_head = context_tp_rank * heads_per_context_worker;
	const int64_t context_end_head = context_start_head + heads_per_context_worker;

	const int64_t decoding_start_layer = decoding_pp_rank * layers_per_decoding_worker;
	const int64_t decoding_end_layer = decoding_start_layer + layers_per_decoding_worker;
	const int64_t decoding_start_head = decoding_tp_rank * heads_per_decoding_worker;
	const int64_t decoding_end_head = decoding_start_head + heads_per_decoding_worker;

	if (context_end_layer <= decoding_start_layer || context_start_layer >= decoding_end_layer ||
		context_end_head <= decoding_start_head || context_start_head >= decoding_end_head) {
		// No overlap
		return false;
	} else {
		// Overlap
		// Register the handle
		const int64_t context_worker_hash = (context_pp_rank<<6) + context_tp_rank;
		CUDA_CHECK(cudaIpcOpenMemHandle(&context_worker_k_cache_addr[context_worker_hash], k_cache_handle, cudaIpcMemLazyEnablePeerAccess));
		CUDA_CHECK(cudaIpcOpenMemHandle(&context_worker_v_cache_addr[context_worker_hash], v_cache_handle, cudaIpcMemLazyEnablePeerAccess));
		return true;
	}
}

/*
migrate_blocks: Migrate blocks from the context stage engine to the decoding stage engine

This function is called by every decoding stage worker when the decoding
stage engine decides to migrate some blocks from the context stage engine
to the decoding stage engine.

In the following code, "pp" stands for "pipeline parallel", and "tp" stands
for "tensor parallel".

Here we do not pass a cudaStream to the function. Instead we use the current
stream indicated by at::cuda::getCurrentCUDAStream(). So it is python's
responsibility to set the current stream before calling this function.
*/

void migrate_blocks(
	// Parallelism parameters for the context stage engine
	const int64_t context_pp_size,
	const int64_t context_tp_size,

	// Block indexes of the context stage engine
	const std::vector<int64_t> &context_block_indexes,

	// Parallelism parameters for the decoding stage engine
	const int64_t decoding_pp_size,
	const int64_t decoding_tp_size,

	// Rank of the decoding stage worker that calls this function
	const int64_t decoding_pp_rank,
	const int64_t decoding_tp_rank,

	// Block indexes of the decoding stage engine
	const std::vector<int64_t> &decoding_block_indexes,

	// The decoding stage worker's KV cache
	torch::Tensor decoding_worker_k_cache,	// [num_blocks, layers_per_decoding_worker, heads_per_decoding_worker, block_size, head_dim]
	torch::Tensor decoding_worker_v_cache
) {
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	assert_whenever(decoding_worker_k_cache.is_contiguous());
	assert_whenever(decoding_worker_v_cache.is_contiguous());

	// Calculate some misc stuff
	const int64_t layers_per_decoding_worker = decoding_worker_k_cache.size(1);
	const int64_t heads_per_decoding_worker = decoding_worker_k_cache.size(2);
	const int64_t block_size = decoding_worker_k_cache.size(3);
	const int64_t head_dim = decoding_worker_k_cache.size(4);
	const int64_t num_layers = layers_per_decoding_worker * decoding_pp_size;
	const int64_t num_heads = heads_per_decoding_worker * decoding_tp_size;
	const int64_t layers_per_context_worker = num_layers / context_pp_size;
	const int64_t heads_per_context_worker = num_heads / context_tp_size;
	const int64_t num_blocks_to_copy = decoding_block_indexes.size();
	const int64_t dtype_size = decoding_worker_k_cache.dtype().itemsize();

	// The current decoding worker's region of the k/v cache
	const int64_t decoding_start_layer = decoding_pp_rank * layers_per_decoding_worker;
	const int64_t decoding_end_layer = decoding_start_layer + layers_per_decoding_worker;
	const int64_t decoding_start_head = decoding_tp_rank * heads_per_decoding_worker;
	const int64_t decoding_end_head = decoding_start_head + heads_per_decoding_worker;

	for (int64_t context_pp_rank = 0; context_pp_rank < context_pp_size; ++context_pp_rank) {
		// First we iterate over every context pp stage
		const int64_t context_start_layer = context_pp_rank * layers_per_context_worker;
		const int64_t context_end_layer = context_start_layer + layers_per_context_worker;
		if (context_end_layer <= decoding_start_layer || context_start_layer >= decoding_end_layer) {
			continue;
		}
		for (int64_t context_tp_rank = 0; context_tp_rank < context_tp_size; ++context_tp_rank) {
			// Then we iterate over every context tp worker in the current pp stage
			const int64_t context_start_head = context_tp_rank * heads_per_context_worker;
			const int64_t context_end_head = context_start_head + heads_per_context_worker;
			if (context_end_head <= decoding_start_head || context_start_head >= decoding_end_head) {
				continue;
			}

			// The current context worker's region intersects with the current decoding worker's region
			// So we need to copy something from the context worker to the decoding worker
			// The context worker holds k/v cache of range [context_start_layer, context_end_layer) x [context_start_head, context_end_head)
			// The decoding worker holds k/v cache of range [decoding_start_layer, decoding_end_layer) x [decoding_start_head, decoding_end_head)
			// We then calculate the intersection of these two ranges
			const int64_t overlap_start_layer = std::max(context_start_layer, decoding_start_layer);
			const int64_t overlap_end_layer = std::min(context_end_layer, decoding_end_layer);
			const int64_t overlap_start_head = std::max(context_start_head, decoding_start_head);
			const int64_t overlap_end_head = std::min(context_end_head, decoding_end_head);
			assert_whenever(overlap_start_layer < overlap_end_layer);
			assert_whenever(overlap_start_head < overlap_end_head);

			// Note that this function is synchronous with respect to the host only if the source or destination of the transfer is host memory.
			// Note also that this copy is serialized with respect to all pending and future asynchronous work in to the current device, the copy's source device, and the copy's destination device (use cudaMemcpy3DPeerAsync to avoid this synchronization).

			// kv cache shape: [num_blocks, layers_per_worker, heads_per_worker, block_size, head_dim]
			for (int64_t block_id = 0; block_id < num_blocks_to_copy; ++block_id) {
				const int64_t context_block_index = context_block_indexes[block_id];
				const int64_t decoding_block_index = decoding_block_indexes[block_id];
				for (int is_value = 0; is_value < 2; ++is_value) {
					const int64_t context_worker_hash = (context_pp_rank<<6) + context_tp_rank;
					char* context_worker_base_ptr = (char*) (is_value ? context_worker_v_cache_addr[context_worker_hash] : context_worker_k_cache_addr[context_worker_hash]);
					if (!context_worker_base_ptr) {
						// This context worker has not registered. Panic
						fprintf(stderr, "Error: context worker %ld-%ld has not registered\n", context_pp_rank, context_tp_rank);
						exit(1);
					}
					CUDA_CHECK(cudaMemcpy2DAsync(
						(char*) (is_value ? decoding_worker_v_cache.data_ptr() : decoding_worker_k_cache.data_ptr())
							+ INDEX_5D(0, layers_per_decoding_worker, heads_per_decoding_worker, block_size, head_dim,
								decoding_block_index,
								overlap_start_layer - decoding_start_layer,
								overlap_start_head - decoding_start_head,
								0, 0) * dtype_size,
						(uint64_t) ((block_size * head_dim * dtype_size) * heads_per_decoding_worker),
						context_worker_base_ptr
							+ INDEX_5D(0, layers_per_context_worker, heads_per_context_worker, block_size, head_dim,
								context_block_index,
								overlap_start_layer - context_start_layer,
								overlap_start_head - context_start_head,
								0, 0) * dtype_size,
						(uint64_t) ((block_size * head_dim * dtype_size) * heads_per_context_worker),
						(size_t) ((overlap_end_head - overlap_start_head) * block_size * head_dim * dtype_size),
						(size_t) (overlap_end_layer - overlap_start_layer),
						cudaMemcpyDeviceToDevice,
						stream
					));
				}
			}
		}
	}
}
