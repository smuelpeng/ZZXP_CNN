#include "./common.hpp"
#include "./syncedmem.hpp"
#include "./util/math_functions.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifdef USE_CUDA
  if (gpu_ptr_ && own_gpu_data_) {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
  }
#endif  // USE_CUDA
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifdef USE_CUDA
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif  // USE_CUDA
    break;
  case HEAD_AT_PRV:
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    CHECK(prv_descriptor_.get());
    prv_descriptor_->convert_from_prv(cpu_ptr_);
    prv_descriptor_->on_to_cpu();
    head_ = SYNCED_PRV;
    break;
  case SYNCED_PRV:
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
#ifdef USE_CUDA
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case HEAD_AT_PRV:
    to_cpu();
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaGetDevice(&gpu_device_));
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif  // USE_CUDA
}

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
#ifdef USE_CUDA
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif  // USE_CUDA
}

void SyncedMemory::set_gpu_data(void* data) {
#ifdef USE_CUDA
  CHECK(data);
  if (own_gpu_data_) {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif  // USE_CUDA
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
#ifdef USE_CUDA
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif  // USE_CUDA
}

void SyncedMemory::set_prv_descriptor(shared_ptr<PrvMemDescr> descriptor,
        bool same_data) {
  // If it wasn't synced before, it won't be now.
  if (descriptor == NULL) {
    if (head_ != UNINITIALIZED)
      head_ = HEAD_AT_CPU;
  } else {
    if ((head_ != HEAD_AT_PRV) && same_data)
      head_ = SYNCED_PRV;
    else
      head_ = HEAD_AT_PRV;
  }

  prv_descriptor_ = descriptor;
}

const void* SyncedMemory::prv_data() {
  if ((head_ != HEAD_AT_PRV) &&
     (head_ != SYNCED_PRV)) {
    return NULL;
  }

  CHECK(prv_descriptor_.get());
  return (const void* ) prv_descriptor_->prv_ptr();
}

void* SyncedMemory::mutable_prv_data() {
  CHECK(prv_descriptor_.get());
  if (head_ == HEAD_AT_CPU) {
    prv_descriptor_->convert_to_prv(cpu_ptr_);
  }
  head_ = HEAD_AT_PRV;
  return prv_descriptor_->prv_ptr();
}


}  // namespace caffe
