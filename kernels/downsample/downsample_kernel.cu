#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

template <typename dtype>
struct super_pixel {
    dtype val;
    int key;
    int super_pixel_id;

    __device__ super_pixel() {}

    __device__ super_pixel(dtype val, int key, int super_pixel_id) : val(val), key(key), super_pixel_id(super_pixel_id) {}

    __device__ friend bool operator<=(const super_pixel &lhs, const super_pixel &rhs) {
        return lhs.key <= rhs.key;
    }
};

// standard partition process of QuickSort().
// It considers the last element as pivot
// and move all smaller element to left of
// it and greater elements to right
template <typename dtype>
__device__ int partition(dtype* arr, int l, int r) {
    dtype x = arr[r];
    int i = l;
    dtype tmp;
    for (int j = l; j <= r - 1; j++) {
        if (arr[j] <= x) {
            tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            i++;
        }
    }
    tmp = arr[i];
    arr[i] = arr[r];
    arr[r] = tmp;
    return i;
}

// This function returns k'th smallest
// element in arr[l..r] using QuickSort
// based method. ASSUMPTION: ALL ELEMENTS
// IN ARR[] ARE DISTINCT
template <typename dtype>
__device__ dtype kth(dtype* arr, int l, int r, int k) {
    while (l <= r) {
        // Partition a[l..r] around last
        // element and get position of pivot
        // element in sorted array
        int index = partition(arr, l, r);

        // If pivot itself is the k-th smallest element
        if (index - l == k - 1) {
            return arr[index];
        }

        // If there are more than k-1 elements on left of pivot,
        // then k-th smallest must be on left side.
        else if (index - l > k - 1) {
            r = index - 1;
        }

        // Else k-th smallest is on the right side.
        else {
            k -= index - l + 1;
            l = index + 1;
        }
    }

    // should not reach here. Just for suppressing warning
    return arr[l];
}

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                               \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;  \
         i += blockDim.x * gridDim.x)

// precondition: labels in each block are sorted
template <typename dtype>
__global__ void downsample_forward_kernel(const int nthreads, const dtype* __restrict__ bottom_data, const int* label,
                                          const int block_size, const float nms_thresh, dtype* __restrict__ top_data) {
   CUDA_1D_KERNEL_LOOP(index, nthreads) {
        dtype vals[64];
        int start[65];
        int num_super_pixel = 0;
        int num_valid_pixel = 0;
        int offset = index * block_size;
        bool new_super_pixel = false;  // flags whether encounter a new super pixel
        for (int i = 0; i < block_size; ++i) {
            if ((i == 0) || (label[offset+i] != label[offset+i-1])) new_super_pixel = true;

            dtype v = bottom_data[offset+i];
            if (v <= 0) continue;
            if (new_super_pixel) {
                start[num_super_pixel++] = num_valid_pixel;
                new_super_pixel = false;
            }
            vals[num_valid_pixel++] = v;
        }
        start[num_super_pixel] = num_valid_pixel;

        super_pixel<dtype> median[64];
        for (int i = 0; i < num_super_pixel; ++i) {
            int l = start[i];
            int r = start[i+1] - 1;
            int k = (r - l + 1) / 2 + 1;
            median[i] = super_pixel<dtype>(kth(vals, l, r, k), -(r - l + 1), i);  // take minus for the ease of sort
        }

        // NMS
        dtype vals_[64];
        int starts_[65];
        int num_mode = 0;
        int idx = 0;
        for (int i = 0; i < num_super_pixel; i++) {
            if (num_mode == 4) continue;
            super_pixel<dtype> p = kth(median, 0, num_super_pixel - 1, 1);
            if (p.key == 1000) break;
            median[0].key = 1000;
            starts_[num_mode++] = idx;
            for (int j = start[p.super_pixel_id]; j < start[p.super_pixel_id+1]; j++) {
                vals_[idx++] = vals[j];
            }
            for (int k = 1; k < num_super_pixel; k++) {
                if (median[k].key == 1000) continue;
                if (abs(median[k].val - p.val) < nms_thresh) {
                    // suppress
                    median[k].key = 1000;
                    for (int j = start[median[k].super_pixel_id]; j < start[median[k].super_pixel_id+1]; j++) {
                        vals_[idx++] = vals[j];
                    }
                }
            }
        }
        starts_[num_mode] = idx;

        offset = index * 4;
        for (int i = 0; i < num_mode; i++) {
            int l = starts_[i];
            int r = starts_[i+1] - 1;
            int k = (r - l + 1) / 2 + 1;
            top_data[offset+i] = kth(vals_, l, r, k);
        }
   }
}

at::Tensor downsample_forward_cuda(const at::Tensor& input,
                                   const at::Tensor& label,
                                   const float nms_thresh) {
  AT_ASSERTM(input.device().is_cuda(), "input must be a CUDA tensor");

  at::cuda::CUDAGuard device_guard(input.device());

  auto output = at::zeros({input.size(0), 4}, input.options());
  auto output_size = input.size(0);
  auto block_size = input.size(1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(at::cuda::ATenCeilDiv(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
            static_cast<int64_t>(4096)));
  dim3 block(512);

  if (output.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
  }

  auto input_ = input.contiguous();
  auto label_ = label.contiguous();

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "downsample_forward_kernel", [&] {
    downsample_forward_kernel<scalar_t><<<grid, block, 0, stream>>>(
        output_size, input_.data_ptr<scalar_t>(), label_.data_ptr<int>(), block_size, nms_thresh, output.data_ptr<scalar_t>());
  });
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}