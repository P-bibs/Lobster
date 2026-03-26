#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

#include <chrono>
#include <cmath>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/tuple>
#include <iostream>
#include <numeric>

template <class T>
std::ostream &operator<<(std::ostream &os, const thrust::device_vector<T> &v) {
  os << "[";
  for (size_t i = 0; i < v.size(); i++) {
    std::cout << v[i] << " ";
  }
  os << "]";
  std::cout << std::endl;
  return os;
}

struct Point {
  uint32_t x, y;
};
struct PointDecomposer {
  __host__ __device__ ::cuda::std::tuple<uint32_t &, uint32_t &> operator()(
      Point &p) const {
    return ::cuda::std::tuple<uint32_t &, uint32_t &>(p.x, p.y);
  }
};
struct decomposer_t {
  __host__ __device__ ::cuda::std::tuple<uint32_t &, uint32_t &> operator()(
      thrust::tuple<uint32_t, uint32_t> &p) const {
    return ::cuda::std::tuple<uint32_t &, uint32_t &>(thrust::get<0>(p),
                                                      thrust::get<1>(p));
  }
};

// int main() {
//   std::cout << "Running..." << std::endl;
//   std::vector<Data> host_data{Data(4.0f, 10, -20ll), Data(2.0f, 20, -10ll),
//                               Data(3.0f, 30, -30ll), Data(1.0f, 40, -40ll),
//                               Data(4.0f, 50, -50ll), Data(2.0f, 60, -60ll),
//                               Data(3.0f, 70, -70ll)};
//   thrust::device_vector<Data> keys(host_data);
//   int num_items = keys.size();
//   std::cout << "Made vector " << keys << std::endl;
//
//   thrust::device_vector<Data> out(num_items);
//
//   // Determine temporary device storage requirements
//   Data *in_ptr = thrust::raw_pointer_cast(keys.data());
//   Data *out_ptr = thrust::raw_pointer_cast(out.data());
//   void *d_temp_storage = nullptr;
//   size_t temp_storage_bytes = 0;
//   cub::DeviceRadixSort::SortKeys<Data, int, decomposer_t>(
//       d_temp_storage, temp_storage_bytes, in_ptr, out_ptr, num_items,
//       decomposer_t{}, 0, 1);
//
//   thrust::device_vector<unsigned char> temp_storage(temp_storage_bytes);
//   std::cout << "Sorting..." << std::endl;
//   d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
//   cub::DeviceRadixSort::SortKeys<Data, int, decomposer_t>(
//       d_temp_storage, temp_storage_bytes, in_ptr, out_ptr, num_items,
//       decomposer_t{});
//   std::cout << "Sorted vector " << out << std::endl;
//   return 0;
// }
//
// int foo() {
//   std::cout << "Running..." << std::endl;
//   int num_items = 7;
//   thrust::device_vector<int> keys(num_items);
//   thrust::sequence(keys.begin(), keys.end());
//   thrust::default_random_engine g;
//   thrust::shuffle(keys.begin(), keys.end(), g);
//   std::cout << "Made vector " << keys << std::endl;
//
//   thrust::device_vector<int> out(num_items);
//
//   // Determine temporary device storage requirements
//   int *in_ptr = thrust::raw_pointer_cast(keys.data());
//   int *out_ptr = thrust::raw_pointer_cast(out.data());
//   void *d_temp_storage = nullptr;
//   size_t temp_storage_bytes = 0;
//   cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, in_ptr,
//                                  out_ptr, num_items);
//   thrust::device_vector<unsigned char> temp_storage(temp_storage_bytes);
//   std::cout << "Sorting..." << std::endl;
//   d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
//   cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, in_ptr,
//                                  out_ptr, num_items);
//   std::cout << "Sorted vector " << out << std::endl;
//   return 0;
// }

void merge_sort(
    thrust::zip_iterator<thrust::tuple<uint32_t *, uint32_t *>> &it,
    thrust::zip_iterator<thrust::tuple<uint32_t *, uint32_t *>> &out,
    std::size_t size) {

  thrust::copy(thrust::device, it, it + size, out);

  //std::size_t temp_storage_bytes = 0;
  //void *d_temp_storage = nullptr;
  //cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, out, size);
  //thrust::device_vector<char> v(temp_storage_bytes);
  //d_temp_storage = thrust::raw_pointer_cast(v.data());
  //cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, out, size);
  thrust::sort(thrust::device, out, out + size);
}

void radix_sort(
    thrust::zip_iterator<thrust::tuple<uint32_t *, uint32_t *>> &it,
    thrust::zip_iterator<thrust::tuple<uint32_t *, uint32_t *>> &out,
    std::size_t size) {
  std::size_t temp_storage_bytes = 0;
  void *d_temp_storage = nullptr;

  thrust::device_vector<thrust::tuple<uint32_t, uint32_t>> compacted(size);
  thrust::copy(thrust::device, it, it + size, compacted.begin());
  thrust::device_vector<thrust::tuple<uint32_t, uint32_t>> compacted_out(size);

  // do the sort
  decomposer_t decomposer;
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                 thrust::raw_pointer_cast(compacted.data()),
                                 thrust::raw_pointer_cast(compacted_out.data()), size,
                                 decomposer);
  thrust::device_vector<char> v(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(v.data());
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                 thrust::raw_pointer_cast(compacted.data()),
                                 thrust::raw_pointer_cast(compacted_out.data()), size,
                                 decomposer);

  thrust::copy(compacted_out.begin(), compacted_out.end(), out);
  return;
}

int main() {
  std::size_t size = 100000000;
  thrust::device_vector<uint32_t> in0(size);
  thrust::device_vector<uint32_t> in1(size);
  thrust::sequence(thrust::device, in0.begin(), in0.end());
  thrust::sequence(thrust::device, in1.begin(), in1.end());

  auto in = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::raw_pointer_cast(in0.data()),
                         thrust::raw_pointer_cast(in1.data())));

  thrust::device_vector<uint32_t> out0(size);
  thrust::device_vector<uint32_t> out1(size);
  auto out = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::raw_pointer_cast(out0.data()),
                         thrust::raw_pointer_cast(out1.data())));

  thrust::default_random_engine g(1);

  const int ITERATIONS = 1000;
  const int WARMUP = 100;

  auto merge_sort_times = std::vector<double>();
  for (int i = 0; i < ITERATIONS; i++) {
    thrust::shuffle(thrust::device, in, in + size, g);
    auto now = std::chrono::high_resolution_clock::now();
    merge_sort(in, out, size);
    auto end = std::chrono::high_resolution_clock::now();
    if (i >= WARMUP) {
      merge_sort_times.push_back(
          std::chrono::duration_cast<std::chrono::duration<double>>(end - now)
              .count());
    }
  }

  auto radix_sort_times = std::vector<double>();
  for (int i = 0; i < ITERATIONS; i++) {
    thrust::shuffle(thrust::device, in, in + size, g);
    auto now = std::chrono::high_resolution_clock::now();
    merge_sort(in, out, size);
    auto end = std::chrono::high_resolution_clock::now();
    if (i >= WARMUP) {
      radix_sort_times.push_back(
          std::chrono::duration_cast<std::chrono::duration<double>>(end - now)
              .count());
    }
  }

  auto merge_avg =
      std::accumulate(merge_sort_times.begin(), merge_sort_times.end(), 0.0) /
      merge_sort_times.size();
  auto radix_avg =
      std::accumulate(radix_sort_times.begin(), radix_sort_times.end(), 0.0) /
      merge_sort_times.size();

  auto merge_stddev = std::sqrt(
      std::accumulate(merge_sort_times.begin(), merge_sort_times.end(), 0.0,
                      [merge_avg](double a, double b) {
                        return a + (b - merge_avg) * (b - merge_avg);
                      }) /
      merge_sort_times.size());
  auto radix_stddev = std::sqrt(
      std::accumulate(radix_sort_times.begin(), radix_sort_times.end(), 0.0,
                      [radix_avg](double a, double b) {
                        return a + (b - radix_avg) * (b - radix_avg);
                      }) /
      radix_sort_times.size());

  std::cout << "Merge sort: " << merge_avg << " +/- " << merge_stddev
            << std::endl;
  std::cout << "Radix sort: " << radix_avg << " +/- " << radix_stddev
            << std::endl;
}
