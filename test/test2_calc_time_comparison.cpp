#include <iostream>
#include <cstdint>
#include <chrono>
#include <cstdlib>
#include "../vectorize.h"

constexpr size_t N = 1 << 20; // 1M elements

int main() {
    // Memory alignment
    float* array1 = new float[N];
    float* array2 = new float[N];
    float* result = new float[N];


    if (!array1 || !array2 || !result) {
        std::cerr << "Memory allocation failed\n";
        return -1;
    }

    // Initialize
    for (size_t i = 0; i < N; i++) {
        array1[i] = static_cast<float>(i) * 0.001f;
        array2[i] = static_cast<float>(i) * 0.002f;
        result[i] = 0.0f;
    }

    // ---------- Scalar addition ----------
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++) {
        result[i] = array1[i] + array2[i];
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_scalar = end_scalar - start_scalar;
    std::cout << "Scalar add time: " << elapsed_scalar.count() << " s\n";

    // ---------- SIMD addition ----------
    auto start_simd = std::chrono::high_resolution_clock::now();

    size_t i = 0;
    for (; i + VEC_WIDTH <= N; i += VEC_WIDTH) {
        vfloat32_t v1 = VEC_LOAD_F(array1 + i);
        vfloat32_t v2 = VEC_LOAD_F(array2 + i);
        vfloat32_t vres = VEC_ADD_F(v1, v2);
        VEC_STORE_F(result + i, vres);
    }

    // Tail
    for (; i < N; i++) {
        result[i] = array1[i] + array2[i];
    }

    auto end_simd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_simd = end_simd - start_simd;
    std::cout << "SIMD add time: " << elapsed_simd.count() << " s\n";


    std::cout << "First 8 results: ";
    for (size_t j = 0; j < 8; j++) {
        std::cout << result[j] << " ";
    }
    std::cout << "\n";

    // free mem
    delete[] array1;
    delete[] array2;
    delete[] result;

    return 0;
}
