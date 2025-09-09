## Vectorize Calc Header
This is a **Single Header lightweighted library** for vectorize SIMD calculating.
 * Author: 月と猫 - LunaNeko
 * Mainly assisted by GPT-5 

中文文档：[README.md](README.md)

Ver: v0.1
- WIP: Adjustment to RISC-V Platform. (Will complete in the next version)

## Tutorial:

Just simply include this header to use.
```C
#include <vectorize.h>
```
## 1. Datatype

This header provides the following macros:

| Macro Name | Type | Description |
|------------|------|-------------|
| `vfloat32_t` | float SIMD vector type | Single-precision floating point vector, e.g., AVX/SSE/NEON |
| `vint_t` | int SIMD vector type | Integer vector type |
| `vmask_t` | int / mask vector type | Used for conditional mask operations |
| `VEC_WIDTH` | int | Float vector width in current platform (file-scope macro, user can access via interface) |

## 2. Macro / Function Table

| Return Type | Macro/Function | Parameters | Mnemonic | Description |
|------------|----------------|-----------|----------|-------------|
| `vfloat32_t` | `VEC_LOAD_F(ptr)` | `(const float *ptr)` | Load Float Vector | Load a float vector from memory |
| `vint_t` | `VEC_LOAD_I(ptr)` | `(const int *ptr)` | Load Int Vector | Load an integer vector from memory |
| `void` | `VEC_STORE_F(ptr, vec)` | `(float *ptr, vfloat32_t vec)` | Store Float Vector | Store a float vector to memory |
| `void` | `VEC_STORE_I(ptr, vec)` | `(int *ptr, vint_t vec)` | Store Int Vector | Store an integer vector to memory |
| `vfloat32_t` | `VEC_ADD_F(a, b)` | `(vfloat32_t a, vfloat32_t b)` | Add Float Vector | Vectorized float addition |
| `vint_t` | `VEC_ADD_I(a, b)` | `(vint_t a, vint_t b)` | Add Int Vector | Vectorized integer addition |
| `vfloat32_t` | `VEC_SUB_F(a, b)` | `(vfloat32_t a, vfloat32_t b)` | Sub Float Vector | Vectorized float subtraction |
| `vint_t` | `VEC_SUB_I(a, b)` | `(vint_t a, vint_t b)` | Sub Int Vector | Vectorized integer subtraction |
| `vfloat32_t` | `VEC_MUL_F(a, b)` | `(vfloat32_t a, vfloat32_t b)` | Mul Float Vector | Vectorized float multiplication |
| `vint_t` | `VEC_MUL_I(a, b)` | `(vint_t a, vint_t b)` | Mul Int Vector | Vectorized integer multiplication |
| `vfloat32_t` | `VEC_DIV_F(a, b)` | `(vfloat32_t a, vfloat32_t b)` | Div Float Vector | Vectorized float division |
| `vfloat32_t` | `VEC_FMA_F(a, b, c)` | `(vfloat32_t a, vfloat32_t b, vfloat32_t c)` | Fused Multiply Add | Vectorized FMA: a*b + c |
| `vfloat32_t` | `VEC_AND_F(a, b)` | `(vfloat32_t a, vfloat32_t b)` | And Float Vector | Bitwise AND on float vector |
| `vint_t` | `VEC_AND_I(a, b)` | `(vint_t a, vint_t b)` | And Int Vector | Bitwise AND on int vector |
| `vfloat32_t` | `VEC_OR_F(a, b)` | `(vfloat32_t a, vfloat32_t b)` | Or Float Vector | Bitwise OR on float vector |
| `vint_t` | `VEC_OR_I(a, b)` | `(vint_t a, vint_t b)` | Or Int Vector | Bitwise OR on int vector |
| `vfloat32_t` | `VEC_XOR_F(a, b)` | `(vfloat32_t a, vfloat32_t b)` | Xor Float Vector | Bitwise XOR on float vector |
| `vint_t` | `VEC_XOR_I(a, b)` | `(vint_t a, vint_t b)` | Xor Int Vector | Bitwise XOR on int vector |
| `vfloat32_t` | `VEC_NOT_F(a)` | `(vfloat32_t a)` | Not Float Vector | Bitwise NOT on float vector |
| `vint_t` | `VEC_NOT_I(a)` | `(vint_t a)` | Not Int Vector | Bitwise NOT on int vector |
| `void` | `VEC_SCATTER_F(dst, idx_vec, vec)` | `(float *dst, vint_t idx_vec, vfloat32_t vec)` | Scatter Float Vector | Scatter float vector to memory using index vector |
| `void` | `VEC_SCATTER_I(dst, idx_vec, vec)` | `(int *dst, vint_t idx_vec, vint_t vec)` | Scatter Int Vector | Scatter int vector to memory using index vector |
| `vfloat32_t` | `VEC_GATHER_F(src, idx_vec)` | `(const float *src, vint_t idx_vec)` | Gather Float Vector | Gather float elements from memory using index vector |
| `vint_t` | `VEC_GATHER_I(src, idx_vec)` | `(const int *src, vint_t idx_vec)` | Gather Int Vector | Gather int elements from memory using index vector |

## 3. Example
```C
#include <stdio.h>
#include <stdint.h>

#include "vectorize.h"

int main() {
    float arr1[8] = {1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0};
    float arr2[8] = {4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2};
    float result[8];

    int index_arr[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    vfloat32_t vec1 = VEC_LOAD_F(arr1);
    vfloat32_t vec2 = VEC_LOAD_F(arr2);
    vfloat32_t vec_result = VEC_ADD_F(vec1, vec2);

    vint_t index_vec = VEC_LOAD_I(index_arr);

    VEC_SCATTER_F(result, index_vec, vec_result);
    printf("Vector width: %d\n", VEC_WIDTH);

    puts("Result: ");
    for (int i = 0; i < 8; i++) {
        printf("%f ", result[i]);
    }
    puts("\n");
    return 0;
}
```

Output:
```powershell
PS E:\Git\luna-vectorize-calc-header> gcc ./test.c -o ./test.exe
PS E:\Git\luna-vectorize-calc-header> ./test.exe
Vector width: 4
Result:
5.200000 6.200000 7.200000 8.200000 0.000000 0.000000 0.000000 0.000000
```

Note: Elements [4] to [7] are not processed because **vector width is limited by the current platform**, causing **automatic truncation**.

