## Vectorize Calc Header
一个**单头文件**的轻量化SIMD向量运算支持库。
 * Author: 月と猫 - LunaNeko
 * Mainly assisted by GPT-5 

English doc: [README_EN.md](README_EN.md)

版本：v0.1

- WIP：对RISC-V平台的封装支持。（将在下个版本完成）

## 使用说明：

使用本库时直接导入即可。
```C
#include <vectorize.h>
```

## 1. 数据类型
头文件宏定义了以下数据内容：

| 宏名 | 类型 | 说明 |
| ---- | ----- | ---- |
| `vfloat32_t` | float SIMD 向量类型  | 单精度浮点向量类型，例如 AVX/SSE/NEON |
| `vint_t`| int SIMD 向量类型    | 整数向量类型                    |
| `vmask_t` | int / mask 向量类型  | 用于条件掩码运算                  |
| `VEC_WIDTH` | int | 当前平台下浮点向量宽度（单文件作用域，用户可通过接口访问）  |

## 2. 宏函数定义表

| 返回值 | 宏/函数名 | 参数 | 助记 | 函数作用 |
|--------|---------|------|------|----------|
| `vfloat32_t` | `VEC_LOAD_F(ptr)` | `(const float *ptr)` | Load Float Vector | 从内存加载浮点向量 |
| `vint_t` | `VEC_LOAD_I(ptr)` | `(const int *ptr)` | Load Int Vector | 从内存加载整数向量 |
| `void` | `VEC_STORE_F(ptr, vec)` | `(float *ptr, vfloat32_t vec)` | Store Float Vector | 将浮点向量存储到内存 |
| `void` | `VEC_STORE_I(ptr, vec)` | `(int *ptr, vint_t vec)` | Store Int Vector | 将整数向量存储到内存 |
| `vfloat32_t` | `VEC_ADD_F(a, b)` | `(vfloat32_t a, vfloat32_t b)` | Add Float Vector | 向量浮点加法 |
| `vint_t` | `VEC_ADD_I(a, b)` | `(vint_t a, vint_t b)` | Add Int Vector | 向量整数加法 |
| `vfloat32_t` | `VEC_SUB_F(a, b)` | `(vfloat32_t a, vfloat32_t b)` | Sub Float Vector | 向量浮点减法 |
| `vint_t` | `VEC_SUB_I(a, b)` | `(vint_t a, vint_t b)` | Sub Int Vector | 向量整数减法 |
| `vfloat32_t` | `VEC_MUL_F(a, b)` | `(vfloat32_t a, vfloat32_t b)` | Mul Float Vector | 向量浮点乘法 |
| `vint_t` | `VEC_MUL_I(a, b)` | `(vint_t a, vint_t b)` | Mul Int Vector | 向量整数乘法 |
| `vfloat32_t` | `VEC_DIV_F(a, b)` | `(vfloat32_t a, vfloat32_t b)` | Div Float Vector | 向量浮点除法 |
| `vfloat32_t` | `VEC_FMA_F(a, b, c)` | `(vfloat32_t a, vfloat32_t b, vfloat32_t c)` | Fused Multiply Add | 向量浮点 FMA: a*b + c |
| `vfloat32_t` | `VEC_AND_F(a, b)` | `(vfloat32_t a, vfloat32_t b)` | And Float Vector | 按位与浮点向量 |
| `vint_t` | `VEC_AND_I(a, b)` | `(vint_t a, vint_t b)` | And Int Vector | 按位与整数向量 |
| `vfloat32_t` | `VEC_OR_F(a, b)` | `(vfloat32_t a, vfloat32_t b)` | Or Float Vector | 按位或浮点向量 |
| `vint_t` | `VEC_OR_I(a, b)` | `(vint_t a, vint_t b)` | Or Int Vector | 按位或整数向量 |
| `vfloat32_t` | `VEC_XOR_F(a, b)` | `(vfloat32_t a, vfloat32_t b)` | Xor Float Vector | 按位异或浮点向量 |
| `vint_t` | `VEC_XOR_I(a, b)` | `(vint_t a, vint_t b)` | Xor Int Vector | 按位异或整数向量 |
| `vfloat32_t` | `VEC_NOT_F(a)` | `(vfloat32_t a)` | Not Float Vector | 按位取反浮点向量 |
| `vint_t` | `VEC_NOT_I(a)` | `(vint_t a)` | Not Int Vector | 按位取反整数向量 |
| `void` | `VEC_SCATTER_F(dst, idx_vec, vec)` | `(float *dst, vint_t idx_vec, vfloat32_t vec)` | Scatter Float Vector | 按索引向量将浮点向量散射存储到内存 |
| `void` | `VEC_SCATTER_I(dst, idx_vec, vec)` | `(int *dst, vint_t idx_vec, vint_t vec)` | Scatter Int Vector | 按索引向量将整数向量散射存储到内存 |
| `vfloat32_t` | `VEC_GATHER_F(src, idx_vec)` | `(const float *src, vint_t idx_vec)` | Gather Float Vector | 按索引向量从内存收集浮点元素 |
| `vint_t` | `VEC_GATHER_I(src, idx_vec)` | `(const int *src, vint_t idx_vec)` | Gather Int Vector | 按索引向量从内存收集整数元素 |

## 3. 使用例
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

输出：
```powershell
PS E:\Git\luna-vectorize-calc-header> gcc ./test.c -o ./test.exe
PS E:\Git\luna-vectorize-calc-header> ./test.exe
Vector width: 4
Result:
5.200000 6.200000 7.200000 8.200000 0.000000 0.000000 0.000000 0.000000
```

经查证，[4]到[7]位未被计算的原因是**向量宽度受当前平台限制**，导致向量被**自动截断**。
