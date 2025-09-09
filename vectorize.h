/*
 * _vectorize.h
 * Author: 月と猫 - LunaNeko
 * - Mainly assisted by GPT-5 
 *
 * 统一向量操作宏接口（以 VEC 开头），支持：
 *  - AVX (256-bit float32, VEC_WIDTH=8)
 *  - SSE (128-bit float32, VEC_WIDTH=4)
 *  - AVX-512 (掩码 load/store)
 *  - ARM NEON (基本支持)
 *  - RISC-V Vector (stub / 可扩展 - TODO：完成这个)
 *  - 标量回退 (float, VEC_WIDTH=1)
 *
 * 注：本头文件以 float32 为主（_F 后缀）。如需 double，可按同样模式扩展。
 * 
 * 警告：如果你要在你的RISC-V工程内使用向量化支持，你必须在你的编译器中打开RVV支持。例：`-march=rv64gcv`，
 *   或`-march=rv64imafdcv`
 */

#ifndef VECTORIZE_HEADER_H
#define VECTORIZE_HEADER_H

#define VECTORIZE_HEADER_H_VERSION 0.1

/* ---------- 平台检测与头文件包含 ---------- */

#if defined(__x86_64__) || defined(__i386__)
  /* 优先 AVX/AVX2/AVX-512，再 SSE */
  #if defined(__AVX512F__)
    #include <immintrin.h>
    #define VEC_IMPL_AVX512 1
    #define VEC_CALC_USABLE 1
  #elif defined(__AVX2__) || defined(__AVX__)
    #include <immintrin.h>
    #define VEC_IMPL_AVX 1
    #define VEC_CALC_USABLE 1
  #elif defined(__SSE__)
    #include <xmmintrin.h>
    #if defined(__SSE2__)
      #include <emmintrin.h>
    #endif
    #if defined(__SSE3__)
      #include <pmmintrin.h>
    #endif
    #if defined(__SSSE3__)
      #include <tmmintrin.h>
    #endif
    #if defined(__SSE4_1__)
      #include <smmintrin.h>
    #endif
    #define VEC_IMPL_SSE 1
    #define VEC_CALC_USABLE 1
  #endif
  /* FMA header is covered by immintrin.h when available */
#endif

/* ARM NEON */
#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
  #if !defined(VEC_IMPL_AVX) && !defined(VEC_IMPL_SSE)
    #include <arm_neon.h>
    #define VEC_IMPL_NEON 1
    #define VEC_CALC_USABLE 1
  #endif
#endif

/* RISC-V Vector (stub) */
#if defined(__riscv) && (defined(__riscv_vector) || defined(__riscv_v))
  #include <riscv_vector.h>
  #define VEC_IMPL_RISCV 1
  #define VEC_CALC_USABLE 1
#endif

/* 默认回退到标量实现（当没有任何向量实现被选中） */
#if !defined(VEC_IMPL_AVX512) && !defined(VEC_IMPL_AVX) && !defined(VEC_IMPL_SSE) && !defined(VEC_IMPL_NEON) && !defined(VEC_IMPL_RISCV)
  #define VEC_IMPL_SCALAR 1
  #define VEC_CALC_USABLE 0
#endif

/* ---------- 类型与宽度定义 ---------- */

#if defined(VEC_IMPL_AVX512)
  typedef __m512 vfloat32_t;       /* 16 x float32 */
  typedef __m512i vint_t;     /* integer version if needed */
  typedef __m512 vmask_t;     /* mask is also __m512 in some intrinsics, but AVX-512 has k-registers; we use intrinsics directly */
  #define VEC_WIDTH_F 16
#elif defined(VEC_IMPL_AVX)
  typedef __m256 vfloat32_t;        /* 8 x float32 */
  typedef __m256i vint_t;
  typedef __m256 vmask_t;
  #define VEC_WIDTH_F 8
#elif defined(VEC_IMPL_SSE)
  typedef __m128 vfloat32_t;        /* 4 x float32 */
  typedef __m128i vint_t;
  typedef __m128 vmask_t;
  #define VEC_WIDTH_F 4
#elif defined(VEC_IMPL_NEON)
  typedef float32x4_t vfloat32_t;   /* NEON 4 x float32 (ARMv8) */
  typedef int32x4_t vint_t;
  typedef uint32x4_t vmask_t;
  #define VEC_WIDTH_F 4
#elif defined(VEC_IMPL_RISCV)
  // RISC-V RVV扩展中的向量长度是动态长度，没有一个固定的vec类型。
  // 然而我现在还没想好到底要如何支持这个动态长度向量。
  #error "This library is currently NOT prepared for dynamic-length vector platform. Please use `#include <riscv_vector>` instead."
  typedef float vfloat32_t;
  typedef int vint_t;
  typedef int vmask_t;
  #define VEC_WIDTH_F 1
#else /* scalar */
  typedef float vfloat32_t;
  typedef int vint_t;
  typedef int vmask_t;
  #define VEC_WIDTH_F 1
#endif


/* ---------- 浮点数：set / zero / load / store（对齐与非对齐） ---------- */

/* set / zero, load / store (aligned + unaligned) */
#if defined(VEC_IMPL_AVX512)
  #define VEC_SET1_F(x) _mm512_set1_ps(x)
  #define VEC_SETZERO_F() _mm512_setzero_ps()
  #define VEC_LOADU_F(p) _mm512_loadu_ps((const float*)(p))
  #define VEC_LOAD_F(p)  _mm512_load_ps((const float*)(p))
  #define VEC_STOREU_F(p,v) _mm512_storeu_ps((float*)(p),(v))
  #define VEC_STORE_F(p,v)  _mm512_store_ps((float*)(p),(v))
#elif defined(VEC_IMPL_AVX)
  #define VEC_SET1_F(x) _mm256_set1_ps(x)
  #define VEC_SETZERO_F() _mm256_setzero_ps()
  #define VEC_LOADU_F(p) _mm256_loadu_ps((const float*)(p))
  #define VEC_LOAD_F(p)  _mm256_load_ps((const float*)(p))
  #define VEC_STOREU_F(p,v) _mm256_storeu_ps((float*)(p),(v))
  #define VEC_STORE_F(p,v)  _mm256_store_ps((float*)(p),(v))
#elif defined(VEC_IMPL_SSE)
  #define VEC_SET1_F(x) _mm_set1_ps(x)
  #define VEC_SETZERO_F() _mm_setzero_ps()
  #define VEC_LOADU_F(p) _mm_loadu_ps((const float*)(p))
  #define VEC_LOAD_F(p)  _mm_load_ps((const float*)(p))
  #define VEC_STOREU_F(p,v) _mm_storeu_ps((float*)(p),(v))
  #define VEC_STORE_F(p,v)  _mm_store_ps((float*)(p),(v))
#elif defined(VEC_IMPL_NEON)
  #define VEC_SET1_F(x) vdupq_n_f32(x)
  #define VEC_SETZERO_F() vdupq_n_f32(0.0f)
  #define VEC_LOADU_F(p) vld1q_f32((const float*)(p))
  /* NEON 不区分 load/store aligned/unaligned 在大多数实现上 */
  #define VEC_STOREU_F(p,v) vst1q_f32((float*)(p),(v))
  #define VEC_LOAD_F(p) VEC_LOADU_F(p)
  #define VEC_STORE_F(p,v) VEC_STOREU_F(p,v)
#elif defined(VEC_IMPL_RISCV) || defined(VEC_IMPL_SCALAR)
  #define VEC_SET1_F(x) (x)
  #define VEC_SETZERO_F() (0.0f)
  #define VEC_LOADU_F(p) (*(const float*)(p))
  #define VEC_LOAD_F(p)   VEC_LOADU_F(p)
  #define VEC_STOREU_F(p,v) (*(float*)(p) = (v))
  #define VEC_STORE_F(p,v)  VEC_STOREU_F(p,v)
#endif

/* ---------- 整数：set / zero / load / store ---------- */
#if defined(VEC_IMPL_AVX512)
  #define VEC_SET1_I(x) _mm512_set1_epi32(x)
  #define VEC_SETZERO_I() _mm512_setzero_si512()
  #define VEC_LOADU_I(p) _mm512_loadu_si512((const int*)(p))
  #define VEC_LOAD_I(p)  _mm512_load_si512((const int*)(p))
  #define VEC_STOREU_I(p,v) _mm512_storeu_si512((int*)(p),(v))
  #define VEC_STORE_I(p,v)  _mm512_store_si512((int*)(p),(v))
#elif defined(VEC_IMPL_AVX)
  #define VEC_SET1_I(x) _mm256_set1_epi32(x)
  #define VEC_SETZERO_I() _mm256_setzero_si256()
  #define VEC_LOADU_I(p) _mm256_loadu_si256((const __m256i*)(p))
  #define VEC_LOAD_I(p)  _mm256_load_si256((const __m256i*)(p))
  #define VEC_STOREU_I(p,v) _mm256_storeu_si256((__m256i*)(p),(v))
  #define VEC_STORE_I(p,v)  _mm256_store_si256((__m256i*)(p),(v))
#elif defined(VEC_IMPL_SSE)
  #define VEC_SET1_I(x) _mm_set1_epi32(x)
  #define VEC_SETZERO_I() _mm_setzero_si128()
  #define VEC_LOADU_I(p) _mm_loadu_si128((const __m128i*)(p))
  #define VEC_LOAD_I(p)  _mm_load_si128((const __m128i*)(p))
  #define VEC_STOREU_I(p,v) _mm_storeu_si128((__m128i*)(p),(v))
  #define VEC_STORE_I(p,v)  _mm_store_si128((__m128i*)(p),(v))
#elif defined(VEC_IMPL_NEON)
  #define VEC_SET1_I(x) vdupq_n_s32(x)
  #define VEC_SETZERO_I() vdupq_n_s32(0)
  #define VEC_LOADU_I(p) vld1q_s32((const int32_t*)(p))
  #define VEC_LOAD_I(p)  VEC_LOADU_I(p)  /* NEON 不区分对齐 */
  #define VEC_STOREU_I(p,v) vst1q_s32((int32_t*)(p),(v))
  #define VEC_STORE_I(p,v)  VEC_STOREU_I(p,v)
#elif defined(VEC_IMPL_RISCV) || defined(VEC_IMPL_SCALAR)
  #define VEC_SET1_I(x) (x)
  #define VEC_SETZERO_I() (0)
  #define VEC_LOADU_I(p) (*(const int32_t*)(p))
  #define VEC_LOAD_I(p)   VEC_LOADU_I(p)
  #define VEC_STOREU_I(p,v) (*(int32_t*)(p) = (v))
  #define VEC_STORE_I(p,v)  VEC_STOREU_I(p,v)
#endif


/* ---------- 基本算术 ：浮点数---------- */

#if defined(VEC_IMPL_AVX512)
  #define VEC_ADD_F(a,b) _mm512_add_ps((a),(b))
  #define VEC_SUB_F(a,b) _mm512_sub_ps((a),(b))
  #define VEC_MUL_F(a,b) _mm512_mul_ps((a),(b))
  #define VEC_DIV_F(a,b) _mm512_div_ps((a),(b))
  #define VEC_MAX_F(a,b) _mm512_max_ps((a),(b))
  #define VEC_MIN_F(a,b) _mm512_min_ps((a),(b))
  #define VEC_FLOOR_F(a) _mm512_floor_ps(a)
  #define VEC_MOD_F(a,b) \
    VEC_SUB_F((a), VEC_MUL_F((b), VEC_FLOOR_F(VEC_DIV_F((a), (b)))))
#elif defined(VEC_IMPL_AVX)
  #define VEC_ADD_F(a,b) _mm256_add_ps((a),(b))
  #define VEC_SUB_F(a,b) _mm256_sub_ps((a),(b))
  #define VEC_MUL_F(a,b) _mm256_mul_ps((a),(b))
  #define VEC_DIV_F(a,b) _mm256_div_ps((a),(b))
  #define VEC_MAX_F(a,b) _mm256_max_ps((a),(b))
  #define VEC_MIN_F(a,b) _mm256_min_ps((a),(b))
  #define VEC_FLOOR_F(a) _mm256_floor_ps(a)
  #define VEC_MOD_F(a,b) \
    VEC_SUB_F((a), VEC_MUL_F((b), VEC_FLOOR_F(VEC_DIV_F((a), (b)))))
#elif defined(VEC_IMPL_SSE)
  #define VEC_ADD_F(a,b) _mm_add_ps((a),(b))
  #define VEC_SUB_F(a,b) _mm_sub_ps((a),(b))
  #define VEC_MUL_F(a,b) _mm_mul_ps((a),(b))
  #define VEC_DIV_F(a,b) _mm_div_ps((a),(b))
  #define VEC_MAX_F(a,b) _mm_max_ps((a),(b))
  #define VEC_MIN_F(a,b) _mm_min_ps((a),(b))
  #define VEC_FLOOR_F(a) _mm_floor_ps(a)
  // 取模操作没有硬件支持，只能多步模拟。原理：(a % b) = a - (b * floor(a / b))
  #define VEC_MOD_F(a,b) \
    VEC_SUB_F((a), VEC_MUL_F((b), VEC_FLOOR_F(VEC_DIV_F((a), (b)))))
#elif defined(VEC_IMPL_NEON)
  #define VEC_ADD_F(a,b) vaddq_f32((a),(b))
  #define VEC_SUB_F(a,b) vsubq_f32((a),(b))
  #define VEC_MUL_F(a,b) vmulq_f32((a),(b))
  #define VEC_DIV_F(a,b) vdivq_f32((a),(b)) /* vdivq_f32 requires VFP/Neon FP support */
  #define VEC_MAX_F(a,b) vmaxq_f32((a),(b))
  #define VEC_MIN_F(a,b) vminq_f32((a),(b))
  #define VEC_FLOOR_F(a) floorq_f32(a)  // 下方为自定义实现，由于硬件不支持floor操作，只能靠多步模拟进行。
static inline float32x4_t floorq_f32(float32x4_t a) {
    int32x4_t i = vcvtq_s32_f32(a);                // 向零取整
    float32x4_t fi = vcvtq_f32_s32(i);             // 转回浮点
    uint32x4_t mask = vcgtq_f32(fi, a);            // 如果 fi > a，说明 a 是负数且被截断
    float32x4_t one = vdupq_n_f32(1.0f);
    return vbslq_f32(mask, vsubq_f32(fi, one), fi);
}
  #define VEC_MOD_F(a,b) \
    VEC_SUB_F((a), VEC_MUL_F((b), VEC_FLOOR_F(VEC_DIV_F((a), (b)))))
#else
  #define VEC_ADD_F(a,b) ((a)+(b))
  #define VEC_SUB_F(a,b) ((a)-(b))
  #define VEC_MUL_F(a,b) ((a)*(b))
  #define VEC_DIV_F(a,b) ((a)/(b))
  #define VEC_MAX_F(a,b) (((a)>(b))?(a):(b))
  #define VEC_MIN_F(a,b) (((a)<(b))?(a):(b))
  #define VEC_FLOOR_F(a) floorf(a)
  #define VEC_MOD_F(a,b) \
    VEC_SUB_F((a), VEC_MUL_F((b), VEC_FLOOR_F(VEC_DIV_F((a), (b)))))
#endif

/* ---------- 基本算术 ：整数 ---------- */
#if defined(VEC_IMPL_AVX512)
  #define VEC_ADD_I(a,b) _mm512_add_epi32((a),(b))
  #define VEC_SUB_I(a,b) _mm512_sub_epi32((a),(b))
  #define VEC_MUL_I(a,b) _mm512_mullo_epi32((a),(b))
  /* 整除/取模没有硬件指令，需要逐元素处理或回退 */
  #define VEC_DIV_I(a,b) VEC_DIV_I_SCALAR(a,b)
  #define VEC_MOD_I(a,b) VEC_MOD_I_SCALAR(a,b)
#elif defined(VEC_IMPL_AVX)
  #define VEC_ADD_I(a,b) _mm256_add_epi32((a),(b))
  #define VEC_SUB_I(a,b) _mm256_sub_epi32((a),(b))
  #define VEC_MUL_I(a,b) _mm256_mullo_epi32((a),(b))
  #define VEC_DIV_I(a,b) VEC_DIV_I_SCALAR(a,b)
  #define VEC_MOD_I(a,b) VEC_MOD_I_SCALAR(a,b)
#elif defined(VEC_IMPL_SSE)
  #define VEC_ADD_I(a,b) _mm_add_epi32((a),(b))
  #define VEC_SUB_I(a,b) _mm_sub_epi32((a),(b))
  #define VEC_MUL_I(a,b) _mm_mullo_epi32((a),(b))
  #define VEC_DIV_I(a,b) VEC_DIV_I_SCALAR(a,b)
  #define VEC_MOD_I(a,b) VEC_MOD_I_SCALAR(a,b)
#elif defined(VEC_IMPL_NEON)
  #define VEC_ADD_I(a,b) vaddq_s32((a),(b))
  #define VEC_SUB_I(a,b) vsubq_s32((a),(b))
  #define VEC_MUL_I(a,b) vmulq_s32((a),(b))
  /* NEON 没有整除/取模指令 */
  #define VEC_DIV_I(a,b) VEC_DIV_I_SCALAR(a,b)
  #define VEC_MOD_I(a,b) VEC_MOD_I_SCALAR(a,b)
}
#else
  #define VEC_ADD_I(a,b) ((a)+(b))
  #define VEC_SUB_I(a,b) ((a)-(b))
  #define VEC_MUL_I(a,b) ((a)*(b))
  #define VEC_DIV_I(a,b) ((a)/(b))
  #define VEC_MOD_I(a,b) ((a)%(b))
#endif

/* ---------- fallback for div/mod - 部分指令集无该指令，提供回退解决方案 ---------- */
static inline vint_t VEC_DIV_I_SCALAR(vint_t a, vint_t b) {
#if defined(VEC_IMPL_SSE) || defined(VEC_IMPL_AVX) || defined(VEC_IMPL_AVX512) || defined(VEC_IMPL_NEON)
    /* 假设 vec_t 是 __m128i / __m256i / __m512i / int32x4_t */
    int tmp[16]; /* 最多支持512位 / 32位整数 */
    int res[16];
    int n = sizeof(a) / sizeof(int);  /* 注意：int的长度在不同平台上不一致 */
    VEC_STOREU_I(tmp, a);
    int tmpb[16];
    VEC_STOREU_I(tmpb, b);
    for(int i=0;i<n;i++) res[i] = tmp[i]/tmpb[i];
    return VEC_LOADU_I(res);
#else
    return a / b;
#endif
}

static inline vint_t VEC_MOD_I_SCALAR(vint_t a, vint_t b) {
#if defined(VEC_IMPL_SSE) || defined(VEC_IMPL_AVX) || defined(VEC_IMPL_AVX512) || defined(VEC_IMPL_NEON)
    int tmp[16]; 
    int res[16];
    int n = sizeof(a) / sizeof(int);
    VEC_STOREU_I(tmp, a);
    int tmpb[16];
    VEC_STOREU_I(tmpb, b);
    for(int i=0;i<n;i++) res[i] = tmp[i]%tmpb[i];
    return VEC_LOADU_I(res);
#else
    return a % b;
#endif
}

/* --------------- vint_t和vfloat_t之间互相转换 ---------------- */
/* x86 SSE/AVX */
#if defined(VEC_IMPL_AVX)
  #define VEC_F2I(a) _mm256_cvttps_epi32(a)     // 浮点->int（向量，截断）
  #define VEC_I2F(a) _mm256_cvtepi32_ps(a)      // 整数->float（向量）
#elif defined(VEC_IMPL_SSE)
  #define VEC_F2I(a) _mm_cvttps_epi32(a)
  #define VEC_I2F(a) _mm_cvtepi32_ps(a)
#elif defined(VEC_IMPL_AVX512)
  #define VEC_F2I(a) _mm512_cvttps_epi32(a)     // 或者其他 AVX-512 variant
  #define VEC_I2F(a) _mm512_cvtepi32_ps(a)
#elif defined(VEC_IMPL_NEON)
/* ARM NEON */
  #define VEC_F2I(a) vcvtq_s32_f32(a)            // 浮点->int
  #define VEC_I2F(a) vcvtq_f32_s32(a)            // 整数->浮点（假定存在）
#else
  // 回退到标量
  #define VEC_F2I(a) (int)(a)
  #define VEC_I2F(a) (float)(a)
#endif

/* ---------- FMA (a*b + c) 支持vint_t和vfloat_t ---------- */
#if (defined(__FMA__) || defined(__AVX2__) || defined(__AVX__))

  /* 如果编译器支持 FMA intrinsic */
  #if defined(VEC_IMPL_AVX512)
    #define VEC_FMA_F(a,b,c) _mm512_fmadd_ps((a),(b),(c))
    #define VEC_FMA_I(a,b,c) _mm512_add_epi32(_mm512_mullo_epi32((a),(b)), (c))
  #elif defined(VEC_IMPL_AVX)
    #define VEC_FMA_F(a,b,c) _mm256_fmadd_ps((a),(b),(c))
    #define VEC_FMA_I(a,b,c) _mm256_add_epi32(_mm256_mullo_epi32((a),(b)), (c))
  #elif defined(VEC_IMPL_SSE) && defined(__FMA__)
    #define VEC_FMA_F(a,b,c) _mm_fmadd_ps((a),(b),(c))
    #define VEC_FMA_I(a,b,c) _mm_add_epi32(_mm_mullo_epi32((a),(b)), (c))
  #else
    /* fallback */
    #define VEC_FMA_F(a,b,c) (VEC_ADD_F(VEC_MUL_F((a),(b)), (c)))
    #define VEC_FMA_I(a,b,c) (VEC_ADD_I(VEC_MUL_I((a),(b)), (c)))
  #endif

#else
  /* fallback */
  #define VEC_FMA_F(a,b,c) (VEC_ADD_F(VEC_MUL_F((a),(b)), (c)))
  #define VEC_FMA_I(a,b,c) (VEC_ADD_I(VEC_MUL_I((a),(b)), (c)))

#endif


/* ---------- sqrt / rcp / rsqrt 仅支持float版本。 ---------- */
#if defined(VEC_IMPL_AVX512)
  #define VEC_SQRT_F(a) _mm512_sqrt_ps(a)
  #define VEC_RSQRT_F(a) _mm512_rsqrt14_ps(a) /* approx if available; otherwise use division */
  #define VEC_RCP_F(a) _mm512_rcp14_ps(a)     /* approx */
#elif defined(VEC_IMPL_AVX)
  #define VEC_SQRT_F(a) _mm256_sqrt_ps(a)
  #define VEC_RSQRT_F(a) _mm256_rsqrt_ps(a)
  #define VEC_RCP_F(a) _mm256_rcp_ps(a)
#elif defined(VEC_IMPL_SSE)
  #define VEC_SQRT_F(a) _mm_sqrt_ps(a)
  #define VEC_RSQRT_F(a) _mm_rsqrt_ps(a)
  #define VEC_RCP_F(a) _mm_rcp_ps(a)
#elif defined(VEC_IMPL_NEON)
  #define VEC_SQRT_F(a) vsqrtq_f32(a) /* may require vfp */
  #define VEC_RSQRT_F(a) vrsqrteq_f32(a) /* Newton iterations recommended */
  #define VEC_RCP_F(a) vrecpeq_f32(a)    /* Newton iterations recommended */
#else
  #define VEC_SQRT_F(a) (sqrtf(a))
  #define VEC_RSQRT_F(a) (1.0f / sqrtf(a))
  #define VEC_RCP_F(a) (1.0f / (a))
#endif

/* ---------- 位运算：AND / OR / XOR / NOT (vfloat_t) ---------- */
#if defined(VEC_IMPL_AVX512)
  #define VEC_AND_F(a,b) _mm512_and_ps((a),(b))
  #define VEC_OR_F(a,b)  _mm512_or_ps((a),(b))
  #define VEC_XOR_F(a,b) _mm512_xor_ps((a),(b))
  #define VEC_NOT_F(a)   _mm512_xor_ps((a), _mm512_castsi512_ps(_mm512_set1_epi32(-1)))
#elif defined(VEC_IMPL_AVX)
  #define VEC_AND_F(a,b) _mm256_and_ps((a),(b))
  #define VEC_OR_F(a,b)  _mm256_or_ps((a),(b))
  #define VEC_XOR_F(a,b) _mm256_xor_ps((a),(b))
  #define VEC_NOT_F(a)   _mm256_xor_ps((a), _mm256_castsi256_ps(_mm256_set1_epi32(-1)))
#elif defined(VEC_IMPL_SSE)
  #define VEC_AND_F(a,b) _mm_and_ps((a),(b))
  #define VEC_OR_F(a,b)  _mm_or_ps((a),(b))
  #define VEC_XOR_F(a,b) _mm_xor_ps((a),(b))
  /* NOT: xor with all-ones */
  #define VEC_NOT_F(a)   _mm_xor_ps((a), _mm_castsi128_ps(_mm_set1_epi32(-1)))
#elif defined(VEC_IMPL_NEON)
  #define VEC_AND_F(a,b) vandq_u32((a),(b)) /* careful: NEON types differ; user must use cast */
  #define VEC_OR_F(a,b)  vorrq_u32((a),(b))
  #define VEC_XOR_F(a,b) veorq_u32((a),(b))
  #define VEC_NOT_F(a)   vmvnq_u32((a))
#else
  #define VEC_AND_F(a,b) ((a)*(b)) /* not meaningful for float; keep for API completeness */
  #define VEC_OR_F(a,b)  ((a)+(b))
  #define VEC_XOR_F(a,b) ((int)(a) ^ (int)(b))
  #define VEC_NOT_F(a)   (~(int)(a))
#endif

/* ---------- 位运算：AND / OR / XOR / NOT (vint_t) ---------- */
#if defined(VEC_IMPL_AVX512)
  #define VEC_AND_I(a,b) _mm512_and_si512((a),(b))
  #define VEC_OR_I(a,b)  _mm512_or_si512((a),(b))
  #define VEC_XOR_I(a,b) _mm512_xor_si512((a),(b))
  #define VEC_NOT_I(a)   _mm512_xor_si512((a), _mm512_set1_epi32(-1))
#elif defined(VEC_IMPL_AVX)
  #define VEC_AND_I(a,b) _mm256_and_si256((a),(b))
  #define VEC_OR_I(a,b)  _mm256_or_si256((a),(b))
  #define VEC_XOR_I(a,b) _mm256_xor_si256((a),(b))
  #define VEC_NOT_I(a)   _mm256_xor_si256((a), _mm256_set1_epi32(-1))
#elif defined(VEC_IMPL_SSE)
  #define VEC_AND_I(a,b) _mm_and_si128((a),(b))
  #define VEC_OR_I(a,b)  _mm_or_si128((a),(b))
  #define VEC_XOR_I(a,b) _mm_xor_si128((a),(b))
  #define VEC_NOT_I(a)   _mm_xor_si128((a), _mm_set1_epi32(-1))
#elif defined(VEC_IMPL_NEON)
  #define VEC_AND_I(a,b) vandq_u32((a),(b))
  #define VEC_OR_I(a,b)  vorrq_u32((a),(b))
  #define VEC_XOR_I(a,b) veorq_u32((a),(b))
  #define VEC_NOT_I(a)   vmvnq_u32((a))
#else
  #define VEC_AND_I(a,b) ((a) & (b))
  #define VEC_OR_I(a,b)  ((a) | (b))
  #define VEC_XOR_I(a,b) ((a) ^ (b))
  #define VEC_NOT_I(a)   (~(a))
#endif


/* ---------- 比较操作（返回 mask-style vectors） ---------- */
/* SSE 提供一系列比较 intrinsic；AVX 使用 _mm256_cmp_ps(a,b,imm) */
/* 我们映射常见的： ==, !=, <, <=, >, >=, ord, unord
   并提供 nlt/nle/ngt/nge 基于现有比较的回退实现（与 AVX imm8 行为对齐或等价）。 */

#if defined(VEC_IMPL_AVX512)
  /* AVX-512 的比较可用 _mm512_cmp_ps_mask (返回 k-mask)，但这里保持返回向量掩码兼容性 */
  #define VEC_CMPEQ_F(a,b)    _mm512_cmp_ps_mask((a),(b), _CMP_EQ_OQ)
  #define VEC_CMPNEQ_F(a,b)   _mm512_cmp_ps_mask((a),(b), _CMP_NEQ_UQ)
  #define VEC_CMPLT_F(a,b)    _mm512_cmp_ps_mask((a),(b), _CMP_LT_OQ)
  #define VEC_CMPLE_F(a,b)    _mm512_cmp_ps_mask((a),(b), _CMP_LE_OQ)
  #define VEC_CMPGT_F(a,b)    _mm512_cmp_ps_mask((a),(b), _CMP_GT_OQ)
  #define VEC_CMPGE_F(a,b)    _mm512_cmp_ps_mask((a),(b), _CMP_GE_OQ)
  #define VEC_CMPORD_F(a,b)   _mm512_cmp_ps_mask((a),(b), _CMP_ORD_Q)
  #define VEC_CMPUNORD_F(a,b) _mm512_cmp_ps_mask((a),(b), _CMP_UNORD_Q)

  /* not-less etc. map to corresponding imm8 */
  #define VEC_CMPNLT_F(a,b)   _mm512_cmp_ps_mask((a),(b), _CMP_NLT_UQ)
  #define VEC_CMPNLE_F(a,b)   _mm512_cmp_ps_mask((a),(b), _CMP_NLE_UQ)
  #define VEC_CMPNGT_F(a,b)   _mm512_cmp_ps_mask((a),(b), _CMP_NGT_UQ)
  #define VEC_CMPNGE_F(a,b)   _mm512_cmp_ps_mask((a),(b), _CMP_NGE_UQ)

#elif defined(VEC_IMPL_AVX)
  /* AVX2: use _mm256_cmp_ps with imm8 */
  #define VEC_CMPEQ_F(a,b)    _mm256_cmp_ps((a),(b), _CMP_EQ_OQ)
  #define VEC_CMPNEQ_F(a,b)   _mm256_cmp_ps((a),(b), _CMP_NEQ_UQ)
  #define VEC_CMPLT_F(a,b)    _mm256_cmp_ps((a),(b), _CMP_LT_OQ)
  #define VEC_CMPLE_F(a,b)    _mm256_cmp_ps((a),(b), _CMP_LE_OQ)
  #define VEC_CMPGT_F(a,b)    _mm256_cmp_ps((a),(b), _CMP_GT_OQ)
  #define VEC_CMPGE_F(a,b)    _mm256_cmp_ps((a),(b), _CMP_GE_OQ)
  #define VEC_CMPORD_F(a,b)   _mm256_cmp_ps((a),(b), _CMP_ORD_Q)
  #define VEC_CMPUNORD_F(a,b) _mm256_cmp_ps((a),(b), _CMP_UNORD_Q)

  #define VEC_CMPNLT_F(a,b)   _mm256_cmp_ps((a),(b), _CMP_NLT_UQ)
  #define VEC_CMPNLE_F(a,b)   _mm256_cmp_ps((a),(b), _CMP_NLE_UQ)
  #define VEC_CMPNGT_F(a,b)   _mm256_cmp_ps((a),(b), _CMP_NGT_UQ)
  #define VEC_CMPNGE_F(a,b)   _mm256_cmp_ps((a),(b), _CMP_NGE_UQ)

#elif defined(VEC_IMPL_SSE)
  /* SSE: use explicit intrinsics (returns __m128 masks 0xFFFFFFFF or 0) */
  #define VEC_CMPEQ_F(a,b)    _mm_cmpeq_ps((a),(b))
  #define VEC_CMPNEQ_F(a,b)   _mm_cmpneq_ps((a),(b))
  #define VEC_CMPLT_F(a,b)    _mm_cmplt_ps((a),(b))
  #define VEC_CMPLE_F(a,b)    _mm_cmple_ps((a),(b))
  #define VEC_CMPGT_F(a,b)    _mm_cmpgt_ps((a),(b))
  #define VEC_CMPGE_F(a,b)    _mm_cmpge_ps((a),(b))
  #define VEC_CMPORD_F(a,b)   _mm_cmpord_ps((a),(b))
  #define VEC_CMPUNORD_F(a,b) _mm_cmpunord_ps((a),(b))

  /* SSE 没有明确命名的 cmpnlt/_cmpnle/_cmpngt/_cmpnge 在有些头文件里存在（如 xmmintrin），
     但如果不可用，我们提供基于现有比较的等价回退：
     cmpnlt (not less-than) := not (a < b) -> 等同于 _mm_cmpge_ps(a,b) 在有序场景（针对 NaN 语义，AVX imm8 更精确）
  */
  #define VEC_CMPNLT_F(a,b)   _mm_cmpge_ps((a),(b))
  #define VEC_CMPNLE_F(a,b)   _mm_cmpgt_ps((a),(b))
  #define VEC_CMPNGT_F(a,b)   _mm_cmple_ps((a),(b))
  #define VEC_CMPNGE_F(a,b)   _mm_cmplt_ps((a),(b))

#elif defined(VEC_IMPL_NEON)
  /* NEON 没有直接等价的全位掩码 float 比较，使用 vcle/vclt 等返回 uint32 masks */
  #define VEC_CMPEQ_F(a,b)    vreinterpretq_u32_f32(vceqq_f32((a),(b)))
  #define VEC_CMPNEQ_F(a,b)   vmvnq_u32(vceqq_f32((a),(b)))
  #define VEC_CMPLT_F(a,b)    vreinterpretq_u32_f32(vcltq_f32((a),(b)))
  #define VEC_CMPLE_F(a,b)    vreinterpretq_u32_f32(vcleq_f32((a),(b)))
  #define VEC_CMPGT_F(a,b)    vreinterpretq_u32_f32(vcgtq_f32((a),(b)))
  #define VEC_CMPGE_F(a,b)    vreinterpretq_u32_f32(vcgeq_f32((a),(b)))
  #define VEC_CMPORD_F(a,b)   /* no direct, fallback to true mask */ vreinterpretq_u32_u32(vdupq_n_u32(0xFFFFFFFF))
  #define VEC_CMPUNORD_F(a,b) /* no direct, fallback to zero */ vreinterpretq_u32_u32(vdupq_n_u32(0x0))

  #define VEC_CMPNLT_F(a,b)   VEC_CMPGE_F(a,b)
  #define VEC_CMPNLE_F(a,b)   VEC_CMPGT_F(a,b)
  #define VEC_CMPNGT_F(a,b)   VEC_CMPLE_F(a,b)
  #define VEC_CMPNGE_F(a,b)   VEC_CMPLT_F(a,b)
#else
  /* scalar fallback: return 0xFFFFFFFF or 0x0 encoded in int mask */
  #define VEC_CMPEQ_F(a,b)    ((a)==(b)?~0u:0u)
  #define VEC_CMPNEQ_F(a,b)   ((a)!=(b)?~0u:0u)
  #define VEC_CMPLT_F(a,b)    ((a)<(b)?~0u:0u)
  #define VEC_CMPLE_F(a,b)    ((a)<=(b)?~0u:0u)
  #define VEC_CMPGT_F(a,b)    ((a)>(b)?~0u:0u)
  #define VEC_CMPGE_F(a,b)    ((a)>=(b)?~0u:0u)
  #define VEC_CMPORD_F(a,b)   ((!(isnan(a)||isnan(b)))?~0u:0u)
  #define VEC_CMPUNORD_F(a,b) ((isnan(a)||isnan(b))?~0u:0u)

  #define VEC_CMPNLT_F(a,b)   VEC_CMPGE_F(a,b)
  #define VEC_CMPNLE_F(a,b)   VEC_CMPGT_F(a,b)
  #define VEC_CMPNGT_F(a,b)   VEC_CMPLE_F(a,b)
  #define VEC_CMPNGE_F(a,b)   VEC_CMPLT_F(a,b)
#endif

/* --------------------- GATHER / SCATTER ------------------------------ */
/*
 * vfloat32_t VEC_GATHER_F(base, idx_vec)
 *   - base (float*): 指针类型，指向被读取的数组头部。实际上是一个float[]。
 *   - idx_vec (vint_t): 用于提供索引。
 * 该函数将返回一个和idx_vec等长的向量。
 *
 * void VEC_SCATTER_F(base, idx_vec, vals)
 *   - base (float*): 指针类型，指向被写入的数组头部。实际上是一个float[]。
 *   - idx_vec (vint_t): 用于提供索引。
 *   - vals (vfloat_t): 用于提供向量值。
 * 将vals[idx_vec[i]]的值储存到base[i]。
 * 
 * TODO：实现int版的计算。
 */

/* AVX2: use _mm256_i32gather_ps (indices as __m256i) */
#if defined(VEC_IMPL_AVX) && defined(__AVX2__)
static inline vfloat32_t VEC_GATHER_F(const float* base, vint_t idx) {
  return _mm256_i32gather_ps(base, idx, 4);
}
static inline void VEC_SCATTER_F(float* base, vint_t idx, vfloat32_t vals) {
  /* AVX2 has no scatter; fallback to element-wise extract/store */
  int indices[8];
  _mm256_storeu_si256((__m256i*)indices, idx);
  float vbuf[8];
  _mm256_storeu_ps(vbuf, vals);
  for (int k = 0; k < 8; ++k) base[indices[k]] = vbuf[k];
}

#elif defined(VEC_IMPL_AVX512)
/* AVX-512: use gather/scatter intrinsics */
static inline vfloat32_t VEC_GATHER_F(const float* base, vint_t idx) {
  return _mm512_i32gather_ps((const void*)base, idx, 4);
}
static inline void VEC_SCATTER_F(float* base, vint_t idx, vfloat32_t vals) {
  _mm512_i32scatter_ps((void*)base, idx, vals, 4);
}

#elif defined(VEC_IMPL_SSE)
/* SSE: no native gather/scatter, implement element-wise for 4 lanes */
static inline vfloat32_t VEC_GATHER_F(const float* base, vint_t idx) {
  int indices[4];
  _mm_storeu_si128((__m128i*)indices, idx);
  float out[4];
  for (int k = 0; k < 4; ++k) out[k] = base[indices[k]];
  return _mm_loadu_ps(out);
}
static inline void VEC_SCATTER_F(float* base, vint_t idx, vfloat32_t vals) {
  int indices[4];
  _mm_storeu_si128((__m128i*)indices, idx);
  float vbuf[4];
  _mm_storeu_ps(vbuf, vals);
  for (int k = 0; k < 4; ++k) base[indices[k]] = vbuf[k];
}

#elif defined(VEC_IMPL_NEON)
/* NEON: emulate by extracting indices and values */
static inline vfloat32_t VEC_GATHER_F(const float* base, vint_t idx) {
  int indices[4];
  vst1q_s32(indices, idx);
  float out[4];
  for (int k = 0; k < 4; ++k) out[k] = base[indices[k]];
  return vld1q_f32(out);
}
static inline void VEC_SCATTER_F(float* base, vint_t idx, vfloat32_t vals) {
  int indices[4];
  vst1q_s32(indices, idx);
  float vbuf[4];
  vst1q_f32(vbuf, vals);
  for (int k = 0; k < 4; ++k) base[indices[k]] = vbuf[k];
}

#else
/* Scalar fallback: indices given as plain int (single-lane) */
static inline float VEC_GATHER_F(const float* base, int idx) {
  return base[idx];
}
static inline void VEC_SCATTER_F(float* base, int idx, float val) {
  base[idx] = val;
}

#endif

/* Convenience wrapper: gather from float-index vector by converting to int indices */
static inline vfloat32_t VEC_GATHER_FROM_F(const float* base, vfloat32_t idx_f) {
#if defined(VEC_IMPL_AVX) || defined(VEC_IMPL_AVX512) || defined(VEC_IMPL_SSE) || defined(VEC_IMPL_NEON)
  vint_t idx_i = VEC_F2I(idx_f);
  return VEC_GATHER_F(base, idx_i);
#else
  int i = (int)idx_f;
  return VEC_GATHER_F(base, i);
#endif
}

/* Normalize integer index vector: (idx & 255) + base */
static inline vint_t VEC_NORMALIZE_INDEX(vint_t idx, int base) {
#if defined(VEC_IMPL_AVX512)
  const __m512i mask = _mm512_set1_epi32(255);
  __m512i t = _mm512_and_si512(idx, mask);
  __m512i b = _mm512_set1_epi32(base);
  return _mm512_add_epi32(t, b);
#elif defined(VEC_IMPL_AVX) && defined(__AVX2__)
  const __m256i mask = _mm256_set1_epi32(255);
  __m256i t = _mm256_and_si256(idx, mask);
  __m256i b = _mm256_set1_epi32(base);
  return _mm256_add_epi32(t, b);
#elif defined(VEC_IMPL_SSE)
  const __m128i mask = _mm_set1_epi32(255);
  __m128i t = _mm_and_si128(idx, mask);
  __m128i b = _mm_set1_epi32(base);
  return _mm_add_epi32(t, b);
#elif defined(VEC_IMPL_NEON)
  const int32x4_t mask = vdupq_n_s32(255);
  int32x4_t t = vandq_s32(idx, mask);
  int32x4_t b = vdupq_n_s32(base);
  return vaddq_s32(t, b);
#else
  /* scalar fallback: assume idx is int */
  idx = (idx & 255) + base;
  return idx;
#endif
}

/* ---------- 条件选择 / blend（按 mask 的符号位或 bitmask 选择） ---------- */
/* 我们提供统一的 VEC_SELECT(mask, a, b) 接口：
   - 在 x86 SSE/AVX 上，mask 是比较结果（0xFFFFFFFF 或 0x0 per-lane）
     _mm_blendv_ps 使用 mask 的 sign-bit 来选择；比较结果恰好适配。
   - 在 AVX-512，我们使用专用的掩码选择或者用 merge 指令。
   - 在 NEON，需使用 vbslq_f32（位选择）。
   - 在 scalar fallback：mask 非零 -> pick b else a.
*/

#if defined(VEC_IMPL_AVX512)
  /* AVX-512: 使用 _mm512_mask_blend_ps 接口（k-mask 需要单独处理，复杂） */
  /* 提供一个向量 merge（这里用 bitwise blend as fallback to be consistent） */
  #define VEC_SELECT(mask,a,b) _mm512_mask_blend_ps((mask),(a),(b))
#elif defined(VEC_IMPL_AVX)
  #if defined(__SSE4_1__) || defined(__AVX__)
    #define VEC_SELECT(mask,a,b) _mm256_blendv_ps((a),(b),(mask))
  #else
    #define VEC_SELECT(mask,a,b) _mm256_or_ps(_mm256_and_ps((mask),(b)), _mm256_andnot_ps((mask),(a)))
  #endif
#elif defined(VEC_IMPL_SSE)
  #if defined(__SSE4_1__)
    #define VEC_SELECT(mask,a,b) _mm_blendv_ps((a),(b),(mask))
  #else
    #define VEC_SELECT(mask,a,b) _mm_or_ps(_mm_and_ps((mask),(b)), _mm_andnot_ps((mask),(a)))
  #endif
#elif defined(VEC_IMPL_NEON)
  /* NEON: use vbslq_f32 expecting mask as uint32x4_t (1-bits per lane) */
  #define VEC_SELECT(mask,a,b) vreinterpretq_f32_u32(vbslq_u32((mask), vreinterpretq_u32_f32(b), vreinterpretq_u32_f32(a)))
#else
  #define VEC_SELECT(mask,a,b) ((mask) ? (b) : (a))
#endif


/* Gather unsigned 8-bit entries into integer vector (zero-extended) */
static inline vint_t VEC_GATHER_U8(const unsigned char* base, vint_t idx) {
#if defined(VEC_IMPL_AVX) && defined(__AVX2__)
  int indices[8]; _mm256_storeu_si256((__m256i*)indices, idx);
  int out[8]; for (int k=0;k<8;k++) out[k] = (int)base[indices[k]];
  return _mm256_loadu_si256((__m256i*)out);
#elif defined(VEC_IMPL_AVX512)
  int indices[16]; _mm512_storeu_si512((__m512i*)indices, idx);
  int out[16]; for (int k=0;k<16;k++) out[k] = (int)base[indices[k]];
  return _mm512_loadu_si512((__m512i*)out);
#elif defined(VEC_IMPL_SSE)
  int indices[4]; _mm_storeu_si128((__m128i*)indices, idx);
  int out[4]; for (int k=0;k<4;k++) out[k] = (int)base[indices[k]];
  return _mm_loadu_si128((__m128i*)out);
#elif defined(VEC_IMPL_NEON)
  int indices[4]; vst1q_s32(indices, idx);
  int out[4]; for (int k=0;k<4;k++) out[k] = (int)base[indices[k]];
  return vld1q_s32(out);
#else
  return (int)base[idx];
#endif
}

/* ---------- Masked load/store (AVX-512 实现，其他平台提供回退实现) ---------- */
#if defined(VEC_IMPL_AVX512)
  /* AVX-512: 使用 mask 参数 k (k-register)，但为兼容我们使用 _mm512_mask_loadu_ps/_mm512_mask_storeu_ps */
  #define VEC_MASK_LOADU_F(dst, mask, src)  dst = _mm512_mask_loadu_ps(dst, mask, (const float*)(src))
  #define VEC_MASK_STOREU_F(dst, mask, src) _mm512_mask_storeu_ps((float*)(dst), mask, src)
#else
  /* 非 AVX-512：回退实现：load then select per-lane (可能成本高，但保证正确) */
  #define VEC_MASK_LOADU_F(dst, mask, src) do { \
      vvfloat32_t _tmp = VEC_LOADU_F(src); \
      dst = VEC_SELECT((mask), dst, _tmp); \
  } while(0)

  #define VEC_MASK_STOREU_F(dst, mask, src) do { \
      vvfloat32_t _cur = VEC_LOADU_F(dst); \
      vvfloat32_t _merged = VEC_SELECT((mask), _cur, (src)); \
      VEC_STOREU_F((dst), _merged); \
  } while(0)
#endif

/* ---------- 其它辅助宏 ---------- */
#define VEC_WIDTH VEC_WIDTH_F

/* 将比较 mask 转换为 1.0f/0.0f 布尔向量（按位与 1.0f） */
#if defined(VEC_IMPL_AVX512)
  #define VEC_MASK_TO_BOOL_F(mask) _mm512_maskz_mov_ps(mask, _mm512_set1_ps(1.0f))
#elif defined(VEC_IMPL_AVX)
  #define VEC_MASK_TO_BOOL_F(mask) _mm256_and_ps((mask), _mm256_set1_ps(1.0f))
#elif defined(VEC_IMPL_SSE)
  #define VEC_MASK_TO_BOOL_F(mask) _mm_and_ps((mask), _mm_set1_ps(1.0f))
#elif defined(VEC_IMPL_NEON)
  #define VEC_MASK_TO_BOOL_F(mask) vreinterpretq_f32_u32(vandq_u32((mask), vreinterpretq_u32_f32(vdupq_n_f32(1.0f))))
#else
  #define VEC_MASK_TO_BOOL_F(mask) ((mask)?1.0f:0.0f)
#endif

/* 将向量重新解释为：float*（谨慎使用） */
#if defined(VEC_IMPL_AVX) || defined(VEC_IMPL_AVX512) || defined(VEC_IMPL_SSE)
  #define VEC_AS_FLOAT_PTR(v) ((const float*)&(v))
#else
  #define VEC_AS_FLOAT_PTR(v) (&(v))
#endif

/* ---------- 结束 ---------- */
#endif /* VECTORIZE_HEADER_H */
