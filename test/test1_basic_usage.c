#include <stdio.h>
#include <stdint.h>

#include "../vectorize.h"

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