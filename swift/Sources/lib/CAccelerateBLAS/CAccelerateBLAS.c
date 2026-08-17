// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

#include "CAccelerateBLAS.h"

#include <Accelerate/Accelerate.h>

void caccel_sgemv_rowmajor(int32_t rows, int32_t cols,
                           const float *a, const float *x, float *y) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                rows, cols, 1.0f, a, cols, x, 1, 0.0f, y, 1);
}

void caccel_saxpy(int32_t count, float alpha, const float *x, float *y) {
    cblas_saxpy(count, alpha, x, 1, y, 1);
}
