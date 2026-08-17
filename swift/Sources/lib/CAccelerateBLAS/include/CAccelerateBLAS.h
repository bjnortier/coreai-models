// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

#ifndef CAccelerateBLAS_h
#define CAccelerateBLAS_h

#include <stdint.h>

// Thin CBLAS wrappers so Swift callers avoid the deprecated classic CBLAS
// symbols. The implementation is compiled with ACCELERATE_NEW_LAPACK (set via
// the target's cSettings), which routes these calls through Accelerate's
// current LP64 CBLAS interface. Confining the macro to this C target keeps it
// out of the Swift targets' build flags, so the package stays publishable
// without `.unsafeFlags`.

/// Row-major single-precision matrix-vector product `y = A * x`.
///
/// `a` is a `rows * cols` matrix (row-major), `x` has `cols` elements, and the
/// result `y` has `rows` elements.
void caccel_sgemv_rowmajor(int32_t rows, int32_t cols,
                           const float *a, const float *x, float *y);

/// Single-precision `y += alpha * x` over `count` elements (unit stride).
void caccel_saxpy(int32_t count, float alpha, const float *x, float *y);

#endif /* CAccelerateBLAS_h */
