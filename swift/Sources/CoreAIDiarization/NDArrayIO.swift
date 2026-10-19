// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

/// An input array for `descriptor`, filled from float32 values in row-major order.
func makeArray(_ descriptor: NDArrayDescriptor, _ values: [Float]) -> NDArray {
    var array = NDArray(descriptor: descriptor.resolvingDynamicDimensions(descriptor.shape))
    fillFloatNDArray(&array, with: values)
    return array
}

/// An output array as float32 values in row-major order, whatever its scalar type.
func readFloats(_ array: NDArray) -> [Float] {
    flattenAsFloat(array)
}
