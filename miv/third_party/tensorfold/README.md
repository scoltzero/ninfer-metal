TensorFold cb2ebf0540f42604e2759b2ddef497861e928248

The M5 tensor fragment implementation in kernels/affine_lane.metal is adapted from lane_qmm.py. The upstream NOTICE and Apache-2.0/MIT texts are retained here. Modified code uses native C++ ownership, padded-N prepared buffers, per-element tails, and a physical-device guard. The portable scalar reference and host adapter are implemented in this project.
