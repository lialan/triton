// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" | FileCheck %s --implicit-check-not=llvm.inline_asm

// amdg.cd_regclass becomes a policy on every MFMA, with no pinning asm.
// Legal immediate C folding remains the backend's responsibility.

// CHECK-LABEL: mfma_cd_regclass_agpr
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_cd_regclass_agpr(%arg0: tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                                        %arg1: tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
                                        %arg2: tensor<16x16xf32, #mma>) {
    // Two K steps, both constrained.
    // CHECK: %[[D0:.*]] = rocdl.mfma.f32.16x16x32.f16 {{.*}} cd_regclass = require_agpr
    // CHECK-NEXT: rocdl.mfma.f32.16x16x32.f16 {{.*}}%[[D0]]{{.*}} cd_regclass = require_agpr
    %dot = tt.dot %arg0, %arg1, %arg2 {amdg.cd_regclass = "a"} : tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: mfma_cd_regclass_constant_acc
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_cd_regclass_constant_acc(%arg0: tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                                                %arg1: tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    // CHECK: %[[D0:.*]] = rocdl.mfma.f32.16x16x32.f16 {{.*}} cd_regclass = require_agpr
    // CHECK-NEXT: rocdl.mfma.f32.16x16x32.f16 {{.*}}%[[D0]]{{.*}} cd_regclass = require_agpr
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %dot = tt.dot %arg0, %arg1, %cst {amdg.cd_regclass = "a"} : tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: mfma_no_cd_regclass
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_no_cd_regclass(%arg0: tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                                      %arg1: tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
                                      %arg2: tensor<16x16xf32, #mma>) {
    // CHECK-NOT: llvm.inline_asm
    // CHECK: rocdl.mfma.f32.16x16x32.f16
    // CHECK: rocdl.mfma.f32.16x16x32.f16
    // CHECK-NOT: llvm.inline_asm
    // CHECK: llvm.return
    %dot = tt.dot %arg0, %arg1, %arg2 : tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// A 32x32 accumulator constrains a 16-register tuple.
// CHECK-LABEL: mfma_cd_regclass_vgpr_32x32
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_cd_regclass_vgpr_32x32(%arg0: tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                                              %arg1: tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
                                              %arg2: tensor<32x32xf32, #mma>) {
    // CHECK: rocdl.mfma.f32.32x32x16.f16 {{.*}} cd_regclass = require_vgpr
    %dot = tt.dot %arg0, %arg1, %arg2 {amdg.cd_regclass = "v"} : tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

// Scaled MFMA constrains each tile and K step, including a zero accumulator.
// CHECK-LABEL: mfma_scaled_cd_regclass_agpr
#linear = #ttg.linear<{register = [[0, 4], [32, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 128], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_scaled_cd_regclass_agpr(%arg0: tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<256x8xi8, #linear>, %arg2: tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<256x8xi8, #linear1>) {
    // CHECK: %[[D0:.*]] = rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} cd_regclass = require_agpr
    // CHECK-NEXT: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}%[[D0]]{{.*}} cd_regclass = require_agpr
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %dots = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e2m1 rhs = e2m1 {amdg.cd_regclass = "a", fastMath = false} : tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear1> -> tensor<256x256xf32, #mma>
    tt.return
  }
}

// -----

// Two-step (ping-pong) scaled MFMA preserves the policy in both passes.
// CHECK-LABEL: mfma_scaled_2step_cd_regclass
#linear = #ttg.linear<{register = [[0, 4], [32, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 4], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [32, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 128], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_scaled_2step_cd_regclass(%arg0: tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<256x8xi8, #linear>, %arg2: tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<256x8xi8, #linear1>) {
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} cd_regclass = require_agpr
    // CHECK: rocdl.s.barrier
    // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} cd_regclass = require_agpr
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %dots = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e2m1 rhs = e2m1 {amdg.cd_regclass = "a", fastMath = false, pingpong_2step} : tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear> * tensor<128x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<256x8xi8, #linear1> -> tensor<256x256xf32, #mma>
    tt.return
  }
}
