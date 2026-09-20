# AOT ID: ['10_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/shchon11/F1tenth/F1tenth_E2E/work/adaptive-racing-v3/controller-proof/inductor-cold-v1/ai/caigwkh4o3fs7fmdvv62yqhfvxur2zznohc5mm2uhdbojdw7ohz5.py
# Topologically Sorted Source Nodes: [speed, square, mul, le, cap, grip, mul_5, mul_6, mul_7, square_2, abs_1, truediv, tanh, mul_1, square_1, mul_2, r, mul_8, square_3, sub, residual, sqrt, square_4, mul_9, clamp_min_1, truediv_1, cap_1, grip_1, mul_10, mul_11, mul_12, square_5, mul_13, square_6, sub_1, residual_1, sqrt_1, square_7, mul_14, clamp_min_3, truediv_2, cap_2, mul_15, atan, steer_cap, neg, getitem_1, sub_2, steer_lo, getitem_2, add_2, steer_hi, le_1, getitem_3, minimum_3, steer, zeros_like, getitem_4, add_3, minimum_4, getitem_5, sub_3, maximum_2, recovery, steer_1, abs_3, clamp_min_4, truediv_5, clamp_2, power, abs_2, truediv_4, tanh_1, mul_16, square_8, mul_17, r_1, hi, mul_27, square_11, square_12, mul_28, tan, curvature, mul_29, square_13, add_5, grip_2, mul_30, mul_31, square_14, cc, ge_3, zeros_like_3, mul_22, square_10, mul_23, mul_24, mul_25, sub_7, bb, square_16, mul_21, square_9, aa, mul_35, mul_36, disc_1, ge_2, clamp_min_7, sqrt_3, den_1, gt_2, and__1, mul_37, clamp_min_8, root_2, full_like_2, root_3, where_4, hi_1, hi_2, mul_44, square_19, square_20, mul_45, mul_46, square_21, add_8, grip_3, mul_47, mul_48, square_22, cc_1, ge_7, zeros_like_5, mul_39, square_18, mul_40, mul_41, mul_42, sub_12, bb_1, square_24, mul_38, square_17, aa_1, mul_52, mul_53, disc_3, ge_6, clamp_min_11, sqrt_5, den_3, gt_5, and__3, mul_54, clamp_min_12, root_6, full_like_4, root_7, where_8, hi_3, sub_16, truediv_10, upper, lo, ge_1, zeros_like_2, neg_1, square_15, mul_32, mul_33, disc, ge, clamp_min_5, sqrt_2, den, gt_1, and_, mul_34, clamp_min_6, root, full_like_1, root_1, where_2, neg_2, lo_1, ge_5, zeros_like_4, neg_3, square_23, mul_49, mul_50, disc_2, ge_4, clamp_min_9, sqrt_4, den_2, gt_4, and__2, mul_51, clamp_min_10, root_4, full_like_3, root_5, where_6, neg_4, lo_2, lo_3, upper_1, bad, gt, bad_1, gt_3, bad_2, gt_6, bad_3, gt_7, or__3], Original ATen: [aten.select, aten.pow, aten.mul, aten.add, aten.full_like, aten.abs, aten.div, aten.tanh, aten.sub, aten.clamp_min, aten.sqrt, aten.minimum, aten.atan, aten.clamp, aten.neg, aten.maximum, aten.le, aten.zeros_like, aten.where, aten.reciprocal, aten.tan, aten.ge, aten.rsub, aten.gt, aten.bitwise_and, aten.bitwise_or]
# Source node to ATen node mapping:
#   aa => sub_6
#   aa_1 => sub_11
#   abs_1 => abs_1
#   abs_2 => abs_2
#   abs_3 => abs_3
#   add_2 => add_2
#   add_3 => add_3
#   add_5 => add_5
#   add_8 => add_8
#   and_ => bitwise_and
#   and__1 => bitwise_and_1
#   and__2 => bitwise_and_2
#   and__3 => bitwise_and_3
#   atan => atan
#   bad => full_default_2
#   bad_1 => bitwise_or
#   bad_2 => bitwise_or_1
#   bad_3 => bitwise_or_2
#   bb => mul_27
#   bb_1 => mul_44
#   cap => full_default
#   cap_1 => minimum
#   cap_2 => minimum_1
#   cc => sub_8
#   cc_1 => sub_13
#   clamp_2 => clamp_max_2
#   clamp_min_1 => clamp_min_1
#   clamp_min_10 => clamp_min_11
#   clamp_min_11 => clamp_min_12
#   clamp_min_12 => clamp_min_13
#   clamp_min_3 => clamp_min_3
#   clamp_min_4 => clamp_min_5
#   clamp_min_5 => clamp_min_6
#   clamp_min_6 => clamp_min_7
#   clamp_min_7 => clamp_min_8
#   clamp_min_8 => clamp_min_9
#   clamp_min_9 => clamp_min_10
#   curvature => div_3
#   den => add_6
#   den_1 => add_7
#   den_2 => add_9
#   den_3 => add_10
#   disc => sub_9
#   disc_1 => sub_10
#   disc_2 => sub_14
#   disc_3 => sub_15
#   full_like_1 => full_default_3
#   full_like_2 => full_default_5
#   full_like_3 => full_default_7
#   full_like_4 => full_default_9
#   ge => ge
#   ge_1 => ge_1
#   ge_2 => ge_2
#   ge_3 => ge_3
#   ge_4 => ge_4
#   ge_5 => ge_5
#   ge_6 => ge_6
#   ge_7 => ge_7
#   getitem_1 => select_1
#   getitem_2 => select_2
#   getitem_3 => select_3
#   getitem_4 => select_4
#   getitem_5 => select_5
#   grip => mul_3
#   grip_1 => mul_4
#   grip_2 => mul_20
#   grip_3 => mul_21
#   gt => gt
#   gt_1 => gt_1
#   gt_2 => gt_2
#   gt_3 => gt_3
#   gt_4 => gt_4
#   gt_5 => gt_5
#   gt_6 => gt_6
#   gt_7 => gt_7
#   hi => sub_5
#   hi_1 => minimum_5
#   hi_2 => clamp_max_3
#   hi_3 => minimum_6
#   le => add
#   le_1 => le
#   lo => sub_4
#   lo_1 => maximum_3
#   lo_2 => maximum_4
#   lo_3 => clamp_min_14
#   maximum_2 => maximum_2
#   minimum_3 => minimum_3
#   minimum_4 => minimum_4
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_17 => mul_17
#   mul_2 => mul_2
#   mul_21 => mul_22
#   mul_22 => mul_23
#   mul_23 => mul_24
#   mul_24 => mul_25
#   mul_25 => mul_26
#   mul_27 => mul_28
#   mul_28 => mul_29
#   mul_29 => mul_30
#   mul_30 => mul_31
#   mul_31 => mul_32
#   mul_32 => mul_33
#   mul_33 => mul_34
#   mul_34 => mul_35
#   mul_35 => mul_36
#   mul_36 => mul_37
#   mul_37 => mul_38
#   mul_38 => mul_39
#   mul_39 => mul_40
#   mul_40 => mul_41
#   mul_41 => mul_42
#   mul_42 => mul_43
#   mul_44 => mul_45
#   mul_45 => mul_46
#   mul_46 => mul_47
#   mul_47 => mul_48
#   mul_48 => mul_49
#   mul_49 => mul_50
#   mul_5 => mul_5
#   mul_50 => mul_51
#   mul_51 => mul_52
#   mul_52 => mul_53
#   mul_53 => mul_54
#   mul_54 => mul_55
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   neg => neg
#   neg_1 => neg_1
#   neg_2 => neg_2
#   neg_3 => neg_3
#   neg_4 => neg_4
#   or__3 => bitwise_or_3
#   power => mul_19
#   r => add_1
#   r_1 => add_4
#   recovery => clamp_max_1, clamp_min_4
#   residual => clamp_min
#   residual_1 => clamp_min_2
#   root => div_5
#   root_1 => where_1
#   root_2 => div_6
#   root_3 => where_3
#   root_4 => div_7
#   root_5 => where_5
#   root_6 => div_8
#   root_7 => where_7
#   speed => select
#   sqrt => sqrt
#   sqrt_1 => sqrt_1
#   sqrt_2 => sqrt_2
#   sqrt_3 => sqrt_3
#   sqrt_4 => sqrt_4
#   sqrt_5 => sqrt_5
#   square => pow_1
#   square_1 => pow_2
#   square_10 => pow_11
#   square_11 => pow_12
#   square_12 => pow_13
#   square_13 => pow_14
#   square_14 => pow_15
#   square_15 => pow_16
#   square_16 => pow_17
#   square_17 => pow_18
#   square_18 => pow_19
#   square_19 => pow_20
#   square_2 => pow_3
#   square_20 => pow_21
#   square_21 => pow_22
#   square_22 => pow_23
#   square_23 => pow_24
#   square_24 => pow_25
#   square_3 => pow_4
#   square_4 => pow_5
#   square_5 => pow_6
#   square_6 => pow_7
#   square_7 => pow_8
#   square_8 => pow_9
#   square_9 => pow_10
#   steer => maximum_1
#   steer_1 => where
#   steer_cap => clamp_max
#   steer_hi => minimum_2
#   steer_lo => maximum
#   sub => sub
#   sub_1 => sub_1
#   sub_12 => sub_12
#   sub_16 => sub_16
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_7 => sub_7
#   tan => tan
#   tanh => tanh
#   tanh_1 => tanh_1
#   truediv => div
#   truediv_1 => div_1
#   truediv_10 => div_9
#   truediv_2 => div_2
#   truediv_4 => div_4
#   truediv_5 => mul_18, reciprocal
#   upper => minimum_7
#   upper_1 => maximum_5
#   where_2 => where_2
#   where_4 => where_4
#   where_6 => where_6
#   where_8 => where_8
#   zeros_like => full_default_1
#   zeros_like_2 => full_default_4
#   zeros_like_3 => full_default_6
#   zeros_like_4 => full_default_8
#   zeros_like_5 => full_default_10
# Graph fragment:
#   %arg0_1 : Tensor "f32[1, 6][6, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %clamp_max : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=clamp_max]
#   %arg2_1 : Tensor "f32[1, 2][24, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %sub_8 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=sub_8]
#   %sub_10 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=sub_10]
#   %sub_13 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=sub_13]
#   %sub_15 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=sub_15]
#   %add_7 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=add_7]
#   %add_10 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=add_10]
#   %sub_9 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=sub_9]
#   %sub_14 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=sub_14]
#   %add_6 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=add_6]
#   %add_9 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=add_9]
#   %clamp_min_14 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=clamp_min_14]
#   %minimum_6 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=minimum_6]
#   %select : Tensor "f32[1][6]cuda:0"[num_users=11] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 1, 3), kwargs = {})
#   %pow_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %mul : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.003), kwargs = {})
#   %add : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 0.3302), kwargs = {})
#   %full_default : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 0.92), kwargs = {})
#   %mul_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, 0.85), kwargs = {})
#   %mul_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, 9.81), kwargs = {})
#   %mul_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, 0.5192307692307692), kwargs = {})
#   %pow_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_7, 2), kwargs = {})
#   %abs_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%select,), kwargs = {})
#   %div : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_1, 0.05), kwargs = {})
#   %tanh : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 0.1), kwargs = {})
#   %pow_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %mul_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_2, 0.01), kwargs = {})
#   %add_1 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %mul_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.5), kwargs = {})
#   %pow_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_8, 2), kwargs = {})
#   %sub : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_3, %pow_4), kwargs = {})
#   %clamp_min : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0), kwargs = {})
#   %sqrt : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min,), kwargs = {})
#   %pow_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %mul_9 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_5, 0.5192307692307692), kwargs = {})
#   %clamp_min_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_9, 1e-06), kwargs = {})
#   %div_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sqrt, %clamp_min_1), kwargs = {})
#   %minimum : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %div_1), kwargs = {})
#   %mul_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 1.0), kwargs = {})
#   %mul_10 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, 0.85), kwargs = {})
#   %mul_11 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, 9.81), kwargs = {})
#   %mul_12 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, 0.4807692307692308), kwargs = {})
#   %pow_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_12, 2), kwargs = {})
#   %mul_13 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.5), kwargs = {})
#   %pow_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_13, 2), kwargs = {})
#   %sub_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_6, %pow_7), kwargs = {})
#   %clamp_min_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_1, 0), kwargs = {})
#   %sqrt_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_2,), kwargs = {})
#   %pow_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %mul_14 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_8, 0.4807692307692308), kwargs = {})
#   %clamp_min_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_14, 1e-06), kwargs = {})
#   %div_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sqrt_1, %clamp_min_3), kwargs = {})
#   %minimum_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%minimum, %div_2), kwargs = {})
#   %mul_15 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %minimum_1), kwargs = {})
#   %atan : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.atan.default](args = (%mul_15,), kwargs = {})
#   %clamp_max : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%atan, 0.4189), kwargs = {})
#   %neg : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%clamp_max,), kwargs = {})
#   %select_1 : Tensor "f32[1][6]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 1, 4), kwargs = {})
#   %sub_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_1, 0.16000000000000003), kwargs = {})
#   %maximum : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.maximum.default](args = (%neg, %sub_2), kwargs = {})
#   %select_2 : Tensor "f32[1][6]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 1, 4), kwargs = {})
#   %add_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_2, 0.16000000000000003), kwargs = {})
#   %minimum_2 : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.minimum.default](args = (%clamp_max, %add_2), kwargs = {})
#   %le : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%maximum, %minimum_2), kwargs = {})
#   %select_3 : Tensor "f32[1][24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg2_1, 1, 0), kwargs = {})
#   %minimum_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%select_3, %minimum_2), kwargs = {})
#   %maximum_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%minimum_3, %maximum), kwargs = {})
#   %full_default_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_4 : Tensor "f32[1][6]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 1, 4), kwargs = {})
#   %add_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_4, 0.16000000000000003), kwargs = {})
#   %minimum_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default_1, %add_3), kwargs = {})
#   %select_5 : Tensor "f32[1][6]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 1, 4), kwargs = {})
#   %sub_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_5, 0.16000000000000003), kwargs = {})
#   %maximum_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%minimum_4, %sub_3), kwargs = {})
#   %clamp_min_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%maximum_2, -0.4189), kwargs = {})
#   %clamp_max_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_4, 0.4189), kwargs = {})
#   %where : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %maximum_1, %clamp_max_1), kwargs = {})
#   %abs_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%select,), kwargs = {})
#   %clamp_min_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%abs_3, 0.001), kwargs = {})
#   %reciprocal : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_5,), kwargs = {})
#   %mul_18 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 7.319), kwargs = {})
#   %clamp_max_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_18, 1), kwargs = {})
#   %mul_19 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_2, 7.0), kwargs = {})
#   %abs_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%select,), kwargs = {})
#   %div_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_2, 0.05), kwargs = {})
#   %tanh_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_4,), kwargs = {})
#   %mul_16 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_1, 0.1), kwargs = {})
#   %pow_9 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %mul_17 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_9, 0.01), kwargs = {})
#   %add_4 : Tensor "f32[1][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %mul_17), kwargs = {})
#   %sub_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_19, %add_4), kwargs = {})
#   %mul_28 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 0.5), kwargs = {})
#   %pow_12 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_28, 2), kwargs = {})
#   %pow_13 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %mul_29 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_13, 0.5192307692307692), kwargs = {})
#   %tan : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tan.default](args = (%where,), kwargs = {})
#   %div_3 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%tan, %add), kwargs = {})
#   %mul_30 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %div_3), kwargs = {})
#   %pow_14 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_30, 2), kwargs = {})
#   %add_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_12, %pow_14), kwargs = {})
#   %mul_20 : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 0.92), kwargs = {})
#   %mul_31 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, 9.81), kwargs = {})
#   %mul_32 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, 0.5192307692307692), kwargs = {})
#   %pow_15 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_32, 2), kwargs = {})
#   %sub_8 : Tensor "f32[1][1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %pow_15), kwargs = {})
#   %ge_3 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_8, 0), kwargs = {})
#   %full_default_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_23 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 0.25), kwargs = {})
#   %pow_11 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_20, 2), kwargs = {})
#   %mul_24 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_11, 9.81), kwargs = {})
#   %mul_25 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, 0.5192307692307692), kwargs = {})
#   %mul_26 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, -0.22410660205935795), kwargs = {})
#   %sub_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_23, %mul_26), kwargs = {})
#   %mul_27 : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, 2), kwargs = {})
#   %pow_17 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_27, 2), kwargs = {})
#   %mul_22 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, -0.22410660205935795), kwargs = {})
#   %pow_10 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_22, 2), kwargs = {})
#   %sub_6 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_10), kwargs = {})
#   %mul_36 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, 4), kwargs = {})
#   %mul_37 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, %sub_8), kwargs = {})
#   %sub_10 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_17, %mul_37), kwargs = {})
#   %ge_2 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_10, 0), kwargs = {})
#   %clamp_min_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_10, 0), kwargs = {})
#   %sqrt_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_8,), kwargs = {})
#   %add_7 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %sqrt_3), kwargs = {})
#   %gt_2 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_7, 1e-12), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_2, %gt_2), kwargs = {})
#   %mul_38 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, -2), kwargs = {})
#   %clamp_min_9 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_7, 1e-12), kwargs = {})
#   %div_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_38, %clamp_min_9), kwargs = {})
#   %full_default_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_1, %div_6, %full_default_5), kwargs = {})
#   %where_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_3, %full_default_6, %where_3), kwargs = {})
#   %minimum_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%sub_5, %where_4), kwargs = {})
#   %clamp_max_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%minimum_5, 22.72870945945946), kwargs = {})
#   %mul_45 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 0.5), kwargs = {})
#   %pow_20 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_45, 2), kwargs = {})
#   %pow_21 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %mul_46 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_21, 0.4807692307692308), kwargs = {})
#   %mul_47 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %div_3), kwargs = {})
#   %pow_22 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_47, 2), kwargs = {})
#   %add_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_20, %pow_22), kwargs = {})
#   %mul_21 : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 1.0), kwargs = {})
#   %mul_48 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, 9.81), kwargs = {})
#   %mul_49 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, 0.4807692307692308), kwargs = {})
#   %pow_23 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_49, 2), kwargs = {})
#   %sub_13 : Tensor "f32[1][1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %pow_23), kwargs = {})
#   %ge_7 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_13, 0), kwargs = {})
#   %full_default_10 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_40 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 0.25), kwargs = {})
#   %pow_19 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_21, 2), kwargs = {})
#   %mul_41 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_19, 9.81), kwargs = {})
#   %mul_42 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, 0.4807692307692308), kwargs = {})
#   %mul_43 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_42, 0.22410660205935795), kwargs = {})
#   %sub_12 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_40, %mul_43), kwargs = {})
#   %mul_44 : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, 2), kwargs = {})
#   %pow_25 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_44, 2), kwargs = {})
#   %mul_39 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, 0.22410660205935795), kwargs = {})
#   %pow_18 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_39, 2), kwargs = {})
#   %sub_11 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_18), kwargs = {})
#   %mul_53 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, 4), kwargs = {})
#   %mul_54 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_53, %sub_13), kwargs = {})
#   %sub_15 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_25, %mul_54), kwargs = {})
#   %ge_6 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_15, 0), kwargs = {})
#   %clamp_min_12 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_15, 0), kwargs = {})
#   %sqrt_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_12,), kwargs = {})
#   %add_10 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %sqrt_5), kwargs = {})
#   %gt_5 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_10, 1e-12), kwargs = {})
#   %bitwise_and_3 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_6, %gt_5), kwargs = {})
#   %mul_55 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, -2), kwargs = {})
#   %clamp_min_13 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_10, 1e-12), kwargs = {})
#   %div_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_55, %clamp_min_13), kwargs = {})
#   %full_default_9 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_3, %div_8, %full_default_9), kwargs = {})
#   %where_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_7, %full_default_10, %where_7), kwargs = {})
#   %minimum_6 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.minimum.default](args = (%clamp_max_3, %where_8), kwargs = {})
#   %sub_16 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (12.0, %select), kwargs = {})
#   %div_9 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_16, 0.05), kwargs = {})
#   %minimum_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%minimum_6, %div_9), kwargs = {})
#   %sub_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (-5.0, %add_4), kwargs = {})
#   %ge_1 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_8, 0), kwargs = {})
#   %full_default_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg_1 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_27,), kwargs = {})
#   %pow_16 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg_1, 2), kwargs = {})
#   %mul_33 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, 4), kwargs = {})
#   %mul_34 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, %sub_8), kwargs = {})
#   %sub_9 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_16, %mul_34), kwargs = {})
#   %ge : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_9, 0), kwargs = {})
#   %clamp_min_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_9, 0), kwargs = {})
#   %sqrt_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_6,), kwargs = {})
#   %add_6 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_1, %sqrt_2), kwargs = {})
#   %gt_1 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_6, 1e-12), kwargs = {})
#   %bitwise_and : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge, %gt_1), kwargs = {})
#   %mul_35 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, -2), kwargs = {})
#   %clamp_min_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_6, 1e-12), kwargs = {})
#   %div_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_35, %clamp_min_7), kwargs = {})
#   %full_default_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and, %div_5, %full_default_3), kwargs = {})
#   %where_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_1, %full_default_4, %where_1), kwargs = {})
#   %neg_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_2,), kwargs = {})
#   %maximum_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%sub_4, %neg_2), kwargs = {})
#   %ge_5 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_13, 0), kwargs = {})
#   %full_default_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg_3 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_44,), kwargs = {})
#   %pow_24 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg_3, 2), kwargs = {})
#   %mul_50 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, 4), kwargs = {})
#   %mul_51 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, %sub_13), kwargs = {})
#   %sub_14 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_24, %mul_51), kwargs = {})
#   %ge_4 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_14, 0), kwargs = {})
#   %clamp_min_10 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_14, 0), kwargs = {})
#   %sqrt_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_10,), kwargs = {})
#   %add_9 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_3, %sqrt_4), kwargs = {})
#   %gt_4 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_9, 1e-12), kwargs = {})
#   %bitwise_and_2 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_4, %gt_4), kwargs = {})
#   %mul_52 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, -2), kwargs = {})
#   %clamp_min_11 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_9, 1e-12), kwargs = {})
#   %div_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_52, %clamp_min_11), kwargs = {})
#   %full_default_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_2, %div_7, %full_default_7), kwargs = {})
#   %where_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_5, %full_default_8, %where_5), kwargs = {})
#   %neg_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_6,), kwargs = {})
#   %maximum_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%maximum_3, %neg_4), kwargs = {})
#   %clamp_min_14 : Tensor "f32[1][1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.clamp_min.default](args = (%maximum_4, -21.045101351351356), kwargs = {})
#   %maximum_5 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.maximum.default](args = (%minimum_7, %clamp_min_14), kwargs = {})
#   %full_default_2 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %gt : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%sub_8, 1e-05), kwargs = {})
#   %bitwise_or : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%full_default_2, %gt), kwargs = {})
#   %gt_3 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%sub_13, 1e-05), kwargs = {})
#   %bitwise_or_1 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_or, %gt_3), kwargs = {})
#   %gt_6 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Tensor](args = (%clamp_min_14, %minimum_6), kwargs = {})
#   %bitwise_or_2 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_or_1, %gt_6), kwargs = {})
#   %gt_7 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Tensor](args = (%maximum, %minimum_2), kwargs = {})
#   %bitwise_or_3 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_or_2, %gt_7), kwargs = {})
#   return %clamp_max,%sub_8,%sub_13,%sub_10,%add_7,%sub_15,%add_10,%minimum_6,%sub_9,%add_6,%sub_14,%add_9,%clamp_min_14,%bitwise_or_3,%maximum_5
triton_poi_fused_abs_add_atan_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_le_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_sqrt_sub_tan_tanh_where_zeros_like_0 = async_compile.triton('triton_poi_fused_abs_add_atan_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_le_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_sqrt_sub_tan_tanh_where_zeros_like_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr9': '*i1', 'out_ptr10': '*fp32', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_atan_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_le_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_sqrt_sub_tan_tanh_where_zeros_like_0', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 4, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_atan_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_le_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_sqrt_sub_tan_tanh_where_zeros_like_0(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr9, out_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.load(in_ptr0 + (3))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp59 = tl.load(in_ptr0 + (4))
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK])
    tmp67 = tl.load(in_ptr2 + (0))
    tmp68 = tl.broadcast_to(tmp67, [XBLOCK])
    tmp2 = tmp1 * tmp1
    tmp3 = 0.003
    tmp4 = tmp2 * tmp3
    tmp5 = 0.3302
    tmp6 = tmp4 + tmp5
    tmp9 = 0.92
    tmp10 = tmp8 * tmp9
    tmp11 = 0.85
    tmp12 = tmp10 * tmp11
    tmp13 = 9.81
    tmp14 = tmp12 * tmp13
    tmp15 = 0.5192307692307692
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl_math.abs(tmp1)
    tmp19 = 20.0
    tmp20 = tmp18 * tmp19
    tmp21 = libdevice.tanh(tmp20)
    tmp22 = 0.1
    tmp23 = tmp21 * tmp22
    tmp24 = 0.01
    tmp25 = tmp2 * tmp24
    tmp26 = tmp23 + tmp25
    tmp27 = 0.5
    tmp28 = tmp26 * tmp27
    tmp29 = tmp28 * tmp28
    tmp30 = tmp17 - tmp29
    tmp31 = 0.0
    tmp32 = triton_helpers.maximum(tmp30, tmp31)
    tmp33 = tl.sqrt_rn(tmp32)
    tmp34 = tmp2 * tmp15
    tmp35 = 1e-06
    tmp36 = triton_helpers.maximum(tmp34, tmp35)
    tmp37 = (tmp33 / tmp36)
    tmp38 = float("inf")
    tmp39 = triton_helpers.minimum(tmp38, tmp37)
    tmp40 = 1.0
    tmp41 = tmp8 * tmp40
    tmp42 = tmp41 * tmp11
    tmp43 = tmp42 * tmp13
    tmp44 = 0.4807692307692308
    tmp45 = tmp43 * tmp44
    tmp46 = tmp45 * tmp45
    tmp47 = tmp46 - tmp29
    tmp48 = triton_helpers.maximum(tmp47, tmp31)
    tmp49 = tl.sqrt_rn(tmp48)
    tmp50 = tmp2 * tmp44
    tmp51 = triton_helpers.maximum(tmp50, tmp35)
    tmp52 = (tmp49 / tmp51)
    tmp53 = triton_helpers.minimum(tmp39, tmp52)
    tmp54 = tmp6 * tmp53
    tmp55 = libdevice.atan(tmp54)
    tmp56 = 0.4189
    tmp57 = triton_helpers.minimum(tmp55, tmp56)
    tmp58 = -tmp57
    tmp61 = 0.16000000000000003
    tmp62 = tmp60 - tmp61
    tmp63 = triton_helpers.maximum(tmp58, tmp62)
    tmp64 = tmp60 + tmp61
    tmp65 = triton_helpers.minimum(tmp57, tmp64)
    tmp66 = tmp63 <= tmp65
    tmp69 = triton_helpers.minimum(tmp68, tmp65)
    tmp70 = triton_helpers.maximum(tmp69, tmp63)
    tmp71 = triton_helpers.minimum(tmp31, tmp64)
    tmp72 = triton_helpers.maximum(tmp71, tmp62)
    tmp73 = -0.4189
    tmp74 = triton_helpers.maximum(tmp72, tmp73)
    tmp75 = triton_helpers.minimum(tmp74, tmp56)
    tmp76 = tl.where(tmp66, tmp70, tmp75)
    tmp77 = libdevice.tan(tmp76)
    tmp78 = (tmp77 / tmp6)
    tmp79 = tmp34 * tmp78
    tmp80 = tmp79 * tmp79
    tmp81 = tmp29 + tmp80
    tmp82 = tmp10 * tmp13
    tmp83 = tmp82 * tmp15
    tmp84 = tmp83 * tmp83
    tmp85 = tmp81 - tmp84
    tmp86 = tmp50 * tmp78
    tmp87 = tmp86 * tmp86
    tmp88 = tmp29 + tmp87
    tmp89 = tmp41 * tmp13
    tmp90 = tmp89 * tmp44
    tmp91 = tmp90 * tmp90
    tmp92 = tmp88 - tmp91
    tmp93 = 0.25
    tmp94 = tmp26 * tmp93
    tmp95 = tmp10 * tmp10
    tmp96 = tmp95 * tmp13
    tmp97 = tmp96 * tmp15
    tmp98 = -0.22410660205935795
    tmp99 = tmp97 * tmp98
    tmp100 = tmp94 - tmp99
    tmp101 = 2.0
    tmp102 = tmp100 * tmp101
    tmp103 = tmp102 * tmp102
    tmp104 = tmp10 * tmp98
    tmp105 = tmp104 * tmp104
    tmp106 = tmp93 - tmp105
    tmp107 = 4.0
    tmp108 = tmp106 * tmp107
    tmp109 = tmp108 * tmp85
    tmp110 = tmp103 - tmp109
    tmp111 = triton_helpers.maximum(tmp110, tmp31)
    tmp112 = tl.sqrt_rn(tmp111)
    tmp113 = tmp102 + tmp112
    tmp114 = tmp41 * tmp41
    tmp115 = tmp114 * tmp13
    tmp116 = tmp115 * tmp44
    tmp117 = 0.22410660205935795
    tmp118 = tmp116 * tmp117
    tmp119 = tmp94 - tmp118
    tmp120 = tmp119 * tmp101
    tmp121 = tmp120 * tmp120
    tmp122 = tmp41 * tmp117
    tmp123 = tmp122 * tmp122
    tmp124 = tmp93 - tmp123
    tmp125 = tmp124 * tmp107
    tmp126 = tmp125 * tmp92
    tmp127 = tmp121 - tmp126
    tmp128 = triton_helpers.maximum(tmp127, tmp31)
    tmp129 = tl.sqrt_rn(tmp128)
    tmp130 = tmp120 + tmp129
    tmp131 = 0.001
    tmp132 = triton_helpers.maximum(tmp18, tmp131)
    tmp133 = tl.full([1], 1, tl.int32)
    tmp134 = (tmp133 / tmp132)
    tmp135 = 7.319
    tmp136 = tmp134 * tmp135
    tmp137 = triton_helpers.minimum(tmp136, tmp40)
    tmp138 = 7.0
    tmp139 = tmp137 * tmp138
    tmp140 = tmp139 - tmp26
    tmp141 = tmp85 >= tmp31
    tmp142 = tmp110 >= tmp31
    tmp143 = 1e-12
    tmp144 = tmp113 > tmp143
    tmp145 = tmp142 & tmp144
    tmp146 = -2.0
    tmp147 = tmp85 * tmp146
    tmp148 = triton_helpers.maximum(tmp113, tmp143)
    tmp149 = (tmp147 / tmp148)
    tmp150 = tl.where(tmp145, tmp149, tmp38)
    tmp151 = tl.where(tmp141, tmp31, tmp150)
    tmp152 = triton_helpers.minimum(tmp140, tmp151)
    tmp153 = 22.72870945945946
    tmp154 = triton_helpers.minimum(tmp152, tmp153)
    tmp155 = tmp92 >= tmp31
    tmp156 = tmp127 >= tmp31
    tmp157 = tmp130 > tmp143
    tmp158 = tmp156 & tmp157
    tmp159 = tmp92 * tmp146
    tmp160 = triton_helpers.maximum(tmp130, tmp143)
    tmp161 = (tmp159 / tmp160)
    tmp162 = tl.where(tmp158, tmp161, tmp38)
    tmp163 = tl.where(tmp155, tmp31, tmp162)
    tmp164 = triton_helpers.minimum(tmp154, tmp163)
    tmp165 = -tmp102
    tmp166 = tmp165 * tmp165
    tmp167 = tmp166 - tmp109
    tmp168 = triton_helpers.maximum(tmp167, tmp31)
    tmp169 = tl.sqrt_rn(tmp168)
    tmp170 = tmp165 + tmp169
    tmp171 = -tmp120
    tmp172 = tmp171 * tmp171
    tmp173 = tmp172 - tmp126
    tmp174 = triton_helpers.maximum(tmp173, tmp31)
    tmp175 = tl.sqrt_rn(tmp174)
    tmp176 = tmp171 + tmp175
    tmp177 = -5.0
    tmp178 = tmp177 - tmp26
    tmp179 = tmp167 >= tmp31
    tmp180 = tmp170 > tmp143
    tmp181 = tmp179 & tmp180
    tmp182 = triton_helpers.maximum(tmp170, tmp143)
    tmp183 = (tmp147 / tmp182)
    tmp184 = tl.where(tmp181, tmp183, tmp38)
    tmp185 = tl.where(tmp141, tmp31, tmp184)
    tmp186 = -tmp185
    tmp187 = triton_helpers.maximum(tmp178, tmp186)
    tmp188 = tmp173 >= tmp31
    tmp189 = tmp176 > tmp143
    tmp190 = tmp188 & tmp189
    tmp191 = triton_helpers.maximum(tmp176, tmp143)
    tmp192 = (tmp159 / tmp191)
    tmp193 = tl.where(tmp190, tmp192, tmp38)
    tmp194 = tl.where(tmp155, tmp31, tmp193)
    tmp195 = -tmp194
    tmp196 = triton_helpers.maximum(tmp187, tmp195)
    tmp197 = -21.045101351351356
    tmp198 = triton_helpers.maximum(tmp196, tmp197)
    tmp199 = 1e-05
    tmp200 = tmp85 > tmp199
    tmp201 = tl.full([1], False, tl.int1)
    tmp202 = tmp201 | tmp200
    tmp203 = tmp92 > tmp199
    tmp204 = tmp202 | tmp203
    tmp205 = tmp198 > tmp164
    tmp206 = tmp204 | tmp205
    tmp207 = tmp63 > tmp65
    tmp208 = tmp206 | tmp207
    tmp209 = 12.0
    tmp210 = tmp209 - tmp1
    tmp211 = tmp210 * tmp19
    tmp212 = triton_helpers.minimum(tmp164, tmp211)
    tmp213 = triton_helpers.maximum(tmp212, tmp198)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp57, None)
    tl.store(in_out_ptr1 + (tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp198, None)
    tl.store(out_ptr9 + (tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp208, None)
    tl.store(out_ptr10 + (tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp213, None)
''', device_str='cuda')


# kernel path: /home/shchon11/F1tenth/F1tenth_E2E/work/adaptive-racing-v3/controller-proof/inductor-cold-v1/qw/cqwxrdenau6jcmxlxsxmt4dizelkl3exc573zq7aa2iv7k6bi7ev.py
# Topologically Sorted Source Nodes: [neg, getitem_1, sub_2, steer_lo, getitem_2, add_2, steer_hi, le_1, getitem_3, minimum_3, steer, zeros_like, getitem_4, add_3, minimum_4, getitem_5, sub_3, maximum_2, recovery, steer_1, bounded, getitem_6, minimum_8, accel], Original ATen: [aten.neg, aten.select, aten.sub, aten.maximum, aten.add, aten.minimum, aten.le, aten.zeros_like, aten.clamp, aten.where, aten.stack]
# Source node to ATen node mapping:
#   accel => maximum_6
#   add_2 => add_2
#   add_3 => add_3
#   bounded => cat, unsqueeze, unsqueeze_1
#   getitem_1 => select_1
#   getitem_2 => select_2
#   getitem_3 => select_3
#   getitem_4 => select_4
#   getitem_5 => select_5
#   getitem_6 => select_6
#   le_1 => le
#   maximum_2 => maximum_2
#   minimum_3 => minimum_3
#   minimum_4 => minimum_4
#   minimum_8 => minimum_8
#   neg => neg
#   recovery => clamp_max_1, clamp_min_4
#   steer => maximum_1
#   steer_1 => where
#   steer_hi => minimum_2
#   steer_lo => maximum
#   sub_2 => sub_2
#   sub_3 => sub_3
#   zeros_like => full_default_1
# Graph fragment:
#   %clamp_max : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=clamp_max]
#   %arg0_1 : Tensor "f32[1, 6][6, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "f32[1, 2][24, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %maximum_5 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=maximum_5]
#   %clamp_min_14 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=clamp_min_14]
#   %neg : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%clamp_max,), kwargs = {})
#   %select_1 : Tensor "f32[1][6]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 1, 4), kwargs = {})
#   %sub_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_1, 0.16000000000000003), kwargs = {})
#   %maximum : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.maximum.default](args = (%neg, %sub_2), kwargs = {})
#   %select_2 : Tensor "f32[1][6]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 1, 4), kwargs = {})
#   %add_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_2, 0.16000000000000003), kwargs = {})
#   %minimum_2 : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.minimum.default](args = (%clamp_max, %add_2), kwargs = {})
#   %le : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%maximum, %minimum_2), kwargs = {})
#   %select_3 : Tensor "f32[1][24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg2_1, 1, 0), kwargs = {})
#   %minimum_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%select_3, %minimum_2), kwargs = {})
#   %maximum_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%minimum_3, %maximum), kwargs = {})
#   %full_default_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_4 : Tensor "f32[1][6]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 1, 4), kwargs = {})
#   %add_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_4, 0.16000000000000003), kwargs = {})
#   %minimum_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default_1, %add_3), kwargs = {})
#   %select_5 : Tensor "f32[1][6]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 1, 4), kwargs = {})
#   %sub_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_5, 0.16000000000000003), kwargs = {})
#   %maximum_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%minimum_4, %sub_3), kwargs = {})
#   %clamp_min_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%maximum_2, -0.4189), kwargs = {})
#   %clamp_max_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_4, 0.4189), kwargs = {})
#   %where : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %maximum_1, %clamp_max_1), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where, 1), kwargs = {})
#   %select_6 : Tensor "f32[1][24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg2_1, 1, 1), kwargs = {})
#   %minimum_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%select_6, %maximum_5), kwargs = {})
#   %maximum_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%minimum_8, %clamp_min_14), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%maximum_6, 1), kwargs = {})
#   %cat : Tensor "f32[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze, %unsqueeze_1], 1), kwargs = {})
#   return %cat
triton_poi_fused_add_clamp_le_maximum_minimum_neg_select_stack_sub_where_zeros_like_1 = async_compile.triton('triton_poi_fused_add_clamp_le_maximum_minimum_neg_select_stack_sub_where_zeros_like_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_le_maximum_minimum_neg_select_stack_sub_where_zeros_like_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_le_maximum_minimum_neg_select_stack_sub_where_zeros_like_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.where(tmp4, tmp6, 0.0)
    tmp8 = -tmp7
    tmp9 = tl.load(in_ptr1 + (4))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp11 = tl.where(tmp4, tmp10, 0.0)
    tmp12 = 0.16000000000000003
    tmp13 = tmp11 - tmp12
    tmp14 = triton_helpers.maximum(tmp8, tmp13)
    tmp15 = tmp11 + tmp12
    tmp16 = triton_helpers.minimum(tmp7, tmp15)
    tmp17 = tmp14 <= tmp16
    tmp18 = tl.load(in_ptr2 + (0))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp20 = tl.where(tmp4, tmp19, 0.0)
    tmp21 = triton_helpers.minimum(tmp20, tmp16)
    tmp22 = triton_helpers.maximum(tmp21, tmp14)
    tmp23 = 0.0
    tmp24 = triton_helpers.minimum(tmp23, tmp15)
    tmp25 = triton_helpers.maximum(tmp24, tmp13)
    tmp26 = -0.4189
    tmp27 = triton_helpers.maximum(tmp25, tmp26)
    tmp28 = 0.4189
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = tl.where(tmp17, tmp22, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp4, tmp30, tmp31)
    tmp33 = tmp0 >= tmp3
    tmp34 = tl.full([1], 2, tl.int64)
    tmp35 = tmp0 < tmp34
    tmp36 = tl.load(in_ptr2 + (1))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp38 = tl.where(tmp33, tmp37, 0.0)
    tmp39 = tl.load(in_ptr3 + (0))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp41 = tl.where(tmp33, tmp40, 0.0)
    tmp42 = triton_helpers.minimum(tmp38, tmp41)
    tmp43 = tl.load(in_ptr4 + (0))
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK])
    tmp45 = tl.where(tmp33, tmp44, 0.0)
    tmp46 = triton_helpers.maximum(tmp42, tmp45)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp33, tmp46, tmp47)
    tmp49 = tl.where(tmp4, tmp32, tmp48)
    tl.store(out_ptr0 + (x0), tmp49, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 6), (6, 1))
        assert_size_stride(arg1_1, (1, ), (1, ))
        assert_size_stride(arg2_1, (1, 2), (24, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, ), (1, ), torch.float32)
            buf8 = empty_strided_cuda((1, ), (1, ), torch.float32)
            buf12 = buf8; del buf8  # reuse
            buf15 = empty_strided_cuda((1, ), (1, ), torch.bool)
            buf13 = empty_strided_cuda((1, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [speed, square, mul, le, cap, grip, mul_5, mul_6, mul_7, square_2, abs_1, truediv, tanh, mul_1, square_1, mul_2, r, mul_8, square_3, sub, residual, sqrt, square_4, mul_9, clamp_min_1, truediv_1, cap_1, grip_1, mul_10, mul_11, mul_12, square_5, mul_13, square_6, sub_1, residual_1, sqrt_1, square_7, mul_14, clamp_min_3, truediv_2, cap_2, mul_15, atan, steer_cap, neg, getitem_1, sub_2, steer_lo, getitem_2, add_2, steer_hi, le_1, getitem_3, minimum_3, steer, zeros_like, getitem_4, add_3, minimum_4, getitem_5, sub_3, maximum_2, recovery, steer_1, abs_3, clamp_min_4, truediv_5, clamp_2, power, abs_2, truediv_4, tanh_1, mul_16, square_8, mul_17, r_1, hi, mul_27, square_11, square_12, mul_28, tan, curvature, mul_29, square_13, add_5, grip_2, mul_30, mul_31, square_14, cc, ge_3, zeros_like_3, mul_22, square_10, mul_23, mul_24, mul_25, sub_7, bb, square_16, mul_21, square_9, aa, mul_35, mul_36, disc_1, ge_2, clamp_min_7, sqrt_3, den_1, gt_2, and__1, mul_37, clamp_min_8, root_2, full_like_2, root_3, where_4, hi_1, hi_2, mul_44, square_19, square_20, mul_45, mul_46, square_21, add_8, grip_3, mul_47, mul_48, square_22, cc_1, ge_7, zeros_like_5, mul_39, square_18, mul_40, mul_41, mul_42, sub_12, bb_1, square_24, mul_38, square_17, aa_1, mul_52, mul_53, disc_3, ge_6, clamp_min_11, sqrt_5, den_3, gt_5, and__3, mul_54, clamp_min_12, root_6, full_like_4, root_7, where_8, hi_3, sub_16, truediv_10, upper, lo, ge_1, zeros_like_2, neg_1, square_15, mul_32, mul_33, disc, ge, clamp_min_5, sqrt_2, den, gt_1, and_, mul_34, clamp_min_6, root, full_like_1, root_1, where_2, neg_2, lo_1, ge_5, zeros_like_4, neg_3, square_23, mul_49, mul_50, disc_2, ge_4, clamp_min_9, sqrt_4, den_2, gt_4, and__2, mul_51, clamp_min_10, root_4, full_like_3, root_5, where_6, neg_4, lo_2, lo_3, upper_1, bad, gt, bad_1, gt_3, bad_2, gt_6, bad_3, gt_7, or__3], Original ATen: [aten.select, aten.pow, aten.mul, aten.add, aten.full_like, aten.abs, aten.div, aten.tanh, aten.sub, aten.clamp_min, aten.sqrt, aten.minimum, aten.atan, aten.clamp, aten.neg, aten.maximum, aten.le, aten.zeros_like, aten.where, aten.reciprocal, aten.tan, aten.ge, aten.rsub, aten.gt, aten.bitwise_and, aten.bitwise_or]
            stream0 = get_raw_stream(0)
            triton_poi_fused_abs_add_atan_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_le_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_sqrt_sub_tan_tanh_where_zeros_like_0.run(buf12, arg0_1, arg1_1, arg2_1, buf0, buf15, buf13, 1, stream=stream0)
            del arg1_1
            buf14 = empty_strided_cuda((1, 2), (2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [neg, getitem_1, sub_2, steer_lo, getitem_2, add_2, steer_hi, le_1, getitem_3, minimum_3, steer, zeros_like, getitem_4, add_3, minimum_4, getitem_5, sub_3, maximum_2, recovery, steer_1, bounded, getitem_6, minimum_8, accel], Original ATen: [aten.neg, aten.select, aten.sub, aten.maximum, aten.add, aten.minimum, aten.le, aten.zeros_like, aten.clamp, aten.where, aten.stack]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_clamp_le_maximum_minimum_neg_select_stack_sub_where_zeros_like_1.run(buf0, arg0_1, arg2_1, buf13, buf12, buf14, 2, stream=stream0)
            del arg0_1
            del arg2_1
            del buf0
        return (buf14, buf12, buf13, buf15, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 6), (6, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 2), (24, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
