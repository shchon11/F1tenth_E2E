# AOT ID: ['3_inference']
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


# kernel path: /home/shchon11/F1tenth/F1tenth_E2E/work/adaptive-racing-v3/controller-proof/inductor-cold-v1/3c/c3ct7jzjipk6unrjieez2m4rks23f4po4bn4diwn3chvkuwcqko3.py
# Topologically Sorted Source Nodes: [bad, speed, abs_1, truediv_1, tanh, mul_1, square_1, mul_2, r, mul_12, square_4, square_5, mul_13, steer, tan, square, mul, add, kap, mul_14, square_6, add_2, getitem_3, grip, mul_15, mul_16, square_7, cc, gt, bad_1, mul_29, square_12, square_13, mul_30, mul_31, square_14, add_5, grip_1, mul_32, mul_33, square_15, cc_1, gt_3, bad_2, lo, ge_1, zeros_like_1, mul_7, square_3, mul_8, mul_9, mul_10, sub_3, bb, neg, square_8, mul_6, square_2, aa, mul_17, mul_18, disc, ge, clamp_min_1, sqrt, den, gt_1, and_, mul_19, clamp_min_2, root, full_like, root_1, where_1, neg_1, lo_1, ge_5, zeros_like_3, mul_24, square_11, mul_25, mul_26, mul_27, sub_8, bb_1, neg_2, square_16, mul_23, square_10, aa_1, mul_34, mul_35, disc_2, ge_4, clamp_min_5, sqrt_2, den_2, gt_4, and__2, mul_36, clamp_min_6, root_4, full_like_2, root_5, where_5, neg_3, lo_2, lo_3, abs_2, clamp_min, truediv_2, clamp, power, hi, ge_3, zeros_like_2, square_9, mul_20, mul_21, disc_1, ge_2, clamp_min_3, sqrt_1, den_1, gt_2, and__1, mul_22, clamp_min_4, root_2, full_like_1, root_3, where_3, hi_1, hi_2, ge_7, zeros_like_4, square_17, mul_37, mul_38, disc_3, ge_6, clamp_min_7, sqrt_3, den_3, gt_5, and__3, mul_39, clamp_min_8, root_6, full_like_3, root_7, where_7, hi_3, gt_6, bad_3, ax, abs_3, truediv_7, tanh_1, mul_40, square_18, mul_41, r_1, add_9, mul_44, square_19, square_20, mul_45, mul_46, square_21, add_10, demand, getitem_4, grip_2, mul_47, add_11, clamp_min_9, available, sub_12, add_12, mul_49, square_22, square_23, mul_50, mul_51, square_24, add_13, demand_1, grip_3, mul_52, add_14, clamp_min_11, available_1, sub_13, sub_14, sub_15, maximum_2, clamp_min_13, getitem_5, sub_16, abs_4, sub_17, clamp_min_14], Original ATen: [aten.zeros_like, aten.slice, aten.select, aten.abs, aten.div, aten.tanh, aten.mul, aten.pow, aten.add, aten.tan, aten.unsqueeze, aten.sub, aten.gt, aten.bitwise_or, aten.rsub, aten.ge, aten.neg, aten.clamp_min, aten.sqrt, aten.bitwise_and, aten.full_like, aten.where, aten.maximum, aten.clamp, aten.reciprocal, aten.minimum]
# Source node to ATen node mapping:
#   aa => sub_2
#   aa_1 => sub_7
#   abs_1 => abs_1
#   abs_2 => abs_2
#   abs_3 => abs_3
#   abs_4 => abs_4
#   add => add
#   add_10 => add_10
#   add_11 => add_11
#   add_12 => add_12
#   add_13 => add_13
#   add_14 => add_14
#   add_2 => add_2
#   add_5 => add_5
#   add_9 => add_9
#   and_ => bitwise_and
#   and__1 => bitwise_and_1
#   and__2 => bitwise_and_2
#   and__3 => bitwise_and_3
#   available => mul_49
#   available_1 => mul_54
#   ax => select_2
#   bad => full_default
#   bad_1 => bitwise_or
#   bad_2 => bitwise_or_1
#   bad_3 => bitwise_or_2
#   bb => mul_12
#   bb_1 => mul_29
#   cc => sub_4
#   cc_1 => sub_9
#   clamp => clamp_max
#   clamp_min => clamp_min
#   clamp_min_1 => clamp_min_1
#   clamp_min_11 => clamp_min_12
#   clamp_min_13 => clamp_min_14
#   clamp_min_14 => clamp_min_15
#   clamp_min_2 => clamp_min_2
#   clamp_min_3 => clamp_min_3
#   clamp_min_4 => clamp_min_4
#   clamp_min_5 => clamp_min_5
#   clamp_min_6 => clamp_min_6
#   clamp_min_7 => clamp_min_7
#   clamp_min_8 => clamp_min_8
#   clamp_min_9 => clamp_min_10
#   demand => sqrt_4
#   demand_1 => sqrt_5
#   den => add_3
#   den_1 => add_4
#   den_2 => add_6
#   den_3 => add_7
#   disc => sub_5
#   disc_1 => sub_6
#   disc_2 => sub_10
#   disc_3 => sub_11
#   full_like => full_default_1
#   full_like_1 => full_default_3
#   full_like_2 => full_default_5
#   full_like_3 => full_default_7
#   ge => ge
#   ge_1 => ge_1
#   ge_2 => ge_2
#   ge_3 => ge_3
#   ge_4 => ge_4
#   ge_5 => ge_5
#   ge_6 => ge_6
#   ge_7 => ge_7
#   getitem_3 => unsqueeze
#   getitem_4 => unsqueeze_1
#   getitem_5 => select_3, slice_2
#   grip => mul_5
#   grip_1 => mul_6
#   grip_2 => mul_43
#   grip_3 => mul_44
#   gt => gt
#   gt_1 => gt_1
#   gt_2 => gt_2
#   gt_3 => gt_3
#   gt_4 => gt_4
#   gt_5 => gt_5
#   gt_6 => gt_6
#   hi => sub_1
#   hi_1 => minimum
#   hi_2 => clamp_max_1
#   hi_3 => minimum_1
#   kap => div
#   lo => sub
#   lo_1 => maximum
#   lo_2 => maximum_1
#   lo_3 => clamp_min_9
#   maximum_2 => maximum_2
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_11
#   mul_12 => mul_13
#   mul_13 => mul_14
#   mul_14 => mul_15
#   mul_15 => mul_16
#   mul_16 => mul_17
#   mul_17 => mul_18
#   mul_18 => mul_19
#   mul_19 => mul_20
#   mul_2 => mul_2
#   mul_20 => mul_21
#   mul_21 => mul_22
#   mul_22 => mul_23
#   mul_23 => mul_24
#   mul_24 => mul_25
#   mul_25 => mul_26
#   mul_26 => mul_27
#   mul_27 => mul_28
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
#   mul_44 => mul_45
#   mul_45 => mul_46
#   mul_46 => mul_47
#   mul_47 => mul_48
#   mul_49 => mul_50
#   mul_50 => mul_51
#   mul_51 => mul_52
#   mul_52 => mul_53
#   mul_6 => mul_7
#   mul_7 => mul_8
#   mul_8 => mul_9
#   mul_9 => mul_10
#   neg => neg
#   neg_1 => neg_1
#   neg_2 => neg_2
#   neg_3 => neg_3
#   power => mul_4
#   r => add_1
#   r_1 => add_8
#   root => div_2
#   root_1 => where
#   root_2 => div_3
#   root_3 => where_2
#   root_4 => div_4
#   root_5 => where_4
#   root_6 => div_5
#   root_7 => where_6
#   speed => select, slice_1
#   sqrt => sqrt
#   sqrt_1 => sqrt_1
#   sqrt_2 => sqrt_2
#   sqrt_3 => sqrt_3
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
#   steer => select_1
#   sub_12 => sub_12
#   sub_13 => sub_13
#   sub_14 => sub_14
#   sub_15 => sub_15
#   sub_16 => sub_16
#   sub_17 => sub_17
#   sub_3 => sub_3
#   sub_8 => sub_8
#   tan => tan
#   tanh => tanh
#   tanh_1 => tanh_1
#   truediv_1 => div_1
#   truediv_2 => mul_3, reciprocal
#   truediv_7 => div_6
#   where_1 => where_1
#   where_3 => where_3
#   where_5 => where_5
#   where_7 => where_7
#   zeros_like_1 => full_default_2
#   zeros_like_2 => full_default_4
#   zeros_like_3 => full_default_6
#   zeros_like_4 => full_default_8
# Graph fragment:
#   %arg0_1 : Tensor "f32[2, 13, 6][78, 6, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[2, 12, 2][24, 2, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[2][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %sub_4 : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=sub_4]
#   %sub_5 : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=sub_5]
#   %add_3 : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=add_3]
#   %sub_9 : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=sub_9]
#   %sub_10 : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=sub_10]
#   %sub_6 : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=sub_6]
#   %add_4 : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=add_4]
#   %sub_11 : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=sub_11]
#   %maximum : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=maximum]
#   %add_6 : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=add_6]
#   %minimum : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=minimum]
#   %add_7 : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=add_7]
#   %clamp_min_9 : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=clamp_min_9]
#   %minimum_1 : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=minimum_1]
#   %full_default : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 12], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_1 : Tensor "f32[2, 12, 6][78, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg0_1, 1, 0, -1), kwargs = {})
#   %select : Tensor "f32[2, 12][78, 6]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.select.int](args = (%slice_1, 2, 3), kwargs = {})
#   %abs_1 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%select,), kwargs = {})
#   %div_1 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_1, 0.05), kwargs = {})
#   %tanh : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_1,), kwargs = {})
#   %mul_1 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 0.1), kwargs = {})
#   %pow_2 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %mul_2 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_2, 0.01), kwargs = {})
#   %add_1 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %mul_13 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.5), kwargs = {})
#   %pow_5 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_13, 2), kwargs = {})
#   %pow_6 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %mul_14 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_6, 0.5192307692307692), kwargs = {})
#   %select_1 : Tensor "f32[2, 12][24, 2]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%arg1_1, 2, 0), kwargs = {})
#   %tan : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tan.default](args = (%select_1,), kwargs = {})
#   %pow_1 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %mul : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.003), kwargs = {})
#   %add : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 0.3302), kwargs = {})
#   %div : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%tan, %add), kwargs = {})
#   %mul_15 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %div), kwargs = {})
#   %pow_7 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_15, 2), kwargs = {})
#   %add_2 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_5, %pow_7), kwargs = {})
#   %unsqueeze : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, 1), kwargs = {})
#   %mul_5 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, 0.92), kwargs = {})
#   %mul_16 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, 9.81), kwargs = {})
#   %mul_17 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, 0.5192307692307692), kwargs = {})
#   %pow_8 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_17, 2), kwargs = {})
#   %sub_4 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %pow_8), kwargs = {})
#   %gt : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%sub_4, 1e-05), kwargs = {})
#   %bitwise_or : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%full_default, %gt), kwargs = {})
#   %mul_30 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.5), kwargs = {})
#   %pow_13 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_30, 2), kwargs = {})
#   %pow_14 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %mul_31 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_14, 0.4807692307692308), kwargs = {})
#   %mul_32 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %div), kwargs = {})
#   %pow_15 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_32, 2), kwargs = {})
#   %add_5 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_13, %pow_15), kwargs = {})
#   %mul_6 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, 1.0), kwargs = {})
#   %mul_33 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, 9.81), kwargs = {})
#   %mul_34 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, 0.4807692307692308), kwargs = {})
#   %pow_16 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_34, 2), kwargs = {})
#   %sub_9 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %pow_16), kwargs = {})
#   %gt_3 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%sub_9, 1e-05), kwargs = {})
#   %bitwise_or_1 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_or, %gt_3), kwargs = {})
#   %sub : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (-5.0, %add_1), kwargs = {})
#   %ge_1 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_4, 0), kwargs = {})
#   %full_default_2 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 12], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_8 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.25), kwargs = {})
#   %pow_4 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_5, 2), kwargs = {})
#   %mul_9 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_4, 9.81), kwargs = {})
#   %mul_10 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, 0.5192307692307692), kwargs = {})
#   %mul_11 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, -0.22410660205935795), kwargs = {})
#   %sub_3 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_8, %mul_11), kwargs = {})
#   %mul_12 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, 2), kwargs = {})
#   %neg : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_12,), kwargs = {})
#   %pow_9 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg, 2), kwargs = {})
#   %mul_7 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, -0.22410660205935795), kwargs = {})
#   %pow_3 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_7, 2), kwargs = {})
#   %sub_2 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_3), kwargs = {})
#   %mul_18 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, 4), kwargs = {})
#   %mul_19 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %sub_4), kwargs = {})
#   %sub_5 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_9, %mul_19), kwargs = {})
#   %ge : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_5, 0), kwargs = {})
#   %clamp_min_1 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_5, 0), kwargs = {})
#   %sqrt : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_1,), kwargs = {})
#   %add_3 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg, %sqrt), kwargs = {})
#   %gt_1 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_3, 1e-12), kwargs = {})
#   %bitwise_and : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge, %gt_1), kwargs = {})
#   %mul_20 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, -2), kwargs = {})
#   %clamp_min_2 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_3, 1e-12), kwargs = {})
#   %div_2 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_20, %clamp_min_2), kwargs = {})
#   %full_default_1 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 12], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and, %div_2, %full_default_1), kwargs = {})
#   %where_1 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_1, %full_default_2, %where), kwargs = {})
#   %neg_1 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_1,), kwargs = {})
#   %maximum : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%sub, %neg_1), kwargs = {})
#   %ge_5 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_9, 0), kwargs = {})
#   %full_default_6 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 12], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_25 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.25), kwargs = {})
#   %pow_12 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_6, 2), kwargs = {})
#   %mul_26 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_12, 9.81), kwargs = {})
#   %mul_27 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, 0.4807692307692308), kwargs = {})
#   %mul_28 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, 0.22410660205935795), kwargs = {})
#   %sub_8 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_25, %mul_28), kwargs = {})
#   %mul_29 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, 2), kwargs = {})
#   %neg_2 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_29,), kwargs = {})
#   %pow_17 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg_2, 2), kwargs = {})
#   %mul_24 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, 0.22410660205935795), kwargs = {})
#   %pow_11 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_24, 2), kwargs = {})
#   %sub_7 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_11), kwargs = {})
#   %mul_35 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, 4), kwargs = {})
#   %mul_36 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, %sub_9), kwargs = {})
#   %sub_10 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_17, %mul_36), kwargs = {})
#   %ge_4 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_10, 0), kwargs = {})
#   %clamp_min_5 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_10, 0), kwargs = {})
#   %sqrt_2 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_5,), kwargs = {})
#   %add_6 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_2, %sqrt_2), kwargs = {})
#   %gt_4 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_6, 1e-12), kwargs = {})
#   %bitwise_and_2 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_4, %gt_4), kwargs = {})
#   %mul_37 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, -2), kwargs = {})
#   %clamp_min_6 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_6, 1e-12), kwargs = {})
#   %div_4 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_37, %clamp_min_6), kwargs = {})
#   %full_default_5 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 12], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_2, %div_4, %full_default_5), kwargs = {})
#   %where_5 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_5, %full_default_6, %where_4), kwargs = {})
#   %neg_3 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_5,), kwargs = {})
#   %maximum_1 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%maximum, %neg_3), kwargs = {})
#   %clamp_min_9 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%maximum_1, -21.045101351351356), kwargs = {})
#   %abs_2 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%select,), kwargs = {})
#   %clamp_min : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%abs_2, 0.001), kwargs = {})
#   %reciprocal : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min,), kwargs = {})
#   %mul_3 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 7.319), kwargs = {})
#   %clamp_max : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_3, 1), kwargs = {})
#   %mul_4 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max, 7.0), kwargs = {})
#   %sub_1 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_4, %add_1), kwargs = {})
#   %ge_3 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_4, 0), kwargs = {})
#   %full_default_4 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 12], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_10 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_12, 2), kwargs = {})
#   %mul_21 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, 4), kwargs = {})
#   %mul_22 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %sub_4), kwargs = {})
#   %sub_6 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_10, %mul_22), kwargs = {})
#   %ge_2 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_6, 0), kwargs = {})
#   %clamp_min_3 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_6, 0), kwargs = {})
#   %sqrt_1 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_3,), kwargs = {})
#   %add_4 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %sqrt_1), kwargs = {})
#   %gt_2 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_4, 1e-12), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_2, %gt_2), kwargs = {})
#   %mul_23 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, -2), kwargs = {})
#   %clamp_min_4 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_4, 1e-12), kwargs = {})
#   %div_3 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_23, %clamp_min_4), kwargs = {})
#   %full_default_3 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 12], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_1, %div_3, %full_default_3), kwargs = {})
#   %where_3 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_3, %full_default_4, %where_2), kwargs = {})
#   %minimum : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%sub_1, %where_3), kwargs = {})
#   %clamp_max_1 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%minimum, 22.72870945945946), kwargs = {})
#   %ge_7 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_9, 0), kwargs = {})
#   %full_default_8 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 12], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_18 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_29, 2), kwargs = {})
#   %mul_38 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, 4), kwargs = {})
#   %mul_39 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, %sub_9), kwargs = {})
#   %sub_11 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_18, %mul_39), kwargs = {})
#   %ge_6 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_11, 0), kwargs = {})
#   %clamp_min_7 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_11, 0), kwargs = {})
#   %sqrt_3 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_7,), kwargs = {})
#   %add_7 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %sqrt_3), kwargs = {})
#   %gt_5 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_7, 1e-12), kwargs = {})
#   %bitwise_and_3 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_6, %gt_5), kwargs = {})
#   %mul_40 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, -2), kwargs = {})
#   %clamp_min_8 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_7, 1e-12), kwargs = {})
#   %div_5 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_40, %clamp_min_8), kwargs = {})
#   %full_default_7 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 12], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_6 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_3, %div_5, %full_default_7), kwargs = {})
#   %where_7 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_7, %full_default_8, %where_6), kwargs = {})
#   %minimum_1 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.minimum.default](args = (%clamp_max_1, %where_7), kwargs = {})
#   %gt_6 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Tensor](args = (%clamp_min_9, %minimum_1), kwargs = {})
#   %bitwise_or_2 : Tensor "b8[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_or_1, %gt_6), kwargs = {})
#   %select_2 : Tensor "f32[2, 12][24, 2]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.select.int](args = (%arg1_1, 2, 1), kwargs = {})
#   %abs_3 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%select,), kwargs = {})
#   %div_6 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_3, 0.05), kwargs = {})
#   %tanh_1 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_6,), kwargs = {})
#   %mul_41 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_1, 0.1), kwargs = {})
#   %pow_19 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %mul_42 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_19, 0.01), kwargs = {})
#   %add_8 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %mul_42), kwargs = {})
#   %add_9 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_2, %add_8), kwargs = {})
#   %mul_45 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, 0.5), kwargs = {})
#   %pow_20 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_45, 2), kwargs = {})
#   %pow_21 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %mul_46 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_21, 0.5192307692307692), kwargs = {})
#   %mul_47 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %div), kwargs = {})
#   %pow_22 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_47, 2), kwargs = {})
#   %add_10 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_20, %pow_22), kwargs = {})
#   %sqrt_4 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_10,), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, 1), kwargs = {})
#   %mul_43 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 0.92), kwargs = {})
#   %mul_48 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, -0.22410660205935795), kwargs = {})
#   %add_11 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, 5.093653846153845), kwargs = {})
#   %clamp_min_10 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_11, 0), kwargs = {})
#   %mul_49 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %clamp_min_10), kwargs = {})
#   %sub_12 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt_4, %mul_49), kwargs = {})
#   %add_12 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_2, %add_8), kwargs = {})
#   %mul_50 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, 0.5), kwargs = {})
#   %pow_23 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_50, 2), kwargs = {})
#   %pow_24 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %mul_51 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_24, 0.4807692307692308), kwargs = {})
#   %mul_52 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, %div), kwargs = {})
#   %pow_25 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_52, 2), kwargs = {})
#   %add_13 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_23, %pow_25), kwargs = {})
#   %sqrt_5 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_13,), kwargs = {})
#   %mul_44 : Tensor "f32[2, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, 1.0), kwargs = {})
#   %mul_53 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, 0.22410660205935795), kwargs = {})
#   %add_14 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, 4.716346153846154), kwargs = {})
#   %clamp_min_12 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_14, 0), kwargs = {})
#   %mul_54 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_44, %clamp_min_12), kwargs = {})
#   %sub_13 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sqrt_5, %mul_54), kwargs = {})
#   %sub_14 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_9, %select_2), kwargs = {})
#   %sub_15 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_2, %minimum_1), kwargs = {})
#   %maximum_2 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%sub_14, %sub_15), kwargs = {})
#   %clamp_min_14 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%maximum_2, 0), kwargs = {})
#   %slice_2 : Tensor "f32[2, 12, 6][78, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg0_1, 1, 0, -1), kwargs = {})
#   %select_3 : Tensor "f32[2, 12][78, 6]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%slice_2, 2, 4), kwargs = {})
#   %sub_16 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_1, %select_3), kwargs = {})
#   %abs_4 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_16,), kwargs = {})
#   %sub_17 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%abs_4, 0.16000000000000003), kwargs = {})
#   %clamp_min_15 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_17, 0), kwargs = {})
#   return %sub_4,%sub_9,%sub_5,%add_3,%maximum,%sub_10,%add_6,%sub_6,%add_4,%minimum,%sub_11,%add_7,%sub_12,%sub_13,%clamp_min_9,%minimum_1,%bitwise_or_2,%clamp_min_14,%clamp_min_15
triton_poi_fused_abs_add_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_slice_sqrt_sub_tan_tanh_unsqueeze_where_zeros_like_0 = async_compile.triton('triton_poi_fused_abs_add_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_slice_sqrt_sub_tan_tanh_unsqueeze_where_zeros_like_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*i1', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_slice_sqrt_sub_tan_tanh_unsqueeze_where_zeros_like_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 7, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_slice_sqrt_sub_tan_tanh_unsqueeze_where_zeros_like_0(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 12)
    x1 = xindex // 12
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + 6*x0 + 78*x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (2*x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp130 = tl.load(in_ptr1 + (1 + 2*x2), xmask, eviction_policy='evict_last')
    tmp185 = tl.load(in_ptr0 + (4 + 6*x0 + 78*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 20.0
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.tanh(tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp0
    tmp8 = 0.01
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = 0.5
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = 0.5192307692307692
    tmp15 = tmp7 * tmp14
    tmp17 = libdevice.tan(tmp16)
    tmp18 = 0.003
    tmp19 = tmp7 * tmp18
    tmp20 = 0.3302
    tmp21 = tmp19 + tmp20
    tmp22 = (tmp17 / tmp21)
    tmp23 = tmp15 * tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tmp13 + tmp24
    tmp27 = 0.92
    tmp28 = tmp26 * tmp27
    tmp29 = 9.81
    tmp30 = tmp28 * tmp29
    tmp31 = tmp30 * tmp14
    tmp32 = tmp31 * tmp31
    tmp33 = tmp25 - tmp32
    tmp34 = 0.4807692307692308
    tmp35 = tmp7 * tmp34
    tmp36 = tmp35 * tmp22
    tmp37 = tmp36 * tmp36
    tmp38 = tmp13 + tmp37
    tmp39 = 1.0
    tmp40 = tmp26 * tmp39
    tmp41 = tmp40 * tmp29
    tmp42 = tmp41 * tmp34
    tmp43 = tmp42 * tmp42
    tmp44 = tmp38 - tmp43
    tmp45 = 0.25
    tmp46 = tmp10 * tmp45
    tmp47 = tmp28 * tmp28
    tmp48 = tmp47 * tmp29
    tmp49 = tmp48 * tmp14
    tmp50 = -0.22410660205935795
    tmp51 = tmp49 * tmp50
    tmp52 = tmp46 - tmp51
    tmp53 = 2.0
    tmp54 = tmp52 * tmp53
    tmp55 = -tmp54
    tmp56 = tmp55 * tmp55
    tmp57 = tmp28 * tmp50
    tmp58 = tmp57 * tmp57
    tmp59 = tmp45 - tmp58
    tmp60 = 4.0
    tmp61 = tmp59 * tmp60
    tmp62 = tmp61 * tmp33
    tmp63 = tmp56 - tmp62
    tmp64 = 0.0
    tmp65 = triton_helpers.maximum(tmp63, tmp64)
    tmp66 = tl.sqrt_rn(tmp65)
    tmp67 = tmp55 + tmp66
    tmp68 = -5.0
    tmp69 = tmp68 - tmp10
    tmp70 = tmp33 >= tmp64
    tmp71 = tmp63 >= tmp64
    tmp72 = 1e-12
    tmp73 = tmp67 > tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = -2.0
    tmp76 = tmp33 * tmp75
    tmp77 = triton_helpers.maximum(tmp67, tmp72)
    tmp78 = (tmp76 / tmp77)
    tmp79 = float("inf")
    tmp80 = tl.where(tmp74, tmp78, tmp79)
    tmp81 = tl.where(tmp70, tmp64, tmp80)
    tmp82 = -tmp81
    tmp83 = triton_helpers.maximum(tmp69, tmp82)
    tmp84 = tmp40 * tmp40
    tmp85 = tmp84 * tmp29
    tmp86 = tmp85 * tmp34
    tmp87 = 0.22410660205935795
    tmp88 = tmp86 * tmp87
    tmp89 = tmp46 - tmp88
    tmp90 = tmp89 * tmp53
    tmp91 = -tmp90
    tmp92 = tmp91 * tmp91
    tmp93 = tmp40 * tmp87
    tmp94 = tmp93 * tmp93
    tmp95 = tmp45 - tmp94
    tmp96 = tmp95 * tmp60
    tmp97 = tmp96 * tmp44
    tmp98 = tmp92 - tmp97
    tmp99 = triton_helpers.maximum(tmp98, tmp64)
    tmp100 = tl.sqrt_rn(tmp99)
    tmp101 = tmp91 + tmp100
    tmp102 = tmp54 * tmp54
    tmp103 = tmp102 - tmp62
    tmp104 = triton_helpers.maximum(tmp103, tmp64)
    tmp105 = tl.sqrt_rn(tmp104)
    tmp106 = tmp54 + tmp105
    tmp107 = 0.001
    tmp108 = triton_helpers.maximum(tmp1, tmp107)
    tmp109 = tl.full([1], 1, tl.int32)
    tmp110 = (tmp109 / tmp108)
    tmp111 = 7.319
    tmp112 = tmp110 * tmp111
    tmp113 = triton_helpers.minimum(tmp112, tmp39)
    tmp114 = 7.0
    tmp115 = tmp113 * tmp114
    tmp116 = tmp115 - tmp10
    tmp117 = tmp103 >= tmp64
    tmp118 = tmp106 > tmp72
    tmp119 = tmp117 & tmp118
    tmp120 = triton_helpers.maximum(tmp106, tmp72)
    tmp121 = (tmp76 / tmp120)
    tmp122 = tl.where(tmp119, tmp121, tmp79)
    tmp123 = tl.where(tmp70, tmp64, tmp122)
    tmp124 = triton_helpers.minimum(tmp116, tmp123)
    tmp125 = tmp90 * tmp90
    tmp126 = tmp125 - tmp97
    tmp127 = triton_helpers.maximum(tmp126, tmp64)
    tmp128 = tl.sqrt_rn(tmp127)
    tmp129 = tmp90 + tmp128
    tmp131 = tmp130 + tmp10
    tmp132 = tmp131 * tmp11
    tmp133 = tmp132 * tmp132
    tmp134 = tmp133 + tmp24
    tmp135 = tl.sqrt_rn(tmp134)
    tmp136 = tmp130 * tmp50
    tmp137 = 5.093653846153845
    tmp138 = tmp136 + tmp137
    tmp139 = triton_helpers.maximum(tmp138, tmp64)
    tmp140 = tmp28 * tmp139
    tmp141 = tmp135 - tmp140
    tmp142 = tmp133 + tmp37
    tmp143 = tl.sqrt_rn(tmp142)
    tmp144 = tmp130 * tmp87
    tmp145 = 4.716346153846154
    tmp146 = tmp144 + tmp145
    tmp147 = triton_helpers.maximum(tmp146, tmp64)
    tmp148 = tmp40 * tmp147
    tmp149 = tmp143 - tmp148
    tmp150 = tmp44 >= tmp64
    tmp151 = tmp98 >= tmp64
    tmp152 = tmp101 > tmp72
    tmp153 = tmp151 & tmp152
    tmp154 = tmp44 * tmp75
    tmp155 = triton_helpers.maximum(tmp101, tmp72)
    tmp156 = (tmp154 / tmp155)
    tmp157 = tl.where(tmp153, tmp156, tmp79)
    tmp158 = tl.where(tmp150, tmp64, tmp157)
    tmp159 = -tmp158
    tmp160 = triton_helpers.maximum(tmp83, tmp159)
    tmp161 = -21.045101351351356
    tmp162 = triton_helpers.maximum(tmp160, tmp161)
    tmp163 = 22.72870945945946
    tmp164 = triton_helpers.minimum(tmp124, tmp163)
    tmp165 = tmp126 >= tmp64
    tmp166 = tmp129 > tmp72
    tmp167 = tmp165 & tmp166
    tmp168 = triton_helpers.maximum(tmp129, tmp72)
    tmp169 = (tmp154 / tmp168)
    tmp170 = tl.where(tmp167, tmp169, tmp79)
    tmp171 = tl.where(tmp150, tmp64, tmp170)
    tmp172 = triton_helpers.minimum(tmp164, tmp171)
    tmp173 = 1e-05
    tmp174 = tmp33 > tmp173
    tmp175 = tl.full([1], False, tl.int1)
    tmp176 = tmp175 | tmp174
    tmp177 = tmp44 > tmp173
    tmp178 = tmp176 | tmp177
    tmp179 = tmp162 > tmp172
    tmp180 = tmp178 | tmp179
    tmp181 = tmp162 - tmp130
    tmp182 = tmp130 - tmp172
    tmp183 = triton_helpers.maximum(tmp181, tmp182)
    tmp184 = triton_helpers.maximum(tmp183, tmp64)
    tmp186 = tmp16 - tmp185
    tmp187 = tl_math.abs(tmp186)
    tmp188 = 0.16000000000000003
    tmp189 = tmp187 - tmp188
    tmp190 = triton_helpers.maximum(tmp189, tmp64)
    tl.store(out_ptr8 + (x2), tmp141, xmask)
    tl.store(out_ptr9 + (x2), tmp149, xmask)
    tl.store(in_out_ptr0 + (x2), tmp162, xmask)
    tl.store(in_out_ptr1 + (x2), tmp172, xmask)
    tl.store(out_ptr10 + (x2), tmp180, xmask)
    tl.store(out_ptr11 + (x2), tmp184, xmask)
    tl.store(out_ptr12 + (x2), tmp190, xmask)
''', device_str='cuda')


# kernel path: /home/shchon11/F1tenth/F1tenth_E2E/work/adaptive-racing-v3/controller-proof/inductor-cold-v1/jx/cjx2jzrl7vsbkkqezlxkcrw5377nq4u3tzixibiyvyqvq37n5fmk.py
# Topologically Sorted Source Nodes: [clamp_min_10, stack, clamp_min_12], Original ATen: [aten.clamp_min, aten.stack]
# Source node to ATen node mapping:
#   clamp_min_10 => clamp_min_11
#   clamp_min_12 => clamp_min_13
#   stack => cat, unsqueeze_2, unsqueeze_3
# Graph fragment:
#   %sub_12 : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=sub_12]
#   %sub_13 : Tensor "f32[2, 12][12, 1]cuda:0" = PlaceHolder[target=sub_13]
#   %clamp_min_11 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_12, 0), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[2, 12, 1][12, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%clamp_min_11, 2), kwargs = {})
#   %clamp_min_13 : Tensor "f32[2, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_13, 0), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[2, 12, 1][12, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%clamp_min_13, 2), kwargs = {})
#   %cat : Tensor "f32[2, 12, 2][24, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_2, %unsqueeze_3], -1), kwargs = {})
#   return %cat
triton_poi_fused_clamp_min_stack_1 = async_compile.triton('triton_poi_fused_clamp_min_stack_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_min_stack_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 384}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_min_stack_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 2, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr1 + (x1), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tl.store(out_ptr0 + (x2), tmp18, xmask)
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
        assert_size_stride(arg0_1, (2, 13, 6), (78, 6, 1))
        assert_size_stride(arg1_1, (2, 12, 2), (24, 2, 1))
        assert_size_stride(arg2_1, (2, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf2 = empty_strided_cuda((2, 12), (12, 1), torch.float32)
            buf4 = buf2; del buf2  # reuse
            buf8 = empty_strided_cuda((2, 12), (12, 1), torch.float32)
            buf10 = buf8; del buf8  # reuse
            buf15 = empty_strided_cuda((2, 12), (12, 1), torch.float32)
            buf16 = empty_strided_cuda((2, 12), (12, 1), torch.float32)
            buf7 = buf4; del buf4  # reuse
            buf13 = buf10; del buf10  # reuse
            buf14 = empty_strided_cuda((2, 12), (12, 1), torch.bool)
            buf18 = empty_strided_cuda((2, 12), (12, 1), torch.float32)
            buf19 = empty_strided_cuda((2, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [bad, speed, abs_1, truediv_1, tanh, mul_1, square_1, mul_2, r, mul_12, square_4, square_5, mul_13, steer, tan, square, mul, add, kap, mul_14, square_6, add_2, getitem_3, grip, mul_15, mul_16, square_7, cc, gt, bad_1, mul_29, square_12, square_13, mul_30, mul_31, square_14, add_5, grip_1, mul_32, mul_33, square_15, cc_1, gt_3, bad_2, lo, ge_1, zeros_like_1, mul_7, square_3, mul_8, mul_9, mul_10, sub_3, bb, neg, square_8, mul_6, square_2, aa, mul_17, mul_18, disc, ge, clamp_min_1, sqrt, den, gt_1, and_, mul_19, clamp_min_2, root, full_like, root_1, where_1, neg_1, lo_1, ge_5, zeros_like_3, mul_24, square_11, mul_25, mul_26, mul_27, sub_8, bb_1, neg_2, square_16, mul_23, square_10, aa_1, mul_34, mul_35, disc_2, ge_4, clamp_min_5, sqrt_2, den_2, gt_4, and__2, mul_36, clamp_min_6, root_4, full_like_2, root_5, where_5, neg_3, lo_2, lo_3, abs_2, clamp_min, truediv_2, clamp, power, hi, ge_3, zeros_like_2, square_9, mul_20, mul_21, disc_1, ge_2, clamp_min_3, sqrt_1, den_1, gt_2, and__1, mul_22, clamp_min_4, root_2, full_like_1, root_3, where_3, hi_1, hi_2, ge_7, zeros_like_4, square_17, mul_37, mul_38, disc_3, ge_6, clamp_min_7, sqrt_3, den_3, gt_5, and__3, mul_39, clamp_min_8, root_6, full_like_3, root_7, where_7, hi_3, gt_6, bad_3, ax, abs_3, truediv_7, tanh_1, mul_40, square_18, mul_41, r_1, add_9, mul_44, square_19, square_20, mul_45, mul_46, square_21, add_10, demand, getitem_4, grip_2, mul_47, add_11, clamp_min_9, available, sub_12, add_12, mul_49, square_22, square_23, mul_50, mul_51, square_24, add_13, demand_1, grip_3, mul_52, add_14, clamp_min_11, available_1, sub_13, sub_14, sub_15, maximum_2, clamp_min_13, getitem_5, sub_16, abs_4, sub_17, clamp_min_14], Original ATen: [aten.zeros_like, aten.slice, aten.select, aten.abs, aten.div, aten.tanh, aten.mul, aten.pow, aten.add, aten.tan, aten.unsqueeze, aten.sub, aten.gt, aten.bitwise_or, aten.rsub, aten.ge, aten.neg, aten.clamp_min, aten.sqrt, aten.bitwise_and, aten.full_like, aten.where, aten.maximum, aten.clamp, aten.reciprocal, aten.minimum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_abs_add_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_slice_sqrt_sub_tan_tanh_unsqueeze_where_zeros_like_0.run(buf7, buf13, arg0_1, arg1_1, arg2_1, buf15, buf16, buf14, buf18, buf19, 24, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            buf17 = empty_strided_cuda((2, 12, 2), (24, 2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [clamp_min_10, stack, clamp_min_12], Original ATen: [aten.clamp_min, aten.stack]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clamp_min_stack_1.run(buf15, buf16, buf17, 48, stream=stream0)
            del buf15
            del buf16
        return (buf7, buf13, buf14, buf17, buf18, buf19, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 13, 6), (78, 6, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 12, 2), (24, 2, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
