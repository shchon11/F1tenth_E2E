# AOT ID: ['8_inference']
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


# kernel path: /home/shchon11/F1tenth/F1tenth_E2E/work/adaptive-racing-v3/controller-proof/inductor-cold-v1/p7/cp7isbballzbrnvxwg26funqhseg3thmczxx7cekx3nrollxhezk.py
# Topologically Sorted Source Nodes: [curve, mul, grip, mul_4, mul_5, square_1, cc, expand_as, ge_1, zeros_like, square_2, mul_3, square, aa, mul_6, mul_7, disc, ge, bb, clamp_min, sqrt, den, gt, and_, mul_8, clamp_min_1, root, full_like_2, root_1, where_1, curve_1, grip_1, mul_10, mul_11, square_4, cc_1, expand_as_1, ge_3, zeros_like_1, square_5, mul_9, square_3, aa_1, mul_12, mul_13, disc_1, ge_2, bb_1, clamp_min_2, sqrt_1, den_1, gt_1, and__1, mul_14, clamp_min_3, root_2, full_like_4, root_3, where_3, curve_2, clamp_min_4, curve_3, add_4, middle, abs_1, truediv_2, tanh, mul_16, square_6, mul_17, add_5, clamp_min_5, truediv_3, clamp, mul_18, sustainable, low, low_1, high_1, add_6, middle_1, abs_2, truediv_4, tanh_1, mul_20, square_7, mul_21, add_7, clamp_min_6, truediv_5, clamp_1, mul_22, sustainable_1, low_2, high_2, add_8, middle_2, abs_3, truediv_6, tanh_2, mul_24, square_8, mul_25, add_9, clamp_min_7, truediv_7, clamp_2, mul_26, sustainable_2, low_3, high_3, add_10, middle_3, abs_4, truediv_8, tanh_3, mul_28, square_9, mul_29, add_11, clamp_min_8, truediv_9, clamp_3, mul_30, sustainable_3, low_4, high_4, add_12, middle_4, abs_5, truediv_10, tanh_4, mul_32, square_10, mul_33, add_13, clamp_min_9, truediv_11, clamp_4, mul_34, sustainable_4, low_5, high_5, add_14, middle_5, abs_6, truediv_12, tanh_5, mul_36, square_11, mul_37, add_15, clamp_min_10, truediv_13, clamp_5, mul_38, sustainable_5, low_6, high_6, add_16, middle_6, abs_7, truediv_14, tanh_6, mul_40, square_12, mul_41, add_17, clamp_min_11, truediv_15, clamp_6, mul_42, sustainable_6, low_7, high_7, add_18, middle_7, abs_8, truediv_16, tanh_7, mul_44, square_13, mul_45, add_19, clamp_min_12, truediv_17, clamp_7, mul_46, sustainable_7, low_8, high_8, add_20, middle_8, abs_9, truediv_18, tanh_8, mul_48, square_14, mul_49, add_21, clamp_min_13, truediv_19, clamp_8, mul_50, sustainable_8, low_9, high_9, add_22, middle_9, abs_10, truediv_20, tanh_9, mul_52, square_15, mul_53, add_23, clamp_min_14, truediv_21, clamp_9, mul_54, sustainable_9, low_10, high_10, add_24, middle_10, abs_11, truediv_22, tanh_10, mul_56, square_16, mul_57, add_25, clamp_min_15, truediv_23, clamp_10, mul_58, sustainable_10, low_11, high_11, add_26, middle_11, abs_12, truediv_24, tanh_11, mul_60, square_17, mul_61, add_27, clamp_min_16, truediv_25, clamp_11, mul_62, sustainable_11, high_12], Original ATen: [aten.full_like, aten.mul, aten.pow, aten.rsub, aten.expand, aten.ge, aten.zeros_like, aten.add, aten.sub, aten.clamp_min, aten.sqrt, aten.gt, aten.bitwise_and, aten.div, aten.where, aten.minimum, aten.abs, aten.tanh, aten.reciprocal, aten.clamp, aten.le]
# Source node to ATen node mapping:
#   aa => add
#   aa_1 => add_2
#   abs_1 => abs_1
#   abs_10 => abs_10
#   abs_11 => abs_11
#   abs_12 => abs_12
#   abs_2 => abs_2
#   abs_3 => abs_3
#   abs_4 => abs_4
#   abs_5 => abs_5
#   abs_6 => abs_6
#   abs_7 => abs_7
#   abs_8 => abs_8
#   abs_9 => abs_9
#   add_10 => add_10
#   add_11 => add_11
#   add_12 => add_12
#   add_13 => add_13
#   add_14 => add_14
#   add_15 => add_15
#   add_16 => add_16
#   add_17 => add_17
#   add_18 => add_18
#   add_19 => add_19
#   add_20 => add_20
#   add_21 => add_21
#   add_22 => add_22
#   add_23 => add_23
#   add_24 => add_24
#   add_25 => add_25
#   add_26 => add_26
#   add_27 => add_27
#   add_4 => minimum_2
#   add_5 => add_5
#   add_6 => add_6
#   add_7 => add_7
#   add_8 => add_8
#   add_9 => add_9
#   and_ => bitwise_and
#   and__1 => bitwise_and_1
#   bb => full_default_1
#   bb_1 => full_default_5
#   cc => sub
#   cc_1 => sub_2
#   clamp => clamp_max
#   clamp_1 => clamp_max_1
#   clamp_10 => clamp_max_10
#   clamp_11 => clamp_max_11
#   clamp_2 => clamp_max_2
#   clamp_3 => clamp_max_3
#   clamp_4 => clamp_max_4
#   clamp_5 => clamp_max_5
#   clamp_6 => clamp_max_6
#   clamp_7 => clamp_max_7
#   clamp_8 => clamp_max_8
#   clamp_9 => clamp_max_9
#   clamp_min => clamp_min
#   clamp_min_1 => clamp_min_1
#   clamp_min_10 => clamp_min_10
#   clamp_min_11 => clamp_min_11
#   clamp_min_12 => clamp_min_12
#   clamp_min_13 => clamp_min_13
#   clamp_min_14 => clamp_min_14
#   clamp_min_15 => clamp_min_15
#   clamp_min_16 => clamp_min_16
#   clamp_min_2 => clamp_min_2
#   clamp_min_3 => clamp_min_3
#   clamp_min_4 => clamp_min_4
#   clamp_min_5 => clamp_min_5
#   clamp_min_6 => clamp_min_6
#   clamp_min_7 => clamp_min_7
#   clamp_min_8 => clamp_min_8
#   clamp_min_9 => clamp_min_9
#   curve => full_default
#   curve_1 => minimum
#   curve_2 => minimum_1
#   curve_3 => sqrt_2
#   den => add_1
#   den_1 => add_3
#   disc => sub_1
#   disc_1 => sub_3
#   expand_as => expand
#   expand_as_1 => expand_1
#   full_like_2 => full_default_3
#   full_like_4 => full_default_7
#   ge => ge
#   ge_1 => ge_1
#   ge_2 => ge_2
#   ge_3 => ge_3
#   grip => mul_1
#   grip_1 => mul_2
#   gt => gt
#   gt_1 => gt_1
#   high_1 => where_5
#   high_10 => where_23
#   high_11 => where_25
#   high_12 => where_27
#   high_2 => where_7
#   high_3 => where_9
#   high_4 => where_11
#   high_5 => where_13
#   high_6 => where_15
#   high_7 => where_17
#   high_8 => where_19
#   high_9 => where_21
#   low => full_default_9
#   low_1 => where_4
#   low_10 => where_22
#   low_11 => where_24
#   low_2 => where_6
#   low_3 => where_8
#   low_4 => where_10
#   low_5 => where_12
#   low_6 => where_14
#   low_7 => where_16
#   low_8 => where_18
#   low_9 => where_20
#   middle => mul_15
#   middle_1 => mul_20
#   middle_10 => mul_65
#   middle_11 => mul_70
#   middle_2 => mul_25
#   middle_3 => mul_30
#   middle_4 => mul_35
#   middle_5 => mul_40
#   middle_6 => mul_45
#   middle_7 => mul_50
#   middle_8 => mul_55
#   middle_9 => mul_60
#   mul => mul
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_16 => mul_16
#   mul_17 => mul_17
#   mul_18 => mul_19
#   mul_20 => mul_21
#   mul_21 => mul_22
#   mul_22 => mul_24
#   mul_24 => mul_26
#   mul_25 => mul_27
#   mul_26 => mul_29
#   mul_28 => mul_31
#   mul_29 => mul_32
#   mul_3 => mul_3
#   mul_30 => mul_34
#   mul_32 => mul_36
#   mul_33 => mul_37
#   mul_34 => mul_39
#   mul_36 => mul_41
#   mul_37 => mul_42
#   mul_38 => mul_44
#   mul_4 => mul_4
#   mul_40 => mul_46
#   mul_41 => mul_47
#   mul_42 => mul_49
#   mul_44 => mul_51
#   mul_45 => mul_52
#   mul_46 => mul_54
#   mul_48 => mul_56
#   mul_49 => mul_57
#   mul_5 => mul_5
#   mul_50 => mul_59
#   mul_52 => mul_61
#   mul_53 => mul_62
#   mul_54 => mul_64
#   mul_56 => mul_66
#   mul_57 => mul_67
#   mul_58 => mul_69
#   mul_6 => mul_6
#   mul_60 => mul_71
#   mul_61 => mul_72
#   mul_62 => mul_74
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   root => div
#   root_1 => where
#   root_2 => div_1
#   root_3 => where_2
#   sqrt => sqrt
#   sqrt_1 => sqrt_1
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
#   square_2 => full_default_2
#   square_3 => pow_4
#   square_4 => pow_5
#   square_5 => full_default_6
#   square_6 => pow_7
#   square_7 => pow_8
#   square_8 => pow_9
#   square_9 => pow_10
#   sustainable => le
#   sustainable_1 => le_1
#   sustainable_10 => le_10
#   sustainable_11 => le_11
#   sustainable_2 => le_2
#   sustainable_3 => le_3
#   sustainable_4 => le_4
#   sustainable_5 => le_5
#   sustainable_6 => le_6
#   sustainable_7 => le_7
#   sustainable_8 => le_8
#   sustainable_9 => le_9
#   tanh => tanh
#   tanh_1 => tanh_1
#   tanh_10 => tanh_10
#   tanh_11 => tanh_11
#   tanh_2 => tanh_2
#   tanh_3 => tanh_3
#   tanh_4 => tanh_4
#   tanh_5 => tanh_5
#   tanh_6 => tanh_6
#   tanh_7 => tanh_7
#   tanh_8 => tanh_8
#   tanh_9 => tanh_9
#   truediv_10 => div_6
#   truediv_11 => mul_38, reciprocal_4
#   truediv_12 => div_7
#   truediv_13 => mul_43, reciprocal_5
#   truediv_14 => div_8
#   truediv_15 => mul_48, reciprocal_6
#   truediv_16 => div_9
#   truediv_17 => mul_53, reciprocal_7
#   truediv_18 => div_10
#   truediv_19 => mul_58, reciprocal_8
#   truediv_2 => div_2
#   truediv_20 => div_11
#   truediv_21 => mul_63, reciprocal_9
#   truediv_22 => div_12
#   truediv_23 => mul_68, reciprocal_10
#   truediv_24 => div_13
#   truediv_25 => mul_73, reciprocal_11
#   truediv_3 => mul_18, reciprocal
#   truediv_4 => div_3
#   truediv_5 => mul_23, reciprocal_1
#   truediv_6 => div_4
#   truediv_7 => mul_28, reciprocal_2
#   truediv_8 => div_5
#   truediv_9 => mul_33, reciprocal_3
#   where_1 => where_1
#   where_3 => where_3
#   zeros_like => full_default_4
#   zeros_like_1 => full_default_8
# Graph fragment:
#   %arg1_1 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg0_1 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sqrt_2 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=sqrt_2]
#   %abs_2 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=abs_2]
#   %pow_8 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=pow_8]
#   %clamp_min_6 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_6]
#   %where_6 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_6]
#   %where_7 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_7]
#   %div_5 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=div_5]
#   %mul_32 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=mul_32]
#   %reciprocal_3 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=reciprocal_3]
#   %where_10 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_10]
#   %where_11 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_11]
#   %div_7 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=div_7]
#   %mul_42 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=mul_42]
#   %reciprocal_5 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=reciprocal_5]
#   %where_14 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_14]
#   %where_15 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_15]
#   %div_9 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=div_9]
#   %mul_52 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=mul_52]
#   %reciprocal_7 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=reciprocal_7]
#   %where_18 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_18]
#   %where_19 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_19]
#   %div_11 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=div_11]
#   %mul_62 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=mul_62]
#   %reciprocal_9 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=reciprocal_9]
#   %where_22 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_22]
#   %where_23 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_23]
#   %div_13 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=div_13]
#   %mul_72 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=mul_72]
#   %reciprocal_11 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=reciprocal_11]
#   %full_default : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 100.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 0.85), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 0.92), kwargs = {})
#   %mul_4 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, 9.81), kwargs = {})
#   %mul_5 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, 0.5192307692307692), kwargs = {})
#   %pow_2 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_5, 2), kwargs = {})
#   %sub : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.0025000000000000005, %pow_2), kwargs = {})
#   %expand : Tensor "f32[1, 25][1, 0]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.expand.default](args = (%sub, [1, 25]), kwargs = {})
#   %ge_1 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%expand, 0), kwargs = {})
#   %full_default_4 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 2.500000277905201e-07), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, 0.5192307692307692), kwargs = {})
#   %pow_1 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_3, 2), kwargs = {})
#   %add : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, 2.5e-05), kwargs = {})
#   %mul_6 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 4), kwargs = {})
#   %mul_7 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %expand), kwargs = {})
#   %sub_1 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_2, %mul_7), kwargs = {})
#   %ge : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_1, 0), kwargs = {})
#   %full_default_1 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 0.0005), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clamp_min : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_1, 0), kwargs = {})
#   %sqrt : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min,), kwargs = {})
#   %add_1 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_1, %sqrt), kwargs = {})
#   %gt : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 1e-12), kwargs = {})
#   %bitwise_and : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge, %gt), kwargs = {})
#   %mul_8 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, -2), kwargs = {})
#   %clamp_min_1 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_1, 1e-12), kwargs = {})
#   %div : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_8, %clamp_min_1), kwargs = {})
#   %full_default_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and, %div, %full_default_3), kwargs = {})
#   %where_1 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_1, %full_default_4, %where), kwargs = {})
#   %minimum : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %where_1), kwargs = {})
#   %mul_2 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 1.0), kwargs = {})
#   %mul_10 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, 9.81), kwargs = {})
#   %mul_11 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, 0.4807692307692308), kwargs = {})
#   %pow_5 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_11, 2), kwargs = {})
#   %sub_2 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.0025000000000000005, %pow_5), kwargs = {})
#   %expand_1 : Tensor "f32[1, 25][1, 0]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.expand.default](args = (%sub_2, [1, 25]), kwargs = {})
#   %ge_3 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%expand_1, 0), kwargs = {})
#   %full_default_8 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_6 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 2.500000277905201e-07), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_9 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, 0.4807692307692308), kwargs = {})
#   %pow_4 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_9, 2), kwargs = {})
#   %add_2 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_4, 2.5e-05), kwargs = {})
#   %mul_12 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 4), kwargs = {})
#   %mul_13 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %expand_1), kwargs = {})
#   %sub_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_6, %mul_13), kwargs = {})
#   %ge_2 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_3, 0), kwargs = {})
#   %full_default_5 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 0.0005), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clamp_min_2 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_3, 0), kwargs = {})
#   %sqrt_1 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_2,), kwargs = {})
#   %add_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_5, %sqrt_1), kwargs = {})
#   %gt_1 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_3, 1e-12), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_2, %gt_1), kwargs = {})
#   %mul_14 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_1, -2), kwargs = {})
#   %clamp_min_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_3, 1e-12), kwargs = {})
#   %div_1 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_14, %clamp_min_3), kwargs = {})
#   %full_default_7 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_1, %div_1, %full_default_7), kwargs = {})
#   %where_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_3, %full_default_8, %where_2), kwargs = {})
#   %minimum_1 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%minimum, %where_3), kwargs = {})
#   %clamp_min_4 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%minimum_1, 0), kwargs = {})
#   %sqrt_2 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_4,), kwargs = {})
#   %minimum_2 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.minimum.default](args = (%arg0_1, %sqrt_2), kwargs = {})
#   %mul_15 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%minimum_2, 0.5), kwargs = {})
#   %abs_1 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_15,), kwargs = {})
#   %div_2 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_1, 0.05), kwargs = {})
#   %tanh : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_2,), kwargs = {})
#   %mul_16 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 0.1), kwargs = {})
#   %pow_7 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_15, 2), kwargs = {})
#   %mul_17 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_7, 0.01), kwargs = {})
#   %add_5 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %mul_17), kwargs = {})
#   %clamp_min_5 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_15, 0.001), kwargs = {})
#   %reciprocal : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_5,), kwargs = {})
#   %mul_18 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 7.319), kwargs = {})
#   %clamp_max : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_18, 1), kwargs = {})
#   %mul_19 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max, 7.0), kwargs = {})
#   %le : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_5, %mul_19), kwargs = {})
#   %full_default_9 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %mul_15, %full_default_9), kwargs = {})
#   %where_5 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %minimum_2, %mul_15), kwargs = {})
#   %add_6 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_4, %where_5), kwargs = {})
#   %mul_20 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 0.5), kwargs = {})
#   %abs_2 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_20,), kwargs = {})
#   %div_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_2, 0.05), kwargs = {})
#   %tanh_1 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_3,), kwargs = {})
#   %mul_21 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_1, 0.1), kwargs = {})
#   %pow_8 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_20, 2), kwargs = {})
#   %mul_22 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_8, 0.01), kwargs = {})
#   %add_7 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %mul_22), kwargs = {})
#   %clamp_min_6 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_20, 0.001), kwargs = {})
#   %reciprocal_1 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_6,), kwargs = {})
#   %mul_23 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 7.319), kwargs = {})
#   %clamp_max_1 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_23, 1), kwargs = {})
#   %mul_24 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_1, 7.0), kwargs = {})
#   %le_1 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_7, %mul_24), kwargs = {})
#   %where_6 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %mul_20, %where_4), kwargs = {})
#   %where_7 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %where_5, %mul_20), kwargs = {})
#   %add_8 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_6, %where_7), kwargs = {})
#   %mul_25 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 0.5), kwargs = {})
#   %abs_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_25,), kwargs = {})
#   %div_4 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_3, 0.05), kwargs = {})
#   %tanh_2 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_4,), kwargs = {})
#   %mul_26 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_2, 0.1), kwargs = {})
#   %pow_9 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_25, 2), kwargs = {})
#   %mul_27 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_9, 0.01), kwargs = {})
#   %add_9 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %mul_27), kwargs = {})
#   %clamp_min_7 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_25, 0.001), kwargs = {})
#   %reciprocal_2 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_7,), kwargs = {})
#   %mul_28 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 7.319), kwargs = {})
#   %clamp_max_2 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_28, 1), kwargs = {})
#   %mul_29 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_2, 7.0), kwargs = {})
#   %le_2 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_9, %mul_29), kwargs = {})
#   %where_8 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_2, %mul_25, %where_6), kwargs = {})
#   %where_9 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_2, %where_7, %mul_25), kwargs = {})
#   %add_10 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_8, %where_9), kwargs = {})
#   %mul_30 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 0.5), kwargs = {})
#   %abs_4 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_30,), kwargs = {})
#   %div_5 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_4, 0.05), kwargs = {})
#   %tanh_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_5,), kwargs = {})
#   %mul_31 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_3, 0.1), kwargs = {})
#   %pow_10 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_30, 2), kwargs = {})
#   %mul_32 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_10, 0.01), kwargs = {})
#   %add_11 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %mul_32), kwargs = {})
#   %clamp_min_8 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_30, 0.001), kwargs = {})
#   %reciprocal_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_8,), kwargs = {})
#   %mul_33 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 7.319), kwargs = {})
#   %clamp_max_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_33, 1), kwargs = {})
#   %mul_34 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_3, 7.0), kwargs = {})
#   %le_3 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_11, %mul_34), kwargs = {})
#   %where_10 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %mul_30, %where_8), kwargs = {})
#   %where_11 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %where_9, %mul_30), kwargs = {})
#   %add_12 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_10, %where_11), kwargs = {})
#   %mul_35 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, 0.5), kwargs = {})
#   %abs_5 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_35,), kwargs = {})
#   %div_6 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_5, 0.05), kwargs = {})
#   %tanh_4 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_6,), kwargs = {})
#   %mul_36 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_4, 0.1), kwargs = {})
#   %pow_11 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_35, 2), kwargs = {})
#   %mul_37 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_11, 0.01), kwargs = {})
#   %add_13 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_36, %mul_37), kwargs = {})
#   %clamp_min_9 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_35, 0.001), kwargs = {})
#   %reciprocal_4 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_9,), kwargs = {})
#   %mul_38 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, 7.319), kwargs = {})
#   %clamp_max_4 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_38, 1), kwargs = {})
#   %mul_39 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_4, 7.0), kwargs = {})
#   %le_4 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_13, %mul_39), kwargs = {})
#   %where_12 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_4, %mul_35, %where_10), kwargs = {})
#   %where_13 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_4, %where_11, %mul_35), kwargs = {})
#   %add_14 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_12, %where_13), kwargs = {})
#   %mul_40 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, 0.5), kwargs = {})
#   %abs_6 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_40,), kwargs = {})
#   %div_7 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_6, 0.05), kwargs = {})
#   %tanh_5 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_7,), kwargs = {})
#   %mul_41 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_5, 0.1), kwargs = {})
#   %pow_12 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_40, 2), kwargs = {})
#   %mul_42 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_12, 0.01), kwargs = {})
#   %add_15 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %mul_42), kwargs = {})
#   %clamp_min_10 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_40, 0.001), kwargs = {})
#   %reciprocal_5 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_10,), kwargs = {})
#   %mul_43 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, 7.319), kwargs = {})
#   %clamp_max_5 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_43, 1), kwargs = {})
#   %mul_44 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_5, 7.0), kwargs = {})
#   %le_5 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_15, %mul_44), kwargs = {})
#   %where_14 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_5, %mul_40, %where_12), kwargs = {})
#   %where_15 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_5, %where_13, %mul_40), kwargs = {})
#   %add_16 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_14, %where_15), kwargs = {})
#   %mul_45 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, 0.5), kwargs = {})
#   %abs_7 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_45,), kwargs = {})
#   %div_8 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_7, 0.05), kwargs = {})
#   %tanh_6 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_8,), kwargs = {})
#   %mul_46 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_6, 0.1), kwargs = {})
#   %pow_13 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_45, 2), kwargs = {})
#   %mul_47 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_13, 0.01), kwargs = {})
#   %add_17 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_46, %mul_47), kwargs = {})
#   %clamp_min_11 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_45, 0.001), kwargs = {})
#   %reciprocal_6 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_11,), kwargs = {})
#   %mul_48 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_6, 7.319), kwargs = {})
#   %clamp_max_6 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_48, 1), kwargs = {})
#   %mul_49 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_6, 7.0), kwargs = {})
#   %le_6 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_17, %mul_49), kwargs = {})
#   %where_16 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_6, %mul_45, %where_14), kwargs = {})
#   %where_17 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_6, %where_15, %mul_45), kwargs = {})
#   %add_18 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_16, %where_17), kwargs = {})
#   %mul_50 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, 0.5), kwargs = {})
#   %abs_8 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_50,), kwargs = {})
#   %div_9 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_8, 0.05), kwargs = {})
#   %tanh_7 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_9,), kwargs = {})
#   %mul_51 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_7, 0.1), kwargs = {})
#   %pow_14 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_50, 2), kwargs = {})
#   %mul_52 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_14, 0.01), kwargs = {})
#   %add_19 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %mul_52), kwargs = {})
#   %clamp_min_12 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_50, 0.001), kwargs = {})
#   %reciprocal_7 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_12,), kwargs = {})
#   %mul_53 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 7.319), kwargs = {})
#   %clamp_max_7 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_53, 1), kwargs = {})
#   %mul_54 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_7, 7.0), kwargs = {})
#   %le_7 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_19, %mul_54), kwargs = {})
#   %where_18 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_7, %mul_50, %where_16), kwargs = {})
#   %where_19 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_7, %where_17, %mul_50), kwargs = {})
#   %add_20 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_18, %where_19), kwargs = {})
#   %mul_55 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_20, 0.5), kwargs = {})
#   %abs_9 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_55,), kwargs = {})
#   %div_10 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_9, 0.05), kwargs = {})
#   %tanh_8 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_10,), kwargs = {})
#   %mul_56 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_8, 0.1), kwargs = {})
#   %pow_15 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_55, 2), kwargs = {})
#   %mul_57 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_15, 0.01), kwargs = {})
#   %add_21 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %mul_57), kwargs = {})
#   %clamp_min_13 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_55, 0.001), kwargs = {})
#   %reciprocal_8 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_13,), kwargs = {})
#   %mul_58 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_8, 7.319), kwargs = {})
#   %clamp_max_8 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_58, 1), kwargs = {})
#   %mul_59 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_8, 7.0), kwargs = {})
#   %le_8 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_21, %mul_59), kwargs = {})
#   %where_20 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_8, %mul_55, %where_18), kwargs = {})
#   %where_21 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_8, %where_19, %mul_55), kwargs = {})
#   %add_22 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_20, %where_21), kwargs = {})
#   %mul_60 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 0.5), kwargs = {})
#   %abs_10 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_60,), kwargs = {})
#   %div_11 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_10, 0.05), kwargs = {})
#   %tanh_9 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_11,), kwargs = {})
#   %mul_61 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_9, 0.1), kwargs = {})
#   %pow_16 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_60, 2), kwargs = {})
#   %mul_62 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_16, 0.01), kwargs = {})
#   %add_23 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_61, %mul_62), kwargs = {})
#   %clamp_min_14 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_60, 0.001), kwargs = {})
#   %reciprocal_9 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_14,), kwargs = {})
#   %mul_63 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_9, 7.319), kwargs = {})
#   %clamp_max_9 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_63, 1), kwargs = {})
#   %mul_64 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_9, 7.0), kwargs = {})
#   %le_9 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_23, %mul_64), kwargs = {})
#   %where_22 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_9, %mul_60, %where_20), kwargs = {})
#   %where_23 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_9, %where_21, %mul_60), kwargs = {})
#   %add_24 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_22, %where_23), kwargs = {})
#   %mul_65 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, 0.5), kwargs = {})
#   %abs_11 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_65,), kwargs = {})
#   %div_12 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_11, 0.05), kwargs = {})
#   %tanh_10 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_12,), kwargs = {})
#   %mul_66 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_10, 0.1), kwargs = {})
#   %pow_17 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_65, 2), kwargs = {})
#   %mul_67 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_17, 0.01), kwargs = {})
#   %add_25 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_66, %mul_67), kwargs = {})
#   %clamp_min_15 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_65, 0.001), kwargs = {})
#   %reciprocal_10 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_15,), kwargs = {})
#   %mul_68 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_10, 7.319), kwargs = {})
#   %clamp_max_10 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_68, 1), kwargs = {})
#   %mul_69 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_10, 7.0), kwargs = {})
#   %le_10 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_25, %mul_69), kwargs = {})
#   %where_24 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_10, %mul_65, %where_22), kwargs = {})
#   %where_25 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_10, %where_23, %mul_65), kwargs = {})
#   %add_26 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_24, %where_25), kwargs = {})
#   %mul_70 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, 0.5), kwargs = {})
#   %abs_12 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_70,), kwargs = {})
#   %div_13 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_12, 0.05), kwargs = {})
#   %tanh_11 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_13,), kwargs = {})
#   %mul_71 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_11, 0.1), kwargs = {})
#   %pow_18 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_70, 2), kwargs = {})
#   %mul_72 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_18, 0.01), kwargs = {})
#   %add_27 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %mul_72), kwargs = {})
#   %clamp_min_16 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_70, 0.001), kwargs = {})
#   %reciprocal_11 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_16,), kwargs = {})
#   %mul_73 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_11, 7.319), kwargs = {})
#   %clamp_max_11 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_73, 1), kwargs = {})
#   %mul_74 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_11, 7.0), kwargs = {})
#   %le_11 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_27, %mul_74), kwargs = {})
#   %where_27 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_11, %where_25, %mul_70), kwargs = {})
#   return %sqrt_2,%abs_2,%pow_8,%clamp_min_6,%where_6,%where_7,%div_5,%mul_32,%reciprocal_3,%where_10,%where_11,%div_7,%mul_42,%reciprocal_5,%where_14,%where_15,%div_9,%mul_52,%reciprocal_7,%where_18,%where_19,%div_11,%mul_62,%reciprocal_9,%where_22,%where_23,%div_13,%mul_72,%reciprocal_11,%where_27
triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_expand_full_like_ge_gt_le_minimum_mul_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0 = async_compile.triton('triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_expand_full_like_ge_gt_le_minimum_mul_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_expand_full_like_ge_gt_le_minimum_mul_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_expand_full_like_ge_gt_le_minimum_mul_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp15 = tl.load(in_ptr1 + (x0), xmask)
    tmp70 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 0.85
    tmp3 = tmp1 * tmp2
    tmp4 = 0.92
    tmp5 = tmp3 * tmp4
    tmp6 = 9.81
    tmp7 = tmp5 * tmp6
    tmp8 = 0.5192307692307692
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = 0.0025000000000000005
    tmp12 = tmp11 - tmp10
    tmp13 = 0.0
    tmp14 = tmp12 >= tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp16 * tmp16
    tmp18 = 2.5e-05
    tmp19 = tmp17 + tmp18
    tmp20 = 4.0
    tmp21 = tmp19 * tmp20
    tmp22 = tmp21 * tmp12
    tmp23 = 2.500000277905201e-07
    tmp24 = tmp23 - tmp22
    tmp25 = tmp24 >= tmp13
    tmp26 = triton_helpers.maximum(tmp24, tmp13)
    tmp27 = tl.sqrt_rn(tmp26)
    tmp28 = 0.0005
    tmp29 = tmp28 + tmp27
    tmp30 = 1e-12
    tmp31 = tmp29 > tmp30
    tmp32 = tmp25 & tmp31
    tmp33 = -2.0
    tmp34 = tmp12 * tmp33
    tmp35 = triton_helpers.maximum(tmp29, tmp30)
    tmp36 = (tmp34 / tmp35)
    tmp37 = float("inf")
    tmp38 = tl.where(tmp32, tmp36, tmp37)
    tmp39 = tl.where(tmp14, tmp13, tmp38)
    tmp40 = 100.0
    tmp41 = triton_helpers.minimum(tmp40, tmp39)
    tmp42 = 1.0
    tmp43 = tmp3 * tmp42
    tmp44 = tmp43 * tmp6
    tmp45 = 0.4807692307692308
    tmp46 = tmp44 * tmp45
    tmp47 = tmp46 * tmp46
    tmp48 = tmp11 - tmp47
    tmp49 = tmp48 >= tmp13
    tmp50 = tmp15 * tmp45
    tmp51 = tmp50 * tmp50
    tmp52 = tmp51 + tmp18
    tmp53 = tmp52 * tmp20
    tmp54 = tmp53 * tmp48
    tmp55 = tmp23 - tmp54
    tmp56 = tmp55 >= tmp13
    tmp57 = triton_helpers.maximum(tmp55, tmp13)
    tmp58 = tl.sqrt_rn(tmp57)
    tmp59 = tmp28 + tmp58
    tmp60 = tmp59 > tmp30
    tmp61 = tmp56 & tmp60
    tmp62 = tmp48 * tmp33
    tmp63 = triton_helpers.maximum(tmp59, tmp30)
    tmp64 = (tmp62 / tmp63)
    tmp65 = tl.where(tmp61, tmp64, tmp37)
    tmp66 = tl.where(tmp49, tmp13, tmp65)
    tmp67 = triton_helpers.minimum(tmp41, tmp66)
    tmp68 = triton_helpers.maximum(tmp67, tmp13)
    tmp69 = tl.sqrt_rn(tmp68)
    tmp71 = triton_helpers.minimum(tmp70, tmp69)
    tmp72 = 0.5
    tmp73 = tmp71 * tmp72
    tmp74 = tl_math.abs(tmp73)
    tmp75 = 20.0
    tmp76 = tmp74 * tmp75
    tmp77 = libdevice.tanh(tmp76)
    tmp78 = 0.1
    tmp79 = tmp77 * tmp78
    tmp80 = tmp73 * tmp73
    tmp81 = 0.01
    tmp82 = tmp80 * tmp81
    tmp83 = tmp79 + tmp82
    tmp84 = 0.001
    tmp85 = triton_helpers.maximum(tmp73, tmp84)
    tmp86 = tl.full([1], 1, tl.int32)
    tmp87 = (tmp86 / tmp85)
    tmp88 = 7.319
    tmp89 = tmp87 * tmp88
    tmp90 = triton_helpers.minimum(tmp89, tmp42)
    tmp91 = 7.0
    tmp92 = tmp90 * tmp91
    tmp93 = tmp83 <= tmp92
    tmp94 = tl.where(tmp93, tmp73, tmp13)
    tmp95 = tl.where(tmp93, tmp71, tmp73)
    tmp96 = tmp94 + tmp95
    tmp97 = tmp96 * tmp72
    tmp98 = tl_math.abs(tmp97)
    tmp99 = tmp97 * tmp97
    tmp100 = triton_helpers.maximum(tmp97, tmp84)
    tmp101 = tmp98 * tmp75
    tmp102 = libdevice.tanh(tmp101)
    tmp103 = tmp102 * tmp78
    tmp104 = tmp99 * tmp81
    tmp105 = tmp103 + tmp104
    tmp106 = (tmp86 / tmp100)
    tmp107 = tmp106 * tmp88
    tmp108 = triton_helpers.minimum(tmp107, tmp42)
    tmp109 = tmp108 * tmp91
    tmp110 = tmp105 <= tmp109
    tmp111 = tl.where(tmp110, tmp97, tmp94)
    tmp112 = tl.where(tmp110, tmp95, tmp97)
    tmp113 = tmp111 + tmp112
    tmp114 = tmp113 * tmp72
    tmp115 = tl_math.abs(tmp114)
    tmp116 = tmp115 * tmp75
    tmp117 = libdevice.tanh(tmp116)
    tmp118 = tmp117 * tmp78
    tmp119 = tmp114 * tmp114
    tmp120 = tmp119 * tmp81
    tmp121 = tmp118 + tmp120
    tmp122 = triton_helpers.maximum(tmp114, tmp84)
    tmp123 = (tmp86 / tmp122)
    tmp124 = tmp123 * tmp88
    tmp125 = triton_helpers.minimum(tmp124, tmp42)
    tmp126 = tmp125 * tmp91
    tmp127 = tmp121 <= tmp126
    tmp128 = tl.where(tmp127, tmp114, tmp111)
    tmp129 = tl.where(tmp127, tmp112, tmp114)
    tmp130 = tmp128 + tmp129
    tmp131 = tmp130 * tmp72
    tmp132 = tl_math.abs(tmp131)
    tmp133 = tmp132 * tmp75
    tmp134 = tmp131 * tmp131
    tmp135 = tmp134 * tmp81
    tmp136 = triton_helpers.maximum(tmp131, tmp84)
    tmp137 = (tmp86 / tmp136)
    tmp138 = libdevice.tanh(tmp133)
    tmp139 = tmp138 * tmp78
    tmp140 = tmp139 + tmp135
    tmp141 = tmp137 * tmp88
    tmp142 = triton_helpers.minimum(tmp141, tmp42)
    tmp143 = tmp142 * tmp91
    tmp144 = tmp140 <= tmp143
    tmp145 = tl.where(tmp144, tmp131, tmp128)
    tmp146 = tl.where(tmp144, tmp129, tmp131)
    tmp147 = tmp145 + tmp146
    tmp148 = tmp147 * tmp72
    tmp149 = tl_math.abs(tmp148)
    tmp150 = tmp149 * tmp75
    tmp151 = libdevice.tanh(tmp150)
    tmp152 = tmp151 * tmp78
    tmp153 = tmp148 * tmp148
    tmp154 = tmp153 * tmp81
    tmp155 = tmp152 + tmp154
    tmp156 = triton_helpers.maximum(tmp148, tmp84)
    tmp157 = (tmp86 / tmp156)
    tmp158 = tmp157 * tmp88
    tmp159 = triton_helpers.minimum(tmp158, tmp42)
    tmp160 = tmp159 * tmp91
    tmp161 = tmp155 <= tmp160
    tmp162 = tl.where(tmp161, tmp148, tmp145)
    tmp163 = tl.where(tmp161, tmp146, tmp148)
    tmp164 = tmp162 + tmp163
    tmp165 = tmp164 * tmp72
    tmp166 = tl_math.abs(tmp165)
    tmp167 = tmp166 * tmp75
    tmp168 = tmp165 * tmp165
    tmp169 = tmp168 * tmp81
    tmp170 = triton_helpers.maximum(tmp165, tmp84)
    tmp171 = (tmp86 / tmp170)
    tmp172 = libdevice.tanh(tmp167)
    tmp173 = tmp172 * tmp78
    tmp174 = tmp173 + tmp169
    tmp175 = tmp171 * tmp88
    tmp176 = triton_helpers.minimum(tmp175, tmp42)
    tmp177 = tmp176 * tmp91
    tmp178 = tmp174 <= tmp177
    tmp179 = tl.where(tmp178, tmp165, tmp162)
    tmp180 = tl.where(tmp178, tmp163, tmp165)
    tmp181 = tmp179 + tmp180
    tmp182 = tmp181 * tmp72
    tmp183 = tl_math.abs(tmp182)
    tmp184 = tmp183 * tmp75
    tmp185 = libdevice.tanh(tmp184)
    tmp186 = tmp185 * tmp78
    tmp187 = tmp182 * tmp182
    tmp188 = tmp187 * tmp81
    tmp189 = tmp186 + tmp188
    tmp190 = triton_helpers.maximum(tmp182, tmp84)
    tmp191 = (tmp86 / tmp190)
    tmp192 = tmp191 * tmp88
    tmp193 = triton_helpers.minimum(tmp192, tmp42)
    tmp194 = tmp193 * tmp91
    tmp195 = tmp189 <= tmp194
    tmp196 = tl.where(tmp195, tmp182, tmp179)
    tmp197 = tl.where(tmp195, tmp180, tmp182)
    tmp198 = tmp196 + tmp197
    tmp199 = tmp198 * tmp72
    tmp200 = tl_math.abs(tmp199)
    tmp201 = tmp200 * tmp75
    tmp202 = tmp199 * tmp199
    tmp203 = tmp202 * tmp81
    tmp204 = triton_helpers.maximum(tmp199, tmp84)
    tmp205 = (tmp86 / tmp204)
    tmp206 = libdevice.tanh(tmp201)
    tmp207 = tmp206 * tmp78
    tmp208 = tmp207 + tmp203
    tmp209 = tmp205 * tmp88
    tmp210 = triton_helpers.minimum(tmp209, tmp42)
    tmp211 = tmp210 * tmp91
    tmp212 = tmp208 <= tmp211
    tmp213 = tl.where(tmp212, tmp199, tmp196)
    tmp214 = tl.where(tmp212, tmp197, tmp199)
    tmp215 = tmp213 + tmp214
    tmp216 = tmp215 * tmp72
    tmp217 = tl_math.abs(tmp216)
    tmp218 = tmp217 * tmp75
    tmp219 = libdevice.tanh(tmp218)
    tmp220 = tmp219 * tmp78
    tmp221 = tmp216 * tmp216
    tmp222 = tmp221 * tmp81
    tmp223 = tmp220 + tmp222
    tmp224 = triton_helpers.maximum(tmp216, tmp84)
    tmp225 = (tmp86 / tmp224)
    tmp226 = tmp225 * tmp88
    tmp227 = triton_helpers.minimum(tmp226, tmp42)
    tmp228 = tmp227 * tmp91
    tmp229 = tmp223 <= tmp228
    tmp230 = tl.where(tmp229, tmp216, tmp213)
    tmp231 = tl.where(tmp229, tmp214, tmp216)
    tmp232 = tmp230 + tmp231
    tmp233 = tmp232 * tmp72
    tmp234 = tl_math.abs(tmp233)
    tmp235 = tmp234 * tmp75
    tmp236 = tmp233 * tmp233
    tmp237 = tmp236 * tmp81
    tmp238 = triton_helpers.maximum(tmp233, tmp84)
    tmp239 = (tmp86 / tmp238)
    tmp240 = libdevice.tanh(tmp235)
    tmp241 = tmp240 * tmp78
    tmp242 = tmp241 + tmp237
    tmp243 = tmp239 * tmp88
    tmp244 = triton_helpers.minimum(tmp243, tmp42)
    tmp245 = tmp244 * tmp91
    tmp246 = tmp242 <= tmp245
    tmp247 = tl.where(tmp246, tmp233, tmp230)
    tmp248 = tl.where(tmp246, tmp231, tmp233)
    tmp249 = tmp247 + tmp248
    tmp250 = tmp249 * tmp72
    tmp251 = tl_math.abs(tmp250)
    tmp252 = tmp251 * tmp75
    tmp253 = libdevice.tanh(tmp252)
    tmp254 = tmp253 * tmp78
    tmp255 = tmp250 * tmp250
    tmp256 = tmp255 * tmp81
    tmp257 = tmp254 + tmp256
    tmp258 = triton_helpers.maximum(tmp250, tmp84)
    tmp259 = (tmp86 / tmp258)
    tmp260 = tmp259 * tmp88
    tmp261 = triton_helpers.minimum(tmp260, tmp42)
    tmp262 = tmp261 * tmp91
    tmp263 = tmp257 <= tmp262
    tmp264 = tl.where(tmp263, tmp250, tmp247)
    tmp265 = tl.where(tmp263, tmp248, tmp250)
    tmp266 = tmp264 + tmp265
    tmp267 = tmp266 * tmp72
    tmp268 = tl_math.abs(tmp267)
    tmp269 = tmp268 * tmp75
    tmp270 = tmp267 * tmp267
    tmp271 = tmp270 * tmp81
    tmp272 = triton_helpers.maximum(tmp267, tmp84)
    tmp273 = (tmp86 / tmp272)
    tmp274 = libdevice.tanh(tmp269)
    tmp275 = tmp274 * tmp78
    tmp276 = tmp275 + tmp271
    tmp277 = tmp273 * tmp88
    tmp278 = triton_helpers.minimum(tmp277, tmp42)
    tmp279 = tmp278 * tmp91
    tmp280 = tmp276 <= tmp279
    tmp281 = tl.where(tmp280, tmp265, tmp267)
    tl.store(out_ptr0 + (x0), tmp69, xmask)
    tl.store(in_out_ptr0 + (x0), tmp281, xmask)
''', device_str='cuda')


# kernel path: /home/shchon11/F1tenth/F1tenth_E2E/work/adaptive-racing-v3/controller-proof/inductor-cold-v1/d2/cd2e7v4yrff7fylgseoz7uyjhb5nn4rbfjsgtouhtweacc4kxppt.py
# Topologically Sorted Source Nodes: [getitem, mul_65, add_4, cap_1, square_18, mul_63, add_28, mul_64, delta, getitem_1, getitem_2, sub_4, abs_13, clamp_min_17, slew_cap], Original ATen: [aten.unsqueeze, aten.mul, aten.add, aten.minimum, aten.pow, aten.atan, aten.slice, aten.sub, aten.abs, aten.clamp_min, aten.div]
# Source node to ATen node mapping:
#   abs_13 => abs_13
#   add_28 => add_28
#   add_4 => minimum_2
#   cap_1 => minimum_3
#   clamp_min_17 => clamp_min_17
#   delta => atan
#   getitem => unsqueeze
#   getitem_1 => slice_1
#   getitem_2 => slice_2
#   mul_63 => mul_75
#   mul_64 => mul_76
#   mul_65 => mul_77
#   slew_cap => div_14
#   square_18 => pow_19
#   sub_4 => sub_4
# Graph fragment:
#   %arg3_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg0_1 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sqrt_2 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=sqrt_2]
#   %where_27 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_27]
#   %arg2_1 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %unsqueeze : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, 1), kwargs = {})
#   %mul_77 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, 3.2), kwargs = {})
#   %minimum_2 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.minimum.default](args = (%arg0_1, %sqrt_2), kwargs = {})
#   %minimum_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.minimum.default](args = (%minimum_2, %where_27), kwargs = {})
#   %pow_19 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%minimum_3, 2), kwargs = {})
#   %mul_75 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_19, 0.003), kwargs = {})
#   %add_28 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_75, 0.3302), kwargs = {})
#   %mul_76 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, %arg2_1), kwargs = {})
#   %atan : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.atan.default](args = (%mul_76,), kwargs = {})
#   %slice_1 : Tensor "f32[1, 24][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%atan, 1, 1, 9223372036854775807), kwargs = {})
#   %slice_2 : Tensor "f32[1, 24][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%atan, 1, 0, -1), kwargs = {})
#   %sub_4 : Tensor "f32[1, 24][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, %slice_2), kwargs = {})
#   %abs_13 : Tensor "f32[1, 24][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_4,), kwargs = {})
#   %clamp_min_17 : Tensor "f32[1, 24][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%abs_13, 1e-06), kwargs = {})
#   %div_14 : Tensor "f32[1, 24][24, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_77, %clamp_min_17), kwargs = {})
#   return %div_14
triton_poi_fused_abs_add_atan_clamp_min_div_minimum_mul_pow_slice_sub_unsqueeze_1 = async_compile.triton('triton_poi_fused_abs_add_atan_clamp_min_div_minimum_mul_pow_slice_sub_unsqueeze_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_atan_clamp_min_div_minimum_mul_pow_slice_sub_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_atan_clamp_min_div_minimum_mul_pow_slice_sub_unsqueeze_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (1 + x0), xmask)
    tmp5 = tl.load(in_ptr2 + (1 + x0), xmask)
    tmp7 = tl.load(in_ptr3 + (1 + x0), xmask)
    tmp14 = tl.load(in_ptr4 + (1 + x0), xmask)
    tmp17 = tl.load(in_ptr1 + (x0), xmask)
    tmp18 = tl.load(in_ptr2 + (x0), xmask)
    tmp20 = tl.load(in_ptr3 + (x0), xmask)
    tmp25 = tl.load(in_ptr4 + (x0), xmask)
    tmp2 = 3.2
    tmp3 = tmp1 * tmp2
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 * tmp8
    tmp10 = 0.003
    tmp11 = tmp9 * tmp10
    tmp12 = 0.3302
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = libdevice.atan(tmp15)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp21 * tmp21
    tmp23 = tmp22 * tmp10
    tmp24 = tmp23 + tmp12
    tmp26 = tmp24 * tmp25
    tmp27 = libdevice.atan(tmp26)
    tmp28 = tmp16 - tmp27
    tmp29 = tl_math.abs(tmp28)
    tmp30 = 1e-06
    tmp31 = triton_helpers.maximum(tmp29, tmp30)
    tmp32 = (tmp3 / tmp31)
    tl.store(out_ptr0 + (x0), tmp32, xmask)
''', device_str='cuda')


# kernel path: /home/shchon11/F1tenth/F1tenth_E2E/work/adaptive-racing-v3/controller-proof/inductor-cold-v1/f2/cf2zq6fq6junplc3j4ptwwndnm2a7fnn7ppon4cupmwb6gik4u3w.py
# Topologically Sorted Source Nodes: [add_4, cap_1, getitem_3, getitem_4, getitem_5, minimum_4, getitem_6, cat, cap_2, abs_15, clamp_min_18, truediv_28, clamp_12, power, abs_14, truediv_27, tanh_12, mul_66, square_19, mul_67, r, hi, mul_77, square_22, square_23, mul_78, mul_79, square_24, add_30, grip_2, mul_80, mul_81, square_25, cc_2, ge_7, zeros_like_5, mul_72, square_21, mul_73, mul_74, mul_75, sub_8, bb_2, square_27, mul_71, square_20, aa_2, mul_85, mul_86, disc_3, ge_6, clamp_min_21, sqrt_4, den_3, gt_4, and__3, mul_87, clamp_min_22, root_6, full_like_6, root_7, where_31, hi_1, hi_2, mul_94, square_30, square_31, mul_95, mul_96, square_32, add_33, grip_3, mul_97, mul_98, square_33, cc_3, ge_11, zeros_like_7, mul_89, square_29, mul_90, mul_91, mul_92, sub_13, bb_3, square_35, mul_88, square_28, aa_3, mul_102, mul_103, disc_5, ge_10, clamp_min_25, sqrt_6, den_5, gt_7, and__5, mul_104, clamp_min_26, root_10, full_like_8, root_11, where_35, hi_3, abs_16, clamp_min_27, truediv_33, clamp_15, hi_4, square_39, mul_115, mul_116, add_36, grip_4, mul_117, mul_118, square_41, cc_4, ge_15, zeros_like_11, mul_109, square_37, mul_110, mul_111, mul_112, sub_20, bb_4, square_43, mul_108, square_36, aa_4, mul_122, mul_123, disc_7, ge_14, clamp_min_30, sqrt_8, den_7, gt_11, and__7, mul_124, clamp_min_31, root_14, full_like_10, root_15, where_39, hi_5, hi_6, square_47, mul_132, mul_133, add_39, grip_5, mul_134, mul_135, square_49, cc_5, ge_19, zeros_like_13, mul_126, square_45, mul_127, mul_128, mul_129, sub_25, bb_5, square_51, mul_125, square_44, aa_5, mul_139, mul_140, disc_9, ge_18, clamp_min_34, sqrt_10, den_9, gt_14, and__9, mul_141, clamp_min_35, root_18, full_like_12, root_19, where_43, hi_7, minimum_10, acc, lo, ge_5, zeros_like_4, neg, square_26, mul_82, mul_83, disc_2, ge_4, clamp_min_19, sqrt_3, den_2, gt_3, and__2, mul_84, clamp_min_20, root_4, full_like_5, root_5, where_29, neg_1, lo_1, ge_9, zeros_like_6, neg_2, square_34, mul_99, mul_100, disc_4, ge_8, clamp_min_23, sqrt_5, den_4, gt_6, and__4, mul_101, clamp_min_24, root_8, full_like_7, root_9, where_33, neg_3, lo_2, lo_3, lo_4, ge_13, zeros_like_10, neg_4, square_42, mul_119, mul_120, disc_6, ge_12, clamp_min_28, sqrt_7, den_6, gt_10, and__6, mul_121, clamp_min_29, root_12, full_like_9, root_13, where_37, neg_5, lo_5, ge_17, zeros_like_12, neg_6, square_50, mul_136, mul_137, disc_8, ge_16, clamp_min_32, sqrt_9, den_8, gt_13, and__8, mul_138, clamp_min_33, root_16, full_like_11, root_17, where_41, neg_7, lo_6, lo_7, maximum_4, neg_8, brk], Original ATen: [aten.add, aten.minimum, aten.slice, aten.cat, aten.abs, aten.clamp_min, aten.reciprocal, aten.mul, aten.clamp, aten.div, aten.tanh, aten.pow, aten.sub, aten.ge, aten.zeros_like, aten.rsub, aten.sqrt, aten.gt, aten.bitwise_and, aten.full_like, aten.where, aten.neg, aten.maximum]
# Source node to ATen node mapping:
#   aa_2 => sub_7
#   aa_3 => sub_12
#   aa_4 => sub_19
#   aa_5 => sub_24
#   abs_14 => abs_14
#   abs_15 => abs_15
#   abs_16 => abs_16
#   acc => clamp_min_38
#   add_30 => add_30
#   add_33 => add_33
#   add_36 => pow_41
#   add_39 => pow_49
#   add_4 => minimum_2
#   and__2 => bitwise_and_2
#   and__3 => bitwise_and_3
#   and__4 => bitwise_and_4
#   and__5 => bitwise_and_5
#   and__6 => bitwise_and_6
#   and__7 => bitwise_and_7
#   and__8 => bitwise_and_8
#   and__9 => bitwise_and_9
#   bb_2 => mul_89
#   bb_3 => mul_106
#   bb_4 => mul_127
#   bb_5 => mul_144
#   brk => clamp_min_39
#   cap_1 => minimum_3
#   cap_2 => minimum_5
#   cat => cat
#   cc_2 => sub_9
#   cc_3 => sub_14
#   cc_4 => sub_21
#   cc_5 => sub_26
#   clamp_12 => clamp_max_12
#   clamp_15 => clamp_max_14
#   clamp_min_18 => clamp_min_18
#   clamp_min_19 => clamp_min_19
#   clamp_min_20 => clamp_min_20
#   clamp_min_21 => clamp_min_21
#   clamp_min_22 => clamp_min_22
#   clamp_min_23 => clamp_min_23
#   clamp_min_24 => clamp_min_24
#   clamp_min_25 => clamp_min_25
#   clamp_min_26 => clamp_min_26
#   clamp_min_27 => clamp_min_28
#   clamp_min_28 => clamp_min_29
#   clamp_min_29 => clamp_min_30
#   clamp_min_30 => clamp_min_31
#   clamp_min_31 => clamp_min_32
#   clamp_min_32 => clamp_min_33
#   clamp_min_33 => clamp_min_34
#   clamp_min_34 => clamp_min_35
#   clamp_min_35 => clamp_min_36
#   den_2 => add_31
#   den_3 => add_32
#   den_4 => add_34
#   den_5 => add_35
#   den_6 => add_37
#   den_7 => add_38
#   den_8 => add_40
#   den_9 => add_41
#   disc_2 => sub_10
#   disc_3 => sub_11
#   disc_4 => sub_15
#   disc_5 => sub_16
#   disc_6 => sub_22
#   disc_7 => sub_23
#   disc_8 => sub_27
#   disc_9 => sub_28
#   full_like_10 => full_default_24
#   full_like_11 => full_default_28
#   full_like_12 => full_default_30
#   full_like_5 => full_default_10
#   full_like_6 => full_default_12
#   full_like_7 => full_default_14
#   full_like_8 => full_default_16
#   full_like_9 => full_default_22
#   ge_10 => ge_10
#   ge_11 => ge_11
#   ge_12 => ge_12
#   ge_13 => ge_13
#   ge_14 => ge_14
#   ge_15 => ge_15
#   ge_16 => ge_16
#   ge_17 => ge_17
#   ge_18 => ge_18
#   ge_19 => ge_19
#   ge_4 => ge_4
#   ge_5 => ge_5
#   ge_6 => ge_6
#   ge_7 => ge_7
#   ge_8 => ge_8
#   ge_9 => ge_9
#   getitem_3 => slice_3
#   getitem_4 => slice_4
#   getitem_5 => slice_5
#   getitem_6 => slice_6
#   grip_2 => mul_82
#   grip_3 => mul_83
#   grip_4 => mul_120
#   grip_5 => mul_121
#   gt_10 => gt_10
#   gt_11 => gt_11
#   gt_13 => gt_13
#   gt_14 => gt_14
#   gt_3 => gt_3
#   gt_4 => gt_4
#   gt_6 => gt_6
#   gt_7 => gt_7
#   hi => sub_6
#   hi_1 => minimum_6
#   hi_2 => clamp_max_13
#   hi_3 => minimum_7
#   hi_4 => mul_119
#   hi_5 => minimum_8
#   hi_6 => clamp_max_15
#   hi_7 => minimum_9
#   lo => sub_5
#   lo_1 => maximum
#   lo_2 => maximum_1
#   lo_3 => clamp_min_27
#   lo_4 => full_default_19
#   lo_5 => maximum_2
#   lo_6 => maximum_3
#   lo_7 => clamp_min_37
#   maximum_4 => maximum_4
#   minimum_10 => minimum_10
#   minimum_4 => minimum_4
#   mul_100 => mul_113
#   mul_101 => mul_114
#   mul_102 => mul_115
#   mul_103 => mul_116
#   mul_104 => mul_117
#   mul_108 => mul_122
#   mul_109 => full_default_20
#   mul_110 => mul_124
#   mul_111 => mul_125
#   mul_112 => mul_126
#   mul_115 => mul_129
#   mul_116 => mul_130
#   mul_117 => mul_131
#   mul_118 => mul_132
#   mul_119 => mul_133
#   mul_120 => mul_134
#   mul_121 => mul_135
#   mul_122 => mul_136
#   mul_123 => mul_137
#   mul_124 => mul_138
#   mul_125 => mul_139
#   mul_126 => full_default_26
#   mul_127 => mul_141
#   mul_128 => mul_142
#   mul_129 => mul_143
#   mul_132 => mul_146
#   mul_133 => mul_147
#   mul_134 => mul_148
#   mul_135 => mul_149
#   mul_136 => mul_150
#   mul_137 => mul_151
#   mul_138 => mul_152
#   mul_139 => mul_153
#   mul_140 => mul_154
#   mul_141 => mul_155
#   mul_66 => mul_78
#   mul_67 => mul_79
#   mul_71 => mul_84
#   mul_72 => mul_85
#   mul_73 => mul_86
#   mul_74 => mul_87
#   mul_75 => mul_88
#   mul_77 => mul_90
#   mul_78 => mul_91
#   mul_79 => mul_92
#   mul_80 => mul_93
#   mul_81 => mul_94
#   mul_82 => mul_95
#   mul_83 => mul_96
#   mul_84 => mul_97
#   mul_85 => mul_98
#   mul_86 => mul_99
#   mul_87 => mul_100
#   mul_88 => mul_101
#   mul_89 => mul_102
#   mul_90 => mul_103
#   mul_91 => mul_104
#   mul_92 => mul_105
#   mul_94 => mul_107
#   mul_95 => mul_108
#   mul_96 => mul_109
#   mul_97 => mul_110
#   mul_98 => mul_111
#   mul_99 => mul_112
#   neg => neg
#   neg_1 => neg_1
#   neg_2 => neg_2
#   neg_3 => neg_3
#   neg_4 => neg_4
#   neg_5 => neg_5
#   neg_6 => neg_6
#   neg_7 => neg_7
#   neg_8 => neg_8
#   power => mul_81
#   r => add_29
#   root_10 => div_19
#   root_11 => where_34
#   root_12 => div_20
#   root_13 => where_36
#   root_14 => div_21
#   root_15 => where_38
#   root_16 => div_22
#   root_17 => where_40
#   root_18 => div_23
#   root_19 => where_42
#   root_4 => div_16
#   root_5 => where_28
#   root_6 => div_17
#   root_7 => where_30
#   root_8 => div_18
#   root_9 => where_32
#   sqrt_10 => sqrt_10
#   sqrt_3 => sqrt_3
#   sqrt_4 => sqrt_4
#   sqrt_5 => sqrt_5
#   sqrt_6 => sqrt_6
#   sqrt_7 => sqrt_7
#   sqrt_8 => sqrt_8
#   sqrt_9 => sqrt_9
#   square_19 => pow_20
#   square_20 => pow_21
#   square_21 => pow_22
#   square_22 => pow_23
#   square_23 => pow_24
#   square_24 => pow_25
#   square_25 => pow_26
#   square_26 => pow_27
#   square_27 => pow_28
#   square_28 => pow_29
#   square_29 => pow_30
#   square_30 => pow_31
#   square_31 => pow_32
#   square_32 => pow_33
#   square_33 => pow_34
#   square_34 => pow_35
#   square_35 => pow_36
#   square_36 => pow_37
#   square_37 => pow_38
#   square_39 => pow_40
#   square_41 => pow_42
#   square_42 => pow_43
#   square_43 => pow_44
#   square_44 => pow_45
#   square_45 => pow_46
#   square_47 => pow_48
#   square_49 => pow_50
#   square_50 => pow_51
#   square_51 => pow_52
#   sub_13 => sub_13
#   sub_20 => sub_20
#   sub_25 => sub_25
#   sub_8 => sub_8
#   tanh_12 => tanh_12
#   truediv_27 => div_15
#   truediv_28 => mul_80, reciprocal_12
#   truediv_33 => mul_118, reciprocal_13
#   where_29 => where_29
#   where_31 => where_31
#   where_33 => where_33
#   where_35 => where_35
#   where_37 => where_37
#   where_39 => where_39
#   where_41 => where_41
#   where_43 => where_43
#   zeros_like_10 => full_default_23
#   zeros_like_11 => full_default_25
#   zeros_like_12 => full_default_29
#   zeros_like_13 => full_default_31
#   zeros_like_4 => full_default_11
#   zeros_like_5 => full_default_13
#   zeros_like_6 => full_default_15
#   zeros_like_7 => full_default_17
# Graph fragment:
#   %arg0_1 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sqrt_2 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=sqrt_2]
#   %where_27 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_27]
#   %div_14 : Tensor "f32[1, 24][24, 1]cuda:0" = PlaceHolder[target=div_14]
#   %arg1_1 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %minimum_5 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=minimum_5]
#   %arg2_1 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %mul_99 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=mul_99]
#   %clamp_min_21 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_21]
#   %clamp_min_22 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_22]
#   %ge_6 : Tensor "b8[1, 25][25, 1]cuda:0" = PlaceHolder[target=ge_6]
#   %gt_4 : Tensor "b8[1, 25][25, 1]cuda:0" = PlaceHolder[target=gt_4]
#   %div_17 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=div_17]
#   %mul_116 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=mul_116]
#   %clamp_min_25 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_25]
#   %clamp_min_26 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_26]
#   %ge_10 : Tensor "b8[1, 25][25, 1]cuda:0" = PlaceHolder[target=ge_10]
#   %gt_7 : Tensor "b8[1, 25][25, 1]cuda:0" = PlaceHolder[target=gt_7]
#   %div_19 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=div_19]
#   %sub_23 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=sub_23]
#   %div_21 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=div_21]
#   %sub_28 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=sub_28]
#   %div_23 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=div_23]
#   %mul_96 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=mul_96]
#   %clamp_min_19 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_19]
#   %clamp_min_20 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_20]
#   %ge_4 : Tensor "b8[1, 25][25, 1]cuda:0" = PlaceHolder[target=ge_4]
#   %gt_3 : Tensor "b8[1, 25][25, 1]cuda:0" = PlaceHolder[target=gt_3]
#   %div_16 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=div_16]
#   %mul_113 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=mul_113]
#   %clamp_min_23 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_23]
#   %clamp_min_24 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_24]
#   %ge_8 : Tensor "b8[1, 25][25, 1]cuda:0" = PlaceHolder[target=ge_8]
#   %gt_6 : Tensor "b8[1, 25][25, 1]cuda:0" = PlaceHolder[target=gt_6]
#   %div_18 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=div_18]
#   %sub_22 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=sub_22]
#   %div_20 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=div_20]
#   %sub_27 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=sub_27]
#   %div_22 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=div_22]
#   %where_31 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_31]
#   %where_35 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_35]
#   %where_39 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_39]
#   %where_43 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_43]
#   %where_29 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_29]
#   %where_33 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_33]
#   %where_37 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_37]
#   %where_41 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=where_41]
#   %minimum_10 : Tensor "f32[1, 25][25, 1]cuda:0" = PlaceHolder[target=minimum_10]
#   %minimum_2 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.minimum.default](args = (%arg0_1, %sqrt_2), kwargs = {})
#   %minimum_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.minimum.default](args = (%minimum_2, %where_27), kwargs = {})
#   %slice_3 : Tensor "f32[1, 1][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%div_14, 1, 0, 1), kwargs = {})
#   %slice_4 : Tensor "f32[1, 23][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%div_14, 1, 0, -1), kwargs = {})
#   %slice_5 : Tensor "f32[1, 23][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%div_14, 1, 1, 9223372036854775807), kwargs = {})
#   %minimum_4 : Tensor "f32[1, 23][23, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%slice_4, %slice_5), kwargs = {})
#   %slice_6 : Tensor "f32[1, 1][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%div_14, 1, -1, 9223372036854775807), kwargs = {})
#   %cat : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_3, %minimum_4, %slice_6], 1), kwargs = {})
#   %minimum_5 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=9] = call_function[target=torch.ops.aten.minimum.default](args = (%minimum_3, %cat), kwargs = {})
#   %abs_15 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%minimum_5,), kwargs = {})
#   %clamp_min_18 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%abs_15, 0.001), kwargs = {})
#   %reciprocal_12 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_18,), kwargs = {})
#   %mul_80 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_12, 7.319), kwargs = {})
#   %clamp_max_12 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_80, 1), kwargs = {})
#   %mul_81 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_12, 7.0), kwargs = {})
#   %abs_14 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%minimum_5,), kwargs = {})
#   %div_15 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_14, 0.05), kwargs = {})
#   %tanh_12 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_15,), kwargs = {})
#   %mul_78 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_12, 0.1), kwargs = {})
#   %pow_20 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%minimum_5, 2), kwargs = {})
#   %mul_79 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_20, 0.01), kwargs = {})
#   %add_29 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_78, %mul_79), kwargs = {})
#   %sub_6 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_81, %add_29), kwargs = {})
#   %mul_90 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, 0.5), kwargs = {})
#   %pow_23 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_90, 2), kwargs = {})
#   %pow_24 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%minimum_5, 2), kwargs = {})
#   %mul_91 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_24, 0.5192307692307692), kwargs = {})
#   %mul_92 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %arg2_1), kwargs = {})
#   %pow_25 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_92, 2), kwargs = {})
#   %add_30 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_23, %pow_25), kwargs = {})
#   %mul_82 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 0.92), kwargs = {})
#   %mul_93 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, 9.81), kwargs = {})
#   %mul_94 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_93, 0.5192307692307692), kwargs = {})
#   %pow_26 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_94, 2), kwargs = {})
#   %sub_9 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_30, %pow_26), kwargs = {})
#   %ge_7 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_9, 0), kwargs = {})
#   %full_default_13 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_85 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, 0.25), kwargs = {})
#   %pow_22 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_82, 2), kwargs = {})
#   %mul_86 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_22, 9.81), kwargs = {})
#   %mul_87 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_86, 0.5192307692307692), kwargs = {})
#   %mul_88 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_87, -0.22410660205935795), kwargs = {})
#   %sub_8 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_85, %mul_88), kwargs = {})
#   %mul_89 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, 2), kwargs = {})
#   %pow_28 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_89, 2), kwargs = {})
#   %mul_84 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, -0.22410660205935795), kwargs = {})
#   %pow_21 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_84, 2), kwargs = {})
#   %sub_7 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_21), kwargs = {})
#   %mul_98 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, 4), kwargs = {})
#   %mul_99 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_98, %sub_9), kwargs = {})
#   %sub_11 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_28, %mul_99), kwargs = {})
#   %ge_6 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_11, 0), kwargs = {})
#   %clamp_min_21 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_11, 0), kwargs = {})
#   %sqrt_4 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_21,), kwargs = {})
#   %add_32 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %sqrt_4), kwargs = {})
#   %gt_4 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_32, 1e-12), kwargs = {})
#   %bitwise_and_3 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_6, %gt_4), kwargs = {})
#   %mul_100 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, -2), kwargs = {})
#   %clamp_min_22 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_32, 1e-12), kwargs = {})
#   %div_17 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_100, %clamp_min_22), kwargs = {})
#   %full_default_12 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_30 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_3, %div_17, %full_default_12), kwargs = {})
#   %where_31 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_7, %full_default_13, %where_30), kwargs = {})
#   %minimum_6 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%sub_6, %where_31), kwargs = {})
#   %clamp_max_13 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%minimum_6, 22.72870945945946), kwargs = {})
#   %mul_107 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, 0.5), kwargs = {})
#   %pow_31 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_107, 2), kwargs = {})
#   %pow_32 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%minimum_5, 2), kwargs = {})
#   %mul_108 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_32, 0.4807692307692308), kwargs = {})
#   %mul_109 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_108, %arg2_1), kwargs = {})
#   %pow_33 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_109, 2), kwargs = {})
#   %add_33 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_31, %pow_33), kwargs = {})
#   %mul_83 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 1.0), kwargs = {})
#   %mul_110 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_83, 9.81), kwargs = {})
#   %mul_111 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_110, 0.4807692307692308), kwargs = {})
#   %pow_34 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_111, 2), kwargs = {})
#   %sub_14 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_33, %pow_34), kwargs = {})
#   %ge_11 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_14, 0), kwargs = {})
#   %full_default_17 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_102 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, 0.25), kwargs = {})
#   %pow_30 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_83, 2), kwargs = {})
#   %mul_103 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_30, 9.81), kwargs = {})
#   %mul_104 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_103, 0.4807692307692308), kwargs = {})
#   %mul_105 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_104, 0.22410660205935795), kwargs = {})
#   %sub_13 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_102, %mul_105), kwargs = {})
#   %mul_106 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, 2), kwargs = {})
#   %pow_36 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_106, 2), kwargs = {})
#   %mul_101 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_83, 0.22410660205935795), kwargs = {})
#   %pow_29 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_101, 2), kwargs = {})
#   %sub_12 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_29), kwargs = {})
#   %mul_115 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, 4), kwargs = {})
#   %mul_116 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_115, %sub_14), kwargs = {})
#   %sub_16 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_36, %mul_116), kwargs = {})
#   %ge_10 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_16, 0), kwargs = {})
#   %clamp_min_25 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_16, 0), kwargs = {})
#   %sqrt_6 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_25,), kwargs = {})
#   %add_35 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_106, %sqrt_6), kwargs = {})
#   %gt_7 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_35, 1e-12), kwargs = {})
#   %bitwise_and_5 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_10, %gt_7), kwargs = {})
#   %mul_117 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, -2), kwargs = {})
#   %clamp_min_26 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_35, 1e-12), kwargs = {})
#   %div_19 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_117, %clamp_min_26), kwargs = {})
#   %full_default_16 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_34 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_5, %div_19, %full_default_16), kwargs = {})
#   %where_35 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_11, %full_default_17, %where_34), kwargs = {})
#   %minimum_7 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%clamp_max_13, %where_35), kwargs = {})
#   %abs_16 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%minimum_5,), kwargs = {})
#   %clamp_min_28 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%abs_16, 0.001), kwargs = {})
#   %reciprocal_13 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_28,), kwargs = {})
#   %mul_118 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_13, 7.319), kwargs = {})
#   %clamp_max_14 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_118, 1), kwargs = {})
#   %mul_119 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_14, 7.0), kwargs = {})
#   %pow_40 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%minimum_5, 2), kwargs = {})
#   %mul_129 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_40, 0.5192307692307692), kwargs = {})
#   %mul_130 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_129, %arg2_1), kwargs = {})
#   %pow_41 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_130, 2), kwargs = {})
#   %mul_120 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 0.92), kwargs = {})
#   %mul_131 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_120, 9.81), kwargs = {})
#   %mul_132 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_131, 0.5192307692307692), kwargs = {})
#   %pow_42 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_132, 2), kwargs = {})
#   %sub_21 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_41, %pow_42), kwargs = {})
#   %ge_15 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_21, 0), kwargs = {})
#   %full_default_25 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_20 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_38 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_120, 2), kwargs = {})
#   %mul_124 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_38, 9.81), kwargs = {})
#   %mul_125 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_124, 0.5192307692307692), kwargs = {})
#   %mul_126 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_125, -0.22410660205935795), kwargs = {})
#   %sub_20 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_20, %mul_126), kwargs = {})
#   %mul_127 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, 2), kwargs = {})
#   %pow_44 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_127, 2), kwargs = {})
#   %mul_122 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_120, -0.22410660205935795), kwargs = {})
#   %pow_37 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_122, 2), kwargs = {})
#   %sub_19 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_37), kwargs = {})
#   %mul_136 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, 4), kwargs = {})
#   %mul_137 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_136, %sub_21), kwargs = {})
#   %sub_23 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_44, %mul_137), kwargs = {})
#   %ge_14 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_23, 0), kwargs = {})
#   %clamp_min_31 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_23, 0), kwargs = {})
#   %sqrt_8 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_31,), kwargs = {})
#   %add_38 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_127, %sqrt_8), kwargs = {})
#   %gt_11 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_38, 1e-12), kwargs = {})
#   %bitwise_and_7 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_14, %gt_11), kwargs = {})
#   %mul_138 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, -2), kwargs = {})
#   %clamp_min_32 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_38, 1e-12), kwargs = {})
#   %div_21 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_138, %clamp_min_32), kwargs = {})
#   %full_default_24 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_38 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_7, %div_21, %full_default_24), kwargs = {})
#   %where_39 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_15, %full_default_25, %where_38), kwargs = {})
#   %minimum_8 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%mul_119, %where_39), kwargs = {})
#   %clamp_max_15 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%minimum_8, 22.72870945945946), kwargs = {})
#   %pow_48 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%minimum_5, 2), kwargs = {})
#   %mul_146 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_48, 0.4807692307692308), kwargs = {})
#   %mul_147 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_146, %arg2_1), kwargs = {})
#   %pow_49 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_147, 2), kwargs = {})
#   %mul_121 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 1.0), kwargs = {})
#   %mul_148 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_121, 9.81), kwargs = {})
#   %mul_149 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, 0.4807692307692308), kwargs = {})
#   %pow_50 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_149, 2), kwargs = {})
#   %sub_26 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_49, %pow_50), kwargs = {})
#   %ge_19 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_26, 0), kwargs = {})
#   %full_default_31 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_26 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_46 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_121, 2), kwargs = {})
#   %mul_141 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_46, 9.81), kwargs = {})
#   %mul_142 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_141, 0.4807692307692308), kwargs = {})
#   %mul_143 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_142, 0.22410660205935795), kwargs = {})
#   %sub_25 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_26, %mul_143), kwargs = {})
#   %mul_144 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, 2), kwargs = {})
#   %pow_52 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_144, 2), kwargs = {})
#   %mul_139 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_121, 0.22410660205935795), kwargs = {})
#   %pow_45 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_139, 2), kwargs = {})
#   %sub_24 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_45), kwargs = {})
#   %mul_153 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, 4), kwargs = {})
#   %mul_154 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_153, %sub_26), kwargs = {})
#   %sub_28 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_52, %mul_154), kwargs = {})
#   %ge_18 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_28, 0), kwargs = {})
#   %clamp_min_35 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_28, 0), kwargs = {})
#   %sqrt_10 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_35,), kwargs = {})
#   %add_41 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_144, %sqrt_10), kwargs = {})
#   %gt_14 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_41, 1e-12), kwargs = {})
#   %bitwise_and_9 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_18, %gt_14), kwargs = {})
#   %mul_155 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, -2), kwargs = {})
#   %clamp_min_36 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_41, 1e-12), kwargs = {})
#   %div_23 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_155, %clamp_min_36), kwargs = {})
#   %full_default_30 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_42 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_9, %div_23, %full_default_30), kwargs = {})
#   %where_43 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_19, %full_default_31, %where_42), kwargs = {})
#   %minimum_9 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%clamp_max_15, %where_43), kwargs = {})
#   %minimum_10 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%minimum_7, %minimum_9), kwargs = {})
#   %clamp_min_38 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%minimum_10, 0), kwargs = {})
#   %sub_5 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (-5.0, %add_29), kwargs = {})
#   %ge_5 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_9, 0), kwargs = {})
#   %full_default_11 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_89,), kwargs = {})
#   %pow_27 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg, 2), kwargs = {})
#   %mul_95 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, 4), kwargs = {})
#   %mul_96 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_95, %sub_9), kwargs = {})
#   %sub_10 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_27, %mul_96), kwargs = {})
#   %ge_4 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_10, 0), kwargs = {})
#   %clamp_min_19 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_10, 0), kwargs = {})
#   %sqrt_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_19,), kwargs = {})
#   %add_31 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg, %sqrt_3), kwargs = {})
#   %gt_3 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_31, 1e-12), kwargs = {})
#   %bitwise_and_2 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_4, %gt_3), kwargs = {})
#   %mul_97 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, -2), kwargs = {})
#   %clamp_min_20 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_31, 1e-12), kwargs = {})
#   %div_16 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_97, %clamp_min_20), kwargs = {})
#   %full_default_10 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_28 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_2, %div_16, %full_default_10), kwargs = {})
#   %where_29 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_5, %full_default_11, %where_28), kwargs = {})
#   %neg_1 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_29,), kwargs = {})
#   %maximum : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%sub_5, %neg_1), kwargs = {})
#   %ge_9 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_14, 0), kwargs = {})
#   %full_default_15 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg_2 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_106,), kwargs = {})
#   %pow_35 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg_2, 2), kwargs = {})
#   %mul_112 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, 4), kwargs = {})
#   %mul_113 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %sub_14), kwargs = {})
#   %sub_15 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_35, %mul_113), kwargs = {})
#   %ge_8 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_15, 0), kwargs = {})
#   %clamp_min_23 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_15, 0), kwargs = {})
#   %sqrt_5 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_23,), kwargs = {})
#   %add_34 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_2, %sqrt_5), kwargs = {})
#   %gt_6 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_34, 1e-12), kwargs = {})
#   %bitwise_and_4 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_8, %gt_6), kwargs = {})
#   %mul_114 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, -2), kwargs = {})
#   %clamp_min_24 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_34, 1e-12), kwargs = {})
#   %div_18 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_114, %clamp_min_24), kwargs = {})
#   %full_default_14 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_32 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_4, %div_18, %full_default_14), kwargs = {})
#   %where_33 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_9, %full_default_15, %where_32), kwargs = {})
#   %neg_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_33,), kwargs = {})
#   %maximum_1 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%maximum, %neg_3), kwargs = {})
#   %clamp_min_27 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%maximum_1, -21.045101351351356), kwargs = {})
#   %full_default_19 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], -5.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ge_13 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_21, 0), kwargs = {})
#   %full_default_23 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg_4 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_127,), kwargs = {})
#   %pow_43 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg_4, 2), kwargs = {})
#   %mul_133 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, 4), kwargs = {})
#   %mul_134 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_133, %sub_21), kwargs = {})
#   %sub_22 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_43, %mul_134), kwargs = {})
#   %ge_12 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_22, 0), kwargs = {})
#   %clamp_min_29 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_22, 0), kwargs = {})
#   %sqrt_7 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_29,), kwargs = {})
#   %add_37 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_4, %sqrt_7), kwargs = {})
#   %gt_10 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_37, 1e-12), kwargs = {})
#   %bitwise_and_6 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_12, %gt_10), kwargs = {})
#   %mul_135 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, -2), kwargs = {})
#   %clamp_min_30 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_37, 1e-12), kwargs = {})
#   %div_20 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_135, %clamp_min_30), kwargs = {})
#   %full_default_22 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_36 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_6, %div_20, %full_default_22), kwargs = {})
#   %where_37 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_13, %full_default_23, %where_36), kwargs = {})
#   %neg_5 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_37,), kwargs = {})
#   %maximum_2 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%full_default_19, %neg_5), kwargs = {})
#   %ge_17 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_26, 0), kwargs = {})
#   %full_default_29 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg_6 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_144,), kwargs = {})
#   %pow_51 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg_6, 2), kwargs = {})
#   %mul_150 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, 4), kwargs = {})
#   %mul_151 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_150, %sub_26), kwargs = {})
#   %sub_27 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_51, %mul_151), kwargs = {})
#   %ge_16 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_27, 0), kwargs = {})
#   %clamp_min_33 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_27, 0), kwargs = {})
#   %sqrt_9 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_33,), kwargs = {})
#   %add_40 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_6, %sqrt_9), kwargs = {})
#   %gt_13 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_40, 1e-12), kwargs = {})
#   %bitwise_and_8 : Tensor "b8[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_16, %gt_13), kwargs = {})
#   %mul_152 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, -2), kwargs = {})
#   %clamp_min_34 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_40, 1e-12), kwargs = {})
#   %div_22 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_152, %clamp_min_34), kwargs = {})
#   %full_default_28 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_40 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_8, %div_22, %full_default_28), kwargs = {})
#   %where_41 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_17, %full_default_29, %where_40), kwargs = {})
#   %neg_7 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_41,), kwargs = {})
#   %maximum_3 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%maximum_2, %neg_7), kwargs = {})
#   %clamp_min_37 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%maximum_3, -21.045101351351356), kwargs = {})
#   %maximum_4 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%clamp_min_27, %clamp_min_37), kwargs = {})
#   %neg_8 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%maximum_4,), kwargs = {})
#   %clamp_min_39 : Tensor "f32[1, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%neg_8, 0), kwargs = {})
#   return %minimum_5,%mul_99,%ge_6,%clamp_min_21,%gt_4,%clamp_min_22,%div_17,%where_31,%mul_116,%ge_10,%clamp_min_25,%gt_7,%clamp_min_26,%div_19,%where_35,%sub_23,%div_21,%where_39,%sub_28,%div_23,%where_43,%mul_96,%ge_4,%clamp_min_19,%gt_3,%clamp_min_20,%div_16,%where_29,%mul_113,%ge_8,%clamp_min_23,%gt_6,%clamp_min_24,%div_18,%where_33,%sub_22,%div_20,%where_37,%sub_27,%div_22,%where_41,%minimum_10,%clamp_min_39,%clamp_min_38
triton_poi_fused_abs_add_bitwise_and_cat_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_slice_sqrt_sub_tanh_where_zeros_like_2 = async_compile.triton('triton_poi_fused_abs_add_bitwise_and_cat_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_slice_sqrt_sub_tanh_where_zeros_like_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr5': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_bitwise_and_cat_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_slice_sqrt_sub_tanh_where_zeros_like_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr5'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 3, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 892}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_bitwise_and_cat_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_slice_sqrt_sub_tanh_where_zeros_like_2(in_out_ptr0, in_out_ptr1, in_out_ptr5, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp31 = tl.load(in_ptr3 + (0))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp57 = tl.load(in_ptr4 + (x0), xmask)
    tmp2 = triton_helpers.minimum(tmp0, tmp1)
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tl.load(in_ptr2 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp12 = tl.where(tmp9, tmp11, 0.0)
    tmp13 = tmp5 >= tmp8
    tmp14 = tl.full([1], 24, tl.int64)
    tmp15 = tmp5 < tmp14
    tmp16 = tmp13 & tmp15
    tmp17 = tl.load(in_ptr2 + ((-1) + x0), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr2 + (1 + ((-1) + x0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tmp5 >= tmp14
    tmp23 = tl.full([1], 25, tl.int64)
    tmp24 = tmp5 < tmp23
    tmp25 = tl.load(in_ptr2 + (23))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp27 = tl.where(tmp22, tmp26, 0.0)
    tmp28 = tl.where(tmp16, tmp21, tmp27)
    tmp29 = tl.where(tmp9, tmp12, tmp28)
    tmp30 = triton_helpers.minimum(tmp4, tmp29)
    tmp33 = 0.92
    tmp34 = tmp32 * tmp33
    tmp35 = -0.22410660205935795
    tmp36 = tmp34 * tmp35
    tmp37 = tmp36 * tmp36
    tmp38 = 0.25
    tmp39 = tmp38 - tmp37
    tmp40 = 4.0
    tmp41 = tmp39 * tmp40
    tmp42 = tl_math.abs(tmp30)
    tmp43 = 20.0
    tmp44 = tmp42 * tmp43
    tmp45 = libdevice.tanh(tmp44)
    tmp46 = 0.1
    tmp47 = tmp45 * tmp46
    tmp48 = tmp30 * tmp30
    tmp49 = 0.01
    tmp50 = tmp48 * tmp49
    tmp51 = tmp47 + tmp50
    tmp52 = 0.5
    tmp53 = tmp51 * tmp52
    tmp54 = tmp53 * tmp53
    tmp55 = 0.5192307692307692
    tmp56 = tmp48 * tmp55
    tmp58 = tmp56 * tmp57
    tmp59 = tmp58 * tmp58
    tmp60 = tmp54 + tmp59
    tmp61 = 9.81
    tmp62 = tmp34 * tmp61
    tmp63 = tmp62 * tmp55
    tmp64 = tmp63 * tmp63
    tmp65 = tmp60 - tmp64
    tmp66 = tmp41 * tmp65
    tmp67 = tmp51 * tmp38
    tmp68 = tmp34 * tmp34
    tmp69 = tmp68 * tmp61
    tmp70 = tmp69 * tmp55
    tmp71 = tmp70 * tmp35
    tmp72 = tmp67 - tmp71
    tmp73 = 2.0
    tmp74 = tmp72 * tmp73
    tmp75 = tmp74 * tmp74
    tmp76 = tmp75 - tmp66
    tmp77 = 0.0
    tmp78 = tmp76 >= tmp77
    tmp79 = triton_helpers.maximum(tmp76, tmp77)
    tmp80 = tl.sqrt_rn(tmp79)
    tmp81 = tmp74 + tmp80
    tmp82 = 1e-12
    tmp83 = tmp81 > tmp82
    tmp84 = triton_helpers.maximum(tmp81, tmp82)
    tmp85 = -2.0
    tmp86 = tmp65 * tmp85
    tmp87 = (tmp86 / tmp84)
    tmp88 = tmp65 >= tmp77
    tmp89 = tmp78 & tmp83
    tmp90 = float("inf")
    tmp91 = tl.where(tmp89, tmp87, tmp90)
    tmp92 = tl.where(tmp88, tmp77, tmp91)
    tmp93 = 1.0
    tmp94 = tmp32 * tmp93
    tmp95 = 0.22410660205935795
    tmp96 = tmp94 * tmp95
    tmp97 = tmp96 * tmp96
    tmp98 = tmp38 - tmp97
    tmp99 = tmp98 * tmp40
    tmp100 = 0.4807692307692308
    tmp101 = tmp48 * tmp100
    tmp102 = tmp101 * tmp57
    tmp103 = tmp102 * tmp102
    tmp104 = tmp54 + tmp103
    tmp105 = tmp94 * tmp61
    tmp106 = tmp105 * tmp100
    tmp107 = tmp106 * tmp106
    tmp108 = tmp104 - tmp107
    tmp109 = tmp99 * tmp108
    tmp110 = tmp94 * tmp94
    tmp111 = tmp110 * tmp61
    tmp112 = tmp111 * tmp100
    tmp113 = tmp112 * tmp95
    tmp114 = tmp67 - tmp113
    tmp115 = tmp114 * tmp73
    tmp116 = tmp115 * tmp115
    tmp117 = tmp116 - tmp109
    tmp118 = tmp117 >= tmp77
    tmp119 = triton_helpers.maximum(tmp117, tmp77)
    tmp120 = tl.sqrt_rn(tmp119)
    tmp121 = tmp115 + tmp120
    tmp122 = tmp121 > tmp82
    tmp123 = triton_helpers.maximum(tmp121, tmp82)
    tmp124 = tmp108 * tmp85
    tmp125 = (tmp124 / tmp123)
    tmp126 = tmp108 >= tmp77
    tmp127 = tmp118 & tmp122
    tmp128 = tl.where(tmp127, tmp125, tmp90)
    tmp129 = tl.where(tmp126, tmp77, tmp128)
    tmp130 = tmp77 - tmp71
    tmp131 = tmp130 * tmp73
    tmp132 = tmp131 * tmp131
    tmp133 = tmp59 - tmp64
    tmp134 = tmp41 * tmp133
    tmp135 = tmp132 - tmp134
    tmp136 = tmp133 * tmp85
    tmp137 = triton_helpers.maximum(tmp135, tmp77)
    tmp138 = tl.sqrt_rn(tmp137)
    tmp139 = tmp131 + tmp138
    tmp140 = triton_helpers.maximum(tmp139, tmp82)
    tmp141 = (tmp136 / tmp140)
    tmp142 = tmp133 >= tmp77
    tmp143 = tmp135 >= tmp77
    tmp144 = tmp139 > tmp82
    tmp145 = tmp143 & tmp144
    tmp146 = tl.where(tmp145, tmp141, tmp90)
    tmp147 = tl.where(tmp142, tmp77, tmp146)
    tmp148 = tmp77 - tmp113
    tmp149 = tmp148 * tmp73
    tmp150 = tmp149 * tmp149
    tmp151 = tmp103 - tmp107
    tmp152 = tmp99 * tmp151
    tmp153 = tmp150 - tmp152
    tmp154 = tmp151 * tmp85
    tmp155 = triton_helpers.maximum(tmp153, tmp77)
    tmp156 = tl.sqrt_rn(tmp155)
    tmp157 = tmp149 + tmp156
    tmp158 = triton_helpers.maximum(tmp157, tmp82)
    tmp159 = (tmp154 / tmp158)
    tmp160 = tmp151 >= tmp77
    tmp161 = tmp153 >= tmp77
    tmp162 = tmp157 > tmp82
    tmp163 = tmp161 & tmp162
    tmp164 = tl.where(tmp163, tmp159, tmp90)
    tmp165 = tl.where(tmp160, tmp77, tmp164)
    tmp166 = -tmp74
    tmp167 = tmp166 * tmp166
    tmp168 = tmp167 - tmp66
    tmp169 = tmp168 >= tmp77
    tmp170 = triton_helpers.maximum(tmp168, tmp77)
    tmp171 = tl.sqrt_rn(tmp170)
    tmp172 = tmp166 + tmp171
    tmp173 = tmp172 > tmp82
    tmp174 = triton_helpers.maximum(tmp172, tmp82)
    tmp175 = (tmp86 / tmp174)
    tmp176 = tmp169 & tmp173
    tmp177 = tl.where(tmp176, tmp175, tmp90)
    tmp178 = tl.where(tmp88, tmp77, tmp177)
    tmp179 = -tmp115
    tmp180 = tmp179 * tmp179
    tmp181 = tmp180 - tmp109
    tmp182 = tmp181 >= tmp77
    tmp183 = triton_helpers.maximum(tmp181, tmp77)
    tmp184 = tl.sqrt_rn(tmp183)
    tmp185 = tmp179 + tmp184
    tmp186 = tmp185 > tmp82
    tmp187 = triton_helpers.maximum(tmp185, tmp82)
    tmp188 = (tmp124 / tmp187)
    tmp189 = tmp182 & tmp186
    tmp190 = tl.where(tmp189, tmp188, tmp90)
    tmp191 = tl.where(tmp126, tmp77, tmp190)
    tmp192 = -tmp131
    tmp193 = tmp192 * tmp192
    tmp194 = tmp193 - tmp134
    tmp195 = triton_helpers.maximum(tmp194, tmp77)
    tmp196 = tl.sqrt_rn(tmp195)
    tmp197 = tmp192 + tmp196
    tmp198 = triton_helpers.maximum(tmp197, tmp82)
    tmp199 = (tmp136 / tmp198)
    tmp200 = tmp194 >= tmp77
    tmp201 = tmp197 > tmp82
    tmp202 = tmp200 & tmp201
    tmp203 = tl.where(tmp202, tmp199, tmp90)
    tmp204 = tl.where(tmp142, tmp77, tmp203)
    tmp205 = -tmp149
    tmp206 = tmp205 * tmp205
    tmp207 = tmp206 - tmp152
    tmp208 = triton_helpers.maximum(tmp207, tmp77)
    tmp209 = tl.sqrt_rn(tmp208)
    tmp210 = tmp205 + tmp209
    tmp211 = triton_helpers.maximum(tmp210, tmp82)
    tmp212 = (tmp154 / tmp211)
    tmp213 = tmp207 >= tmp77
    tmp214 = tmp210 > tmp82
    tmp215 = tmp213 & tmp214
    tmp216 = tl.where(tmp215, tmp212, tmp90)
    tmp217 = tl.where(tmp160, tmp77, tmp216)
    tmp218 = 0.001
    tmp219 = triton_helpers.maximum(tmp42, tmp218)
    tmp220 = tl.full([1], 1, tl.int32)
    tmp221 = (tmp220 / tmp219)
    tmp222 = 7.319
    tmp223 = tmp221 * tmp222
    tmp224 = triton_helpers.minimum(tmp223, tmp93)
    tmp225 = 7.0
    tmp226 = tmp224 * tmp225
    tmp227 = tmp226 - tmp51
    tmp228 = triton_helpers.minimum(tmp227, tmp92)
    tmp229 = 22.72870945945946
    tmp230 = triton_helpers.minimum(tmp228, tmp229)
    tmp231 = triton_helpers.minimum(tmp230, tmp129)
    tmp232 = triton_helpers.minimum(tmp226, tmp147)
    tmp233 = triton_helpers.minimum(tmp232, tmp229)
    tmp234 = triton_helpers.minimum(tmp233, tmp165)
    tmp235 = triton_helpers.minimum(tmp231, tmp234)
    tmp236 = -5.0
    tmp237 = tmp236 - tmp51
    tmp238 = -tmp178
    tmp239 = triton_helpers.maximum(tmp237, tmp238)
    tmp240 = -tmp191
    tmp241 = triton_helpers.maximum(tmp239, tmp240)
    tmp242 = -21.045101351351356
    tmp243 = triton_helpers.maximum(tmp241, tmp242)
    tmp244 = -tmp204
    tmp245 = triton_helpers.maximum(tmp236, tmp244)
    tmp246 = -tmp217
    tmp247 = triton_helpers.maximum(tmp245, tmp246)
    tmp248 = triton_helpers.maximum(tmp247, tmp242)
    tmp249 = triton_helpers.maximum(tmp243, tmp248)
    tmp250 = -tmp249
    tmp251 = triton_helpers.maximum(tmp250, tmp77)
    tmp252 = triton_helpers.maximum(tmp235, tmp77)
    tl.store(in_out_ptr0 + (x0), tmp30, xmask)
    tl.store(in_out_ptr5 + (x0), tmp251, xmask)
    tl.store(in_out_ptr1 + (x0), tmp252, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 25), (25, 1))
        assert_size_stride(arg1_1, (1, 1), (1, 1))
        assert_size_stride(arg2_1, (1, 25), (25, 1))
        assert_size_stride(arg3_1, (1, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 25), (25, 1), torch.float32)
            buf26 = empty_strided_cuda((1, 25), (25, 1), torch.float32)
            buf29 = buf26; del buf26  # reuse
            # Topologically Sorted Source Nodes: [curve, mul, grip, mul_4, mul_5, square_1, cc, expand_as, ge_1, zeros_like, square_2, mul_3, square, aa, mul_6, mul_7, disc, ge, bb, clamp_min, sqrt, den, gt, and_, mul_8, clamp_min_1, root, full_like_2, root_1, where_1, curve_1, grip_1, mul_10, mul_11, square_4, cc_1, expand_as_1, ge_3, zeros_like_1, square_5, mul_9, square_3, aa_1, mul_12, mul_13, disc_1, ge_2, bb_1, clamp_min_2, sqrt_1, den_1, gt_1, and__1, mul_14, clamp_min_3, root_2, full_like_4, root_3, where_3, curve_2, clamp_min_4, curve_3, add_4, middle, abs_1, truediv_2, tanh, mul_16, square_6, mul_17, add_5, clamp_min_5, truediv_3, clamp, mul_18, sustainable, low, low_1, high_1, add_6, middle_1, abs_2, truediv_4, tanh_1, mul_20, square_7, mul_21, add_7, clamp_min_6, truediv_5, clamp_1, mul_22, sustainable_1, low_2, high_2, add_8, middle_2, abs_3, truediv_6, tanh_2, mul_24, square_8, mul_25, add_9, clamp_min_7, truediv_7, clamp_2, mul_26, sustainable_2, low_3, high_3, add_10, middle_3, abs_4, truediv_8, tanh_3, mul_28, square_9, mul_29, add_11, clamp_min_8, truediv_9, clamp_3, mul_30, sustainable_3, low_4, high_4, add_12, middle_4, abs_5, truediv_10, tanh_4, mul_32, square_10, mul_33, add_13, clamp_min_9, truediv_11, clamp_4, mul_34, sustainable_4, low_5, high_5, add_14, middle_5, abs_6, truediv_12, tanh_5, mul_36, square_11, mul_37, add_15, clamp_min_10, truediv_13, clamp_5, mul_38, sustainable_5, low_6, high_6, add_16, middle_6, abs_7, truediv_14, tanh_6, mul_40, square_12, mul_41, add_17, clamp_min_11, truediv_15, clamp_6, mul_42, sustainable_6, low_7, high_7, add_18, middle_7, abs_8, truediv_16, tanh_7, mul_44, square_13, mul_45, add_19, clamp_min_12, truediv_17, clamp_7, mul_46, sustainable_7, low_8, high_8, add_20, middle_8, abs_9, truediv_18, tanh_8, mul_48, square_14, mul_49, add_21, clamp_min_13, truediv_19, clamp_8, mul_50, sustainable_8, low_9, high_9, add_22, middle_9, abs_10, truediv_20, tanh_9, mul_52, square_15, mul_53, add_23, clamp_min_14, truediv_21, clamp_9, mul_54, sustainable_9, low_10, high_10, add_24, middle_10, abs_11, truediv_22, tanh_10, mul_56, square_16, mul_57, add_25, clamp_min_15, truediv_23, clamp_10, mul_58, sustainable_10, low_11, high_11, add_26, middle_11, abs_12, truediv_24, tanh_11, mul_60, square_17, mul_61, add_27, clamp_min_16, truediv_25, clamp_11, mul_62, sustainable_11, high_12], Original ATen: [aten.full_like, aten.mul, aten.pow, aten.rsub, aten.expand, aten.ge, aten.zeros_like, aten.add, aten.sub, aten.clamp_min, aten.sqrt, aten.gt, aten.bitwise_and, aten.div, aten.where, aten.minimum, aten.abs, aten.tanh, aten.reciprocal, aten.clamp, aten.le]
            stream0 = get_raw_stream(0)
            triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_expand_full_like_ge_gt_le_minimum_mul_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0.run(buf29, arg1_1, arg2_1, arg0_1, buf0, 25, stream=stream0)
            buf30 = empty_strided_cuda((1, 24), (24, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem, mul_65, add_4, cap_1, square_18, mul_63, add_28, mul_64, delta, getitem_1, getitem_2, sub_4, abs_13, clamp_min_17, slew_cap], Original ATen: [aten.unsqueeze, aten.mul, aten.add, aten.minimum, aten.pow, aten.atan, aten.slice, aten.sub, aten.abs, aten.clamp_min, aten.div]
            stream0 = get_raw_stream(0)
            triton_poi_fused_abs_add_atan_clamp_min_div_minimum_mul_pow_slice_sub_unsqueeze_1.run(arg3_1, arg0_1, buf0, buf29, arg2_1, buf30, 24, stream=stream0)
            del arg3_1
            buf31 = buf29; del buf29  # reuse
            buf36 = empty_strided_cuda((1, 25), (25, 1), torch.float32)
            buf37 = buf36; del buf36  # reuse
            buf38 = buf37; del buf37  # reuse
            buf58 = empty_strided_cuda((1, 25), (25, 1), torch.float32)
            buf59 = buf58; del buf58  # reuse
            buf60 = buf59; del buf59  # reuse
            buf52 = buf38; del buf38  # reuse
            buf74 = buf60; del buf60  # reuse
            buf53 = buf52; del buf52  # reuse
            # Topologically Sorted Source Nodes: [add_4, cap_1, getitem_3, getitem_4, getitem_5, minimum_4, getitem_6, cat, cap_2, abs_15, clamp_min_18, truediv_28, clamp_12, power, abs_14, truediv_27, tanh_12, mul_66, square_19, mul_67, r, hi, mul_77, square_22, square_23, mul_78, mul_79, square_24, add_30, grip_2, mul_80, mul_81, square_25, cc_2, ge_7, zeros_like_5, mul_72, square_21, mul_73, mul_74, mul_75, sub_8, bb_2, square_27, mul_71, square_20, aa_2, mul_85, mul_86, disc_3, ge_6, clamp_min_21, sqrt_4, den_3, gt_4, and__3, mul_87, clamp_min_22, root_6, full_like_6, root_7, where_31, hi_1, hi_2, mul_94, square_30, square_31, mul_95, mul_96, square_32, add_33, grip_3, mul_97, mul_98, square_33, cc_3, ge_11, zeros_like_7, mul_89, square_29, mul_90, mul_91, mul_92, sub_13, bb_3, square_35, mul_88, square_28, aa_3, mul_102, mul_103, disc_5, ge_10, clamp_min_25, sqrt_6, den_5, gt_7, and__5, mul_104, clamp_min_26, root_10, full_like_8, root_11, where_35, hi_3, abs_16, clamp_min_27, truediv_33, clamp_15, hi_4, square_39, mul_115, mul_116, add_36, grip_4, mul_117, mul_118, square_41, cc_4, ge_15, zeros_like_11, mul_109, square_37, mul_110, mul_111, mul_112, sub_20, bb_4, square_43, mul_108, square_36, aa_4, mul_122, mul_123, disc_7, ge_14, clamp_min_30, sqrt_8, den_7, gt_11, and__7, mul_124, clamp_min_31, root_14, full_like_10, root_15, where_39, hi_5, hi_6, square_47, mul_132, mul_133, add_39, grip_5, mul_134, mul_135, square_49, cc_5, ge_19, zeros_like_13, mul_126, square_45, mul_127, mul_128, mul_129, sub_25, bb_5, square_51, mul_125, square_44, aa_5, mul_139, mul_140, disc_9, ge_18, clamp_min_34, sqrt_10, den_9, gt_14, and__9, mul_141, clamp_min_35, root_18, full_like_12, root_19, where_43, hi_7, minimum_10, acc, lo, ge_5, zeros_like_4, neg, square_26, mul_82, mul_83, disc_2, ge_4, clamp_min_19, sqrt_3, den_2, gt_3, and__2, mul_84, clamp_min_20, root_4, full_like_5, root_5, where_29, neg_1, lo_1, ge_9, zeros_like_6, neg_2, square_34, mul_99, mul_100, disc_4, ge_8, clamp_min_23, sqrt_5, den_4, gt_6, and__4, mul_101, clamp_min_24, root_8, full_like_7, root_9, where_33, neg_3, lo_2, lo_3, lo_4, ge_13, zeros_like_10, neg_4, square_42, mul_119, mul_120, disc_6, ge_12, clamp_min_28, sqrt_7, den_6, gt_10, and__6, mul_121, clamp_min_29, root_12, full_like_9, root_13, where_37, neg_5, lo_5, ge_17, zeros_like_12, neg_6, square_50, mul_136, mul_137, disc_8, ge_16, clamp_min_32, sqrt_9, den_8, gt_13, and__8, mul_138, clamp_min_33, root_16, full_like_11, root_17, where_41, neg_7, lo_6, lo_7, maximum_4, neg_8, brk], Original ATen: [aten.add, aten.minimum, aten.slice, aten.cat, aten.abs, aten.clamp_min, aten.reciprocal, aten.mul, aten.clamp, aten.div, aten.tanh, aten.pow, aten.sub, aten.ge, aten.zeros_like, aten.rsub, aten.sqrt, aten.gt, aten.bitwise_and, aten.full_like, aten.where, aten.neg, aten.maximum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_abs_add_bitwise_and_cat_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_slice_sqrt_sub_tanh_where_zeros_like_2.run(buf31, buf53, buf74, arg0_1, buf0, buf30, arg1_1, arg2_1, 25, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            del buf30
        return (buf31, buf0, buf53, buf74, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 25), (25, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 25), (25, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
