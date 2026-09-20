# AOT ID: ['4_inference']
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


# kernel path: /home/shchon11/F1tenth/F1tenth_E2E/work/adaptive-racing-v3/controller-proof/inductor-cold-v1/5e/c5etyf6a45p34aplbspo3df7tnqtzuxfemr2vwcmecc2mbchgv4j.py
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
#   %arg2_1 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg1_1 : Tensor "f32[48, 1][1, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %bitwise_and : Tensor "b8[48, 25][25, 1]cuda:0" = PlaceHolder[target=bitwise_and]
#   %div : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=div]
#   %bitwise_and_1 : Tensor "b8[48, 25][25, 1]cuda:0" = PlaceHolder[target=bitwise_and_1]
#   %div_1 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=div_1]
#   %minimum_1 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=minimum_1]
#   %arg0_1 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sqrt_2 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=sqrt_2]
#   %abs_2 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=abs_2]
#   %pow_8 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=pow_8]
#   %clamp_min_6 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_6]
#   %where_6 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_6]
#   %where_7 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_7]
#   %div_5 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=div_5]
#   %mul_32 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=mul_32]
#   %reciprocal_3 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=reciprocal_3]
#   %where_10 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_10]
#   %where_11 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_11]
#   %div_7 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=div_7]
#   %mul_42 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=mul_42]
#   %reciprocal_5 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=reciprocal_5]
#   %where_14 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_14]
#   %where_15 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_15]
#   %div_9 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=div_9]
#   %mul_52 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=mul_52]
#   %reciprocal_7 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=reciprocal_7]
#   %where_18 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_18]
#   %where_19 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_19]
#   %div_11 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=div_11]
#   %mul_62 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=mul_62]
#   %reciprocal_9 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=reciprocal_9]
#   %where_22 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_22]
#   %where_23 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_23]
#   %div_13 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=div_13]
#   %mul_72 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=mul_72]
#   %reciprocal_11 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=reciprocal_11]
#   %full_default : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 100.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 0.85), kwargs = {})
#   %mul_1 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 0.92), kwargs = {})
#   %mul_4 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, 9.81), kwargs = {})
#   %mul_5 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, 0.5192307692307692), kwargs = {})
#   %pow_2 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_5, 2), kwargs = {})
#   %sub : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.0025000000000000005, %pow_2), kwargs = {})
#   %expand : Tensor "f32[48, 25][1, 0]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.expand.default](args = (%sub, [48, 25]), kwargs = {})
#   %ge_1 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%expand, 0), kwargs = {})
#   %full_default_4 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 2.500000277905201e-07), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, 0.5192307692307692), kwargs = {})
#   %pow_1 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_3, 2), kwargs = {})
#   %add : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, 2.5e-05), kwargs = {})
#   %mul_6 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 4), kwargs = {})
#   %mul_7 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %expand), kwargs = {})
#   %sub_1 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_2, %mul_7), kwargs = {})
#   %ge : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_1, 0), kwargs = {})
#   %full_default_1 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 0.0005), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clamp_min : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_1, 0), kwargs = {})
#   %sqrt : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min,), kwargs = {})
#   %add_1 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_1, %sqrt), kwargs = {})
#   %gt : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 1e-12), kwargs = {})
#   %bitwise_and : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge, %gt), kwargs = {})
#   %mul_8 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, -2), kwargs = {})
#   %clamp_min_1 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_1, 1e-12), kwargs = {})
#   %div : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_8, %clamp_min_1), kwargs = {})
#   %full_default_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and, %div, %full_default_3), kwargs = {})
#   %where_1 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_1, %full_default_4, %where), kwargs = {})
#   %minimum : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %where_1), kwargs = {})
#   %mul_2 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 1.0), kwargs = {})
#   %mul_10 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, 9.81), kwargs = {})
#   %mul_11 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, 0.4807692307692308), kwargs = {})
#   %pow_5 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_11, 2), kwargs = {})
#   %sub_2 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.0025000000000000005, %pow_5), kwargs = {})
#   %expand_1 : Tensor "f32[48, 25][1, 0]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.expand.default](args = (%sub_2, [48, 25]), kwargs = {})
#   %ge_3 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%expand_1, 0), kwargs = {})
#   %full_default_8 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_6 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 2.500000277905201e-07), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_9 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, 0.4807692307692308), kwargs = {})
#   %pow_4 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_9, 2), kwargs = {})
#   %add_2 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_4, 2.5e-05), kwargs = {})
#   %mul_12 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 4), kwargs = {})
#   %mul_13 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %expand_1), kwargs = {})
#   %sub_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_6, %mul_13), kwargs = {})
#   %ge_2 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_3, 0), kwargs = {})
#   %full_default_5 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 0.0005), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clamp_min_2 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_3, 0), kwargs = {})
#   %sqrt_1 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_2,), kwargs = {})
#   %add_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_5, %sqrt_1), kwargs = {})
#   %gt_1 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_3, 1e-12), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_2, %gt_1), kwargs = {})
#   %mul_14 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_1, -2), kwargs = {})
#   %clamp_min_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_3, 1e-12), kwargs = {})
#   %div_1 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_14, %clamp_min_3), kwargs = {})
#   %full_default_7 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_1, %div_1, %full_default_7), kwargs = {})
#   %where_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_3, %full_default_8, %where_2), kwargs = {})
#   %minimum_1 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%minimum, %where_3), kwargs = {})
#   %clamp_min_4 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%minimum_1, 0), kwargs = {})
#   %sqrt_2 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_4,), kwargs = {})
#   %minimum_2 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.minimum.default](args = (%arg0_1, %sqrt_2), kwargs = {})
#   %mul_15 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%minimum_2, 0.5), kwargs = {})
#   %abs_1 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_15,), kwargs = {})
#   %div_2 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_1, 0.05), kwargs = {})
#   %tanh : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_2,), kwargs = {})
#   %mul_16 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 0.1), kwargs = {})
#   %pow_7 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_15, 2), kwargs = {})
#   %mul_17 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_7, 0.01), kwargs = {})
#   %add_5 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %mul_17), kwargs = {})
#   %clamp_min_5 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_15, 0.001), kwargs = {})
#   %reciprocal : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_5,), kwargs = {})
#   %mul_18 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 7.319), kwargs = {})
#   %clamp_max : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_18, 1), kwargs = {})
#   %mul_19 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max, 7.0), kwargs = {})
#   %le : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_5, %mul_19), kwargs = {})
#   %full_default_9 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %mul_15, %full_default_9), kwargs = {})
#   %where_5 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %minimum_2, %mul_15), kwargs = {})
#   %add_6 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_4, %where_5), kwargs = {})
#   %mul_20 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 0.5), kwargs = {})
#   %abs_2 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_20,), kwargs = {})
#   %div_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_2, 0.05), kwargs = {})
#   %tanh_1 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_3,), kwargs = {})
#   %mul_21 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_1, 0.1), kwargs = {})
#   %pow_8 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_20, 2), kwargs = {})
#   %mul_22 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_8, 0.01), kwargs = {})
#   %add_7 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %mul_22), kwargs = {})
#   %clamp_min_6 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_20, 0.001), kwargs = {})
#   %reciprocal_1 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_6,), kwargs = {})
#   %mul_23 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 7.319), kwargs = {})
#   %clamp_max_1 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_23, 1), kwargs = {})
#   %mul_24 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_1, 7.0), kwargs = {})
#   %le_1 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_7, %mul_24), kwargs = {})
#   %where_6 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %mul_20, %where_4), kwargs = {})
#   %where_7 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %where_5, %mul_20), kwargs = {})
#   %add_8 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_6, %where_7), kwargs = {})
#   %mul_25 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 0.5), kwargs = {})
#   %abs_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_25,), kwargs = {})
#   %div_4 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_3, 0.05), kwargs = {})
#   %tanh_2 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_4,), kwargs = {})
#   %mul_26 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_2, 0.1), kwargs = {})
#   %pow_9 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_25, 2), kwargs = {})
#   %mul_27 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_9, 0.01), kwargs = {})
#   %add_9 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %mul_27), kwargs = {})
#   %clamp_min_7 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_25, 0.001), kwargs = {})
#   %reciprocal_2 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_7,), kwargs = {})
#   %mul_28 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 7.319), kwargs = {})
#   %clamp_max_2 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_28, 1), kwargs = {})
#   %mul_29 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_2, 7.0), kwargs = {})
#   %le_2 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_9, %mul_29), kwargs = {})
#   %where_8 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_2, %mul_25, %where_6), kwargs = {})
#   %where_9 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_2, %where_7, %mul_25), kwargs = {})
#   %add_10 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_8, %where_9), kwargs = {})
#   %mul_30 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 0.5), kwargs = {})
#   %abs_4 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_30,), kwargs = {})
#   %div_5 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_4, 0.05), kwargs = {})
#   %tanh_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_5,), kwargs = {})
#   %mul_31 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_3, 0.1), kwargs = {})
#   %pow_10 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_30, 2), kwargs = {})
#   %mul_32 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_10, 0.01), kwargs = {})
#   %add_11 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %mul_32), kwargs = {})
#   %clamp_min_8 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_30, 0.001), kwargs = {})
#   %reciprocal_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_8,), kwargs = {})
#   %mul_33 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 7.319), kwargs = {})
#   %clamp_max_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_33, 1), kwargs = {})
#   %mul_34 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_3, 7.0), kwargs = {})
#   %le_3 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_11, %mul_34), kwargs = {})
#   %where_10 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %mul_30, %where_8), kwargs = {})
#   %where_11 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %where_9, %mul_30), kwargs = {})
#   %add_12 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_10, %where_11), kwargs = {})
#   %mul_35 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, 0.5), kwargs = {})
#   %abs_5 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_35,), kwargs = {})
#   %div_6 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_5, 0.05), kwargs = {})
#   %tanh_4 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_6,), kwargs = {})
#   %mul_36 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_4, 0.1), kwargs = {})
#   %pow_11 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_35, 2), kwargs = {})
#   %mul_37 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_11, 0.01), kwargs = {})
#   %add_13 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_36, %mul_37), kwargs = {})
#   %clamp_min_9 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_35, 0.001), kwargs = {})
#   %reciprocal_4 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_9,), kwargs = {})
#   %mul_38 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, 7.319), kwargs = {})
#   %clamp_max_4 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_38, 1), kwargs = {})
#   %mul_39 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_4, 7.0), kwargs = {})
#   %le_4 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_13, %mul_39), kwargs = {})
#   %where_12 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_4, %mul_35, %where_10), kwargs = {})
#   %where_13 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_4, %where_11, %mul_35), kwargs = {})
#   %add_14 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_12, %where_13), kwargs = {})
#   %mul_40 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, 0.5), kwargs = {})
#   %abs_6 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_40,), kwargs = {})
#   %div_7 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_6, 0.05), kwargs = {})
#   %tanh_5 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_7,), kwargs = {})
#   %mul_41 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_5, 0.1), kwargs = {})
#   %pow_12 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_40, 2), kwargs = {})
#   %mul_42 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_12, 0.01), kwargs = {})
#   %add_15 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %mul_42), kwargs = {})
#   %clamp_min_10 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_40, 0.001), kwargs = {})
#   %reciprocal_5 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_10,), kwargs = {})
#   %mul_43 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, 7.319), kwargs = {})
#   %clamp_max_5 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_43, 1), kwargs = {})
#   %mul_44 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_5, 7.0), kwargs = {})
#   %le_5 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_15, %mul_44), kwargs = {})
#   %where_14 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_5, %mul_40, %where_12), kwargs = {})
#   %where_15 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_5, %where_13, %mul_40), kwargs = {})
#   %add_16 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_14, %where_15), kwargs = {})
#   %mul_45 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, 0.5), kwargs = {})
#   %abs_7 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_45,), kwargs = {})
#   %div_8 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_7, 0.05), kwargs = {})
#   %tanh_6 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_8,), kwargs = {})
#   %mul_46 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_6, 0.1), kwargs = {})
#   %pow_13 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_45, 2), kwargs = {})
#   %mul_47 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_13, 0.01), kwargs = {})
#   %add_17 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_46, %mul_47), kwargs = {})
#   %clamp_min_11 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_45, 0.001), kwargs = {})
#   %reciprocal_6 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_11,), kwargs = {})
#   %mul_48 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_6, 7.319), kwargs = {})
#   %clamp_max_6 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_48, 1), kwargs = {})
#   %mul_49 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_6, 7.0), kwargs = {})
#   %le_6 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_17, %mul_49), kwargs = {})
#   %where_16 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_6, %mul_45, %where_14), kwargs = {})
#   %where_17 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_6, %where_15, %mul_45), kwargs = {})
#   %add_18 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_16, %where_17), kwargs = {})
#   %mul_50 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, 0.5), kwargs = {})
#   %abs_8 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_50,), kwargs = {})
#   %div_9 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_8, 0.05), kwargs = {})
#   %tanh_7 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_9,), kwargs = {})
#   %mul_51 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_7, 0.1), kwargs = {})
#   %pow_14 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_50, 2), kwargs = {})
#   %mul_52 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_14, 0.01), kwargs = {})
#   %add_19 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %mul_52), kwargs = {})
#   %clamp_min_12 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_50, 0.001), kwargs = {})
#   %reciprocal_7 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_12,), kwargs = {})
#   %mul_53 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 7.319), kwargs = {})
#   %clamp_max_7 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_53, 1), kwargs = {})
#   %mul_54 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_7, 7.0), kwargs = {})
#   %le_7 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_19, %mul_54), kwargs = {})
#   %where_18 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_7, %mul_50, %where_16), kwargs = {})
#   %where_19 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_7, %where_17, %mul_50), kwargs = {})
#   %add_20 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_18, %where_19), kwargs = {})
#   %mul_55 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_20, 0.5), kwargs = {})
#   %abs_9 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_55,), kwargs = {})
#   %div_10 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_9, 0.05), kwargs = {})
#   %tanh_8 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_10,), kwargs = {})
#   %mul_56 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_8, 0.1), kwargs = {})
#   %pow_15 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_55, 2), kwargs = {})
#   %mul_57 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_15, 0.01), kwargs = {})
#   %add_21 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %mul_57), kwargs = {})
#   %clamp_min_13 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_55, 0.001), kwargs = {})
#   %reciprocal_8 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_13,), kwargs = {})
#   %mul_58 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_8, 7.319), kwargs = {})
#   %clamp_max_8 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_58, 1), kwargs = {})
#   %mul_59 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_8, 7.0), kwargs = {})
#   %le_8 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_21, %mul_59), kwargs = {})
#   %where_20 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_8, %mul_55, %where_18), kwargs = {})
#   %where_21 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_8, %where_19, %mul_55), kwargs = {})
#   %add_22 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_20, %where_21), kwargs = {})
#   %mul_60 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 0.5), kwargs = {})
#   %abs_10 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_60,), kwargs = {})
#   %div_11 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_10, 0.05), kwargs = {})
#   %tanh_9 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_11,), kwargs = {})
#   %mul_61 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_9, 0.1), kwargs = {})
#   %pow_16 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_60, 2), kwargs = {})
#   %mul_62 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_16, 0.01), kwargs = {})
#   %add_23 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_61, %mul_62), kwargs = {})
#   %clamp_min_14 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_60, 0.001), kwargs = {})
#   %reciprocal_9 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_14,), kwargs = {})
#   %mul_63 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_9, 7.319), kwargs = {})
#   %clamp_max_9 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_63, 1), kwargs = {})
#   %mul_64 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_9, 7.0), kwargs = {})
#   %le_9 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_23, %mul_64), kwargs = {})
#   %where_22 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_9, %mul_60, %where_20), kwargs = {})
#   %where_23 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_9, %where_21, %mul_60), kwargs = {})
#   %add_24 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_22, %where_23), kwargs = {})
#   %mul_65 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, 0.5), kwargs = {})
#   %abs_11 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_65,), kwargs = {})
#   %div_12 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_11, 0.05), kwargs = {})
#   %tanh_10 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_12,), kwargs = {})
#   %mul_66 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_10, 0.1), kwargs = {})
#   %pow_17 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_65, 2), kwargs = {})
#   %mul_67 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_17, 0.01), kwargs = {})
#   %add_25 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_66, %mul_67), kwargs = {})
#   %clamp_min_15 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_65, 0.001), kwargs = {})
#   %reciprocal_10 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_15,), kwargs = {})
#   %mul_68 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_10, 7.319), kwargs = {})
#   %clamp_max_10 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_68, 1), kwargs = {})
#   %mul_69 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_10, 7.0), kwargs = {})
#   %le_10 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_25, %mul_69), kwargs = {})
#   %where_24 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_10, %mul_65, %where_22), kwargs = {})
#   %where_25 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_10, %where_23, %mul_65), kwargs = {})
#   %add_26 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_24, %where_25), kwargs = {})
#   %mul_70 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, 0.5), kwargs = {})
#   %abs_12 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_70,), kwargs = {})
#   %div_13 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_12, 0.05), kwargs = {})
#   %tanh_11 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_13,), kwargs = {})
#   %mul_71 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_11, 0.1), kwargs = {})
#   %pow_18 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_70, 2), kwargs = {})
#   %mul_72 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_18, 0.01), kwargs = {})
#   %add_27 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %mul_72), kwargs = {})
#   %clamp_min_16 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_70, 0.001), kwargs = {})
#   %reciprocal_11 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_16,), kwargs = {})
#   %mul_73 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_11, 7.319), kwargs = {})
#   %clamp_max_11 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_73, 1), kwargs = {})
#   %mul_74 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_11, 7.0), kwargs = {})
#   %le_11 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_27, %mul_74), kwargs = {})
#   %where_27 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_11, %where_25, %mul_70), kwargs = {})
#   return %bitwise_and,%div,%bitwise_and_1,%div_1,%minimum_1,%sqrt_2,%abs_2,%pow_8,%clamp_min_6,%where_6,%where_7,%div_5,%mul_32,%reciprocal_3,%where_10,%where_11,%div_7,%mul_42,%reciprocal_5,%where_14,%where_15,%div_9,%mul_52,%reciprocal_7,%where_18,%where_19,%div_11,%mul_62,%reciprocal_9,%where_22,%where_23,%div_13,%mul_72,%reciprocal_11,%where_27
triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_expand_full_like_ge_gt_le_minimum_mul_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0 = async_compile.triton('triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_expand_full_like_ge_gt_le_minimum_mul_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_expand_full_like_ge_gt_le_minimum_mul_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 28800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_expand_full_like_ge_gt_le_minimum_mul_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 25
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp8 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr2 + (x2), xmask)
    tmp1 = 0.5192307692307692
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = 2.5e-05
    tmp5 = tmp3 + tmp4
    tmp6 = 4.0
    tmp7 = tmp5 * tmp6
    tmp9 = 0.85
    tmp10 = tmp8 * tmp9
    tmp11 = 0.92
    tmp12 = tmp10 * tmp11
    tmp13 = 9.81
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14 * tmp1
    tmp16 = tmp15 * tmp15
    tmp17 = 0.0025000000000000005
    tmp18 = tmp17 - tmp16
    tmp19 = tmp7 * tmp18
    tmp20 = 2.500000277905201e-07
    tmp21 = tmp20 - tmp19
    tmp22 = 0.0
    tmp23 = tmp21 >= tmp22
    tmp24 = triton_helpers.maximum(tmp21, tmp22)
    tmp25 = tl.sqrt_rn(tmp24)
    tmp26 = 0.0005
    tmp27 = tmp26 + tmp25
    tmp28 = 1e-12
    tmp29 = tmp27 > tmp28
    tmp30 = tmp23 & tmp29
    tmp31 = -2.0
    tmp32 = tmp18 * tmp31
    tmp33 = triton_helpers.maximum(tmp27, tmp28)
    tmp34 = (tmp32 / tmp33)
    tmp35 = 0.4807692307692308
    tmp36 = tmp0 * tmp35
    tmp37 = tmp36 * tmp36
    tmp38 = tmp37 + tmp4
    tmp39 = tmp38 * tmp6
    tmp40 = 1.0
    tmp41 = tmp10 * tmp40
    tmp42 = tmp41 * tmp13
    tmp43 = tmp42 * tmp35
    tmp44 = tmp43 * tmp43
    tmp45 = tmp17 - tmp44
    tmp46 = tmp39 * tmp45
    tmp47 = tmp20 - tmp46
    tmp48 = tmp47 >= tmp22
    tmp49 = triton_helpers.maximum(tmp47, tmp22)
    tmp50 = tl.sqrt_rn(tmp49)
    tmp51 = tmp26 + tmp50
    tmp52 = tmp51 > tmp28
    tmp53 = tmp48 & tmp52
    tmp54 = tmp45 * tmp31
    tmp55 = triton_helpers.maximum(tmp51, tmp28)
    tmp56 = (tmp54 / tmp55)
    tmp57 = tmp18 >= tmp22
    tmp58 = float("inf")
    tmp59 = tl.where(tmp30, tmp34, tmp58)
    tmp60 = tl.where(tmp57, tmp22, tmp59)
    tmp61 = 100.0
    tmp62 = triton_helpers.minimum(tmp61, tmp60)
    tmp63 = tmp45 >= tmp22
    tmp64 = tl.where(tmp53, tmp56, tmp58)
    tmp65 = tl.where(tmp63, tmp22, tmp64)
    tmp66 = triton_helpers.minimum(tmp62, tmp65)
    tmp67 = triton_helpers.maximum(tmp66, tmp22)
    tmp68 = tl.sqrt_rn(tmp67)
    tmp70 = triton_helpers.minimum(tmp69, tmp68)
    tmp71 = 0.5
    tmp72 = tmp70 * tmp71
    tmp73 = tl_math.abs(tmp72)
    tmp74 = 20.0
    tmp75 = tmp73 * tmp74
    tmp76 = libdevice.tanh(tmp75)
    tmp77 = 0.1
    tmp78 = tmp76 * tmp77
    tmp79 = tmp72 * tmp72
    tmp80 = 0.01
    tmp81 = tmp79 * tmp80
    tmp82 = tmp78 + tmp81
    tmp83 = 0.001
    tmp84 = triton_helpers.maximum(tmp72, tmp83)
    tmp85 = tl.full([1], 1, tl.int32)
    tmp86 = (tmp85 / tmp84)
    tmp87 = 7.319
    tmp88 = tmp86 * tmp87
    tmp89 = triton_helpers.minimum(tmp88, tmp40)
    tmp90 = 7.0
    tmp91 = tmp89 * tmp90
    tmp92 = tmp82 <= tmp91
    tmp93 = tl.where(tmp92, tmp72, tmp22)
    tmp94 = tl.where(tmp92, tmp70, tmp72)
    tmp95 = tmp93 + tmp94
    tmp96 = tmp95 * tmp71
    tmp97 = tl_math.abs(tmp96)
    tmp98 = tmp96 * tmp96
    tmp99 = triton_helpers.maximum(tmp96, tmp83)
    tmp100 = tmp97 * tmp74
    tmp101 = libdevice.tanh(tmp100)
    tmp102 = tmp101 * tmp77
    tmp103 = tmp98 * tmp80
    tmp104 = tmp102 + tmp103
    tmp105 = (tmp85 / tmp99)
    tmp106 = tmp105 * tmp87
    tmp107 = triton_helpers.minimum(tmp106, tmp40)
    tmp108 = tmp107 * tmp90
    tmp109 = tmp104 <= tmp108
    tmp110 = tl.where(tmp109, tmp96, tmp93)
    tmp111 = tl.where(tmp109, tmp94, tmp96)
    tmp112 = tmp110 + tmp111
    tmp113 = tmp112 * tmp71
    tmp114 = tl_math.abs(tmp113)
    tmp115 = tmp114 * tmp74
    tmp116 = libdevice.tanh(tmp115)
    tmp117 = tmp116 * tmp77
    tmp118 = tmp113 * tmp113
    tmp119 = tmp118 * tmp80
    tmp120 = tmp117 + tmp119
    tmp121 = triton_helpers.maximum(tmp113, tmp83)
    tmp122 = (tmp85 / tmp121)
    tmp123 = tmp122 * tmp87
    tmp124 = triton_helpers.minimum(tmp123, tmp40)
    tmp125 = tmp124 * tmp90
    tmp126 = tmp120 <= tmp125
    tmp127 = tl.where(tmp126, tmp113, tmp110)
    tmp128 = tl.where(tmp126, tmp111, tmp113)
    tmp129 = tmp127 + tmp128
    tmp130 = tmp129 * tmp71
    tmp131 = tl_math.abs(tmp130)
    tmp132 = tmp131 * tmp74
    tmp133 = tmp130 * tmp130
    tmp134 = tmp133 * tmp80
    tmp135 = triton_helpers.maximum(tmp130, tmp83)
    tmp136 = (tmp85 / tmp135)
    tmp137 = libdevice.tanh(tmp132)
    tmp138 = tmp137 * tmp77
    tmp139 = tmp138 + tmp134
    tmp140 = tmp136 * tmp87
    tmp141 = triton_helpers.minimum(tmp140, tmp40)
    tmp142 = tmp141 * tmp90
    tmp143 = tmp139 <= tmp142
    tmp144 = tl.where(tmp143, tmp130, tmp127)
    tmp145 = tl.where(tmp143, tmp128, tmp130)
    tmp146 = tmp144 + tmp145
    tmp147 = tmp146 * tmp71
    tmp148 = tl_math.abs(tmp147)
    tmp149 = tmp148 * tmp74
    tmp150 = libdevice.tanh(tmp149)
    tmp151 = tmp150 * tmp77
    tmp152 = tmp147 * tmp147
    tmp153 = tmp152 * tmp80
    tmp154 = tmp151 + tmp153
    tmp155 = triton_helpers.maximum(tmp147, tmp83)
    tmp156 = (tmp85 / tmp155)
    tmp157 = tmp156 * tmp87
    tmp158 = triton_helpers.minimum(tmp157, tmp40)
    tmp159 = tmp158 * tmp90
    tmp160 = tmp154 <= tmp159
    tmp161 = tl.where(tmp160, tmp147, tmp144)
    tmp162 = tl.where(tmp160, tmp145, tmp147)
    tmp163 = tmp161 + tmp162
    tmp164 = tmp163 * tmp71
    tmp165 = tl_math.abs(tmp164)
    tmp166 = tmp165 * tmp74
    tmp167 = tmp164 * tmp164
    tmp168 = tmp167 * tmp80
    tmp169 = triton_helpers.maximum(tmp164, tmp83)
    tmp170 = (tmp85 / tmp169)
    tmp171 = libdevice.tanh(tmp166)
    tmp172 = tmp171 * tmp77
    tmp173 = tmp172 + tmp168
    tmp174 = tmp170 * tmp87
    tmp175 = triton_helpers.minimum(tmp174, tmp40)
    tmp176 = tmp175 * tmp90
    tmp177 = tmp173 <= tmp176
    tmp178 = tl.where(tmp177, tmp164, tmp161)
    tmp179 = tl.where(tmp177, tmp162, tmp164)
    tmp180 = tmp178 + tmp179
    tmp181 = tmp180 * tmp71
    tmp182 = tl_math.abs(tmp181)
    tmp183 = tmp182 * tmp74
    tmp184 = libdevice.tanh(tmp183)
    tmp185 = tmp184 * tmp77
    tmp186 = tmp181 * tmp181
    tmp187 = tmp186 * tmp80
    tmp188 = tmp185 + tmp187
    tmp189 = triton_helpers.maximum(tmp181, tmp83)
    tmp190 = (tmp85 / tmp189)
    tmp191 = tmp190 * tmp87
    tmp192 = triton_helpers.minimum(tmp191, tmp40)
    tmp193 = tmp192 * tmp90
    tmp194 = tmp188 <= tmp193
    tmp195 = tl.where(tmp194, tmp181, tmp178)
    tmp196 = tl.where(tmp194, tmp179, tmp181)
    tmp197 = tmp195 + tmp196
    tmp198 = tmp197 * tmp71
    tmp199 = tl_math.abs(tmp198)
    tmp200 = tmp199 * tmp74
    tmp201 = tmp198 * tmp198
    tmp202 = tmp201 * tmp80
    tmp203 = triton_helpers.maximum(tmp198, tmp83)
    tmp204 = (tmp85 / tmp203)
    tmp205 = libdevice.tanh(tmp200)
    tmp206 = tmp205 * tmp77
    tmp207 = tmp206 + tmp202
    tmp208 = tmp204 * tmp87
    tmp209 = triton_helpers.minimum(tmp208, tmp40)
    tmp210 = tmp209 * tmp90
    tmp211 = tmp207 <= tmp210
    tmp212 = tl.where(tmp211, tmp198, tmp195)
    tmp213 = tl.where(tmp211, tmp196, tmp198)
    tmp214 = tmp212 + tmp213
    tmp215 = tmp214 * tmp71
    tmp216 = tl_math.abs(tmp215)
    tmp217 = tmp216 * tmp74
    tmp218 = libdevice.tanh(tmp217)
    tmp219 = tmp218 * tmp77
    tmp220 = tmp215 * tmp215
    tmp221 = tmp220 * tmp80
    tmp222 = tmp219 + tmp221
    tmp223 = triton_helpers.maximum(tmp215, tmp83)
    tmp224 = (tmp85 / tmp223)
    tmp225 = tmp224 * tmp87
    tmp226 = triton_helpers.minimum(tmp225, tmp40)
    tmp227 = tmp226 * tmp90
    tmp228 = tmp222 <= tmp227
    tmp229 = tl.where(tmp228, tmp215, tmp212)
    tmp230 = tl.where(tmp228, tmp213, tmp215)
    tmp231 = tmp229 + tmp230
    tmp232 = tmp231 * tmp71
    tmp233 = tl_math.abs(tmp232)
    tmp234 = tmp233 * tmp74
    tmp235 = tmp232 * tmp232
    tmp236 = tmp235 * tmp80
    tmp237 = triton_helpers.maximum(tmp232, tmp83)
    tmp238 = (tmp85 / tmp237)
    tmp239 = libdevice.tanh(tmp234)
    tmp240 = tmp239 * tmp77
    tmp241 = tmp240 + tmp236
    tmp242 = tmp238 * tmp87
    tmp243 = triton_helpers.minimum(tmp242, tmp40)
    tmp244 = tmp243 * tmp90
    tmp245 = tmp241 <= tmp244
    tmp246 = tl.where(tmp245, tmp232, tmp229)
    tmp247 = tl.where(tmp245, tmp230, tmp232)
    tmp248 = tmp246 + tmp247
    tmp249 = tmp248 * tmp71
    tmp250 = tl_math.abs(tmp249)
    tmp251 = tmp250 * tmp74
    tmp252 = libdevice.tanh(tmp251)
    tmp253 = tmp252 * tmp77
    tmp254 = tmp249 * tmp249
    tmp255 = tmp254 * tmp80
    tmp256 = tmp253 + tmp255
    tmp257 = triton_helpers.maximum(tmp249, tmp83)
    tmp258 = (tmp85 / tmp257)
    tmp259 = tmp258 * tmp87
    tmp260 = triton_helpers.minimum(tmp259, tmp40)
    tmp261 = tmp260 * tmp90
    tmp262 = tmp256 <= tmp261
    tmp263 = tl.where(tmp262, tmp249, tmp246)
    tmp264 = tl.where(tmp262, tmp247, tmp249)
    tmp265 = tmp263 + tmp264
    tmp266 = tmp265 * tmp71
    tmp267 = tl_math.abs(tmp266)
    tmp268 = tmp267 * tmp74
    tmp269 = tmp266 * tmp266
    tmp270 = tmp269 * tmp80
    tmp271 = triton_helpers.maximum(tmp266, tmp83)
    tmp272 = (tmp85 / tmp271)
    tmp273 = libdevice.tanh(tmp268)
    tmp274 = tmp273 * tmp77
    tmp275 = tmp274 + tmp270
    tmp276 = tmp272 * tmp87
    tmp277 = triton_helpers.minimum(tmp276, tmp40)
    tmp278 = tmp277 * tmp90
    tmp279 = tmp275 <= tmp278
    tmp280 = tl.where(tmp279, tmp264, tmp266)
    tl.store(in_out_ptr0 + (x2), tmp68, xmask)
    tl.store(in_out_ptr1 + (x2), tmp280, xmask)
''', device_str='cuda')


# kernel path: /home/shchon11/F1tenth/F1tenth_E2E/work/adaptive-racing-v3/controller-proof/inductor-cold-v1/wq/cwq5roiwkump32bjrjmuwrufrgsvqrgrnip732otm5tlsztil4ri.py
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
#   %arg3_1 : Tensor "f32[48][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg0_1 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sqrt_2 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=sqrt_2]
#   %where_27 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_27]
#   %arg2_1 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %unsqueeze : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, 1), kwargs = {})
#   %mul_77 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, 3.2), kwargs = {})
#   %minimum_2 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.minimum.default](args = (%arg0_1, %sqrt_2), kwargs = {})
#   %minimum_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.minimum.default](args = (%minimum_2, %where_27), kwargs = {})
#   %pow_19 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%minimum_3, 2), kwargs = {})
#   %mul_75 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_19, 0.003), kwargs = {})
#   %add_28 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_75, 0.3302), kwargs = {})
#   %mul_76 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, %arg2_1), kwargs = {})
#   %atan : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.atan.default](args = (%mul_76,), kwargs = {})
#   %slice_1 : Tensor "f32[48, 24][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%atan, 1, 1, 9223372036854775807), kwargs = {})
#   %slice_2 : Tensor "f32[48, 24][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%atan, 1, 0, -1), kwargs = {})
#   %sub_4 : Tensor "f32[48, 24][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, %slice_2), kwargs = {})
#   %abs_13 : Tensor "f32[48, 24][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_4,), kwargs = {})
#   %clamp_min_17 : Tensor "f32[48, 24][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%abs_13, 1e-06), kwargs = {})
#   %div_14 : Tensor "f32[48, 24][24, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_77, %clamp_min_17), kwargs = {})
#   return %div_14
triton_poi_fused_abs_add_atan_clamp_min_div_minimum_mul_pow_slice_sub_unsqueeze_1 = async_compile.triton('triton_poi_fused_abs_add_atan_clamp_min_div_minimum_mul_pow_slice_sub_unsqueeze_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_atan_clamp_min_div_minimum_mul_pow_slice_sub_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 46080}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_atan_clamp_min_div_minimum_mul_pow_slice_sub_unsqueeze_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 24
    x0 = (xindex % 24)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (1 + x0 + 25*x1), xmask)
    tmp4 = tl.load(in_ptr2 + (1 + x0 + 25*x1), xmask)
    tmp6 = tl.load(in_ptr3 + (1 + x0 + 25*x1), xmask)
    tmp13 = tl.load(in_ptr4 + (1 + x0 + 25*x1), xmask)
    tmp16 = tl.load(in_ptr1 + (x0 + 25*x1), xmask)
    tmp17 = tl.load(in_ptr2 + (x0 + 25*x1), xmask)
    tmp19 = tl.load(in_ptr3 + (x0 + 25*x1), xmask)
    tmp24 = tl.load(in_ptr4 + (x0 + 25*x1), xmask)
    tmp1 = 3.2
    tmp2 = tmp0 * tmp1
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp7 = triton_helpers.minimum(tmp5, tmp6)
    tmp8 = tmp7 * tmp7
    tmp9 = 0.003
    tmp10 = tmp8 * tmp9
    tmp11 = 0.3302
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 * tmp13
    tmp15 = libdevice.atan(tmp14)
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp20 * tmp20
    tmp22 = tmp21 * tmp9
    tmp23 = tmp22 + tmp11
    tmp25 = tmp23 * tmp24
    tmp26 = libdevice.atan(tmp25)
    tmp27 = tmp15 - tmp26
    tmp28 = tl_math.abs(tmp27)
    tmp29 = 1e-06
    tmp30 = triton_helpers.maximum(tmp28, tmp29)
    tmp31 = (tmp2 / tmp30)
    tl.store(out_ptr0 + (x2), tmp31, xmask)
''', device_str='cuda')


# kernel path: /home/shchon11/F1tenth/F1tenth_E2E/work/adaptive-racing-v3/controller-proof/inductor-cold-v1/o2/co2rt7cnvkh2ahwxkcplln353o243ntjsv6s7dm5qny2p3koabk4.py
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
#   %arg0_1 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sqrt_2 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=sqrt_2]
#   %where_27 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_27]
#   %div_14 : Tensor "f32[48, 24][24, 1]cuda:0" = PlaceHolder[target=div_14]
#   %arg1_1 : Tensor "f32[48, 1][1, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %minimum_5 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=minimum_5]
#   %arg2_1 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %mul_99 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=mul_99]
#   %clamp_min_21 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_21]
#   %clamp_min_22 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_22]
#   %ge_6 : Tensor "b8[48, 25][25, 1]cuda:0" = PlaceHolder[target=ge_6]
#   %gt_4 : Tensor "b8[48, 25][25, 1]cuda:0" = PlaceHolder[target=gt_4]
#   %div_17 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=div_17]
#   %mul_116 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=mul_116]
#   %clamp_min_25 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_25]
#   %clamp_min_26 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_26]
#   %ge_10 : Tensor "b8[48, 25][25, 1]cuda:0" = PlaceHolder[target=ge_10]
#   %gt_7 : Tensor "b8[48, 25][25, 1]cuda:0" = PlaceHolder[target=gt_7]
#   %div_19 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=div_19]
#   %sub_23 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=sub_23]
#   %div_21 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=div_21]
#   %sub_28 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=sub_28]
#   %div_23 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=div_23]
#   %mul_96 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=mul_96]
#   %clamp_min_19 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_19]
#   %clamp_min_20 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_20]
#   %ge_4 : Tensor "b8[48, 25][25, 1]cuda:0" = PlaceHolder[target=ge_4]
#   %gt_3 : Tensor "b8[48, 25][25, 1]cuda:0" = PlaceHolder[target=gt_3]
#   %div_16 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=div_16]
#   %mul_113 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=mul_113]
#   %clamp_min_23 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_23]
#   %clamp_min_24 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=clamp_min_24]
#   %ge_8 : Tensor "b8[48, 25][25, 1]cuda:0" = PlaceHolder[target=ge_8]
#   %gt_6 : Tensor "b8[48, 25][25, 1]cuda:0" = PlaceHolder[target=gt_6]
#   %div_18 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=div_18]
#   %sub_22 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=sub_22]
#   %div_20 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=div_20]
#   %sub_27 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=sub_27]
#   %div_22 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=div_22]
#   %where_31 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_31]
#   %where_35 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_35]
#   %where_39 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_39]
#   %where_43 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_43]
#   %where_29 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_29]
#   %where_33 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_33]
#   %where_37 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_37]
#   %where_41 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=where_41]
#   %minimum_10 : Tensor "f32[48, 25][25, 1]cuda:0" = PlaceHolder[target=minimum_10]
#   %minimum_2 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.minimum.default](args = (%arg0_1, %sqrt_2), kwargs = {})
#   %minimum_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.minimum.default](args = (%minimum_2, %where_27), kwargs = {})
#   %slice_3 : Tensor "f32[48, 1][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%div_14, 1, 0, 1), kwargs = {})
#   %slice_4 : Tensor "f32[48, 23][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%div_14, 1, 0, -1), kwargs = {})
#   %slice_5 : Tensor "f32[48, 23][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%div_14, 1, 1, 9223372036854775807), kwargs = {})
#   %minimum_4 : Tensor "f32[48, 23][23, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%slice_4, %slice_5), kwargs = {})
#   %slice_6 : Tensor "f32[48, 1][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%div_14, 1, -1, 9223372036854775807), kwargs = {})
#   %cat : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_3, %minimum_4, %slice_6], 1), kwargs = {})
#   %minimum_5 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=9] = call_function[target=torch.ops.aten.minimum.default](args = (%minimum_3, %cat), kwargs = {})
#   %abs_15 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%minimum_5,), kwargs = {})
#   %clamp_min_18 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%abs_15, 0.001), kwargs = {})
#   %reciprocal_12 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_18,), kwargs = {})
#   %mul_80 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_12, 7.319), kwargs = {})
#   %clamp_max_12 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_80, 1), kwargs = {})
#   %mul_81 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_12, 7.0), kwargs = {})
#   %abs_14 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%minimum_5,), kwargs = {})
#   %div_15 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_14, 0.05), kwargs = {})
#   %tanh_12 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_15,), kwargs = {})
#   %mul_78 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_12, 0.1), kwargs = {})
#   %pow_20 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%minimum_5, 2), kwargs = {})
#   %mul_79 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_20, 0.01), kwargs = {})
#   %add_29 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_78, %mul_79), kwargs = {})
#   %sub_6 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_81, %add_29), kwargs = {})
#   %mul_90 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, 0.5), kwargs = {})
#   %pow_23 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_90, 2), kwargs = {})
#   %pow_24 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%minimum_5, 2), kwargs = {})
#   %mul_91 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_24, 0.5192307692307692), kwargs = {})
#   %mul_92 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %arg2_1), kwargs = {})
#   %pow_25 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_92, 2), kwargs = {})
#   %add_30 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_23, %pow_25), kwargs = {})
#   %mul_82 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 0.92), kwargs = {})
#   %mul_93 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, 9.81), kwargs = {})
#   %mul_94 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_93, 0.5192307692307692), kwargs = {})
#   %pow_26 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_94, 2), kwargs = {})
#   %sub_9 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_30, %pow_26), kwargs = {})
#   %ge_7 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_9, 0), kwargs = {})
#   %full_default_13 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_85 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, 0.25), kwargs = {})
#   %pow_22 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_82, 2), kwargs = {})
#   %mul_86 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_22, 9.81), kwargs = {})
#   %mul_87 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_86, 0.5192307692307692), kwargs = {})
#   %mul_88 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_87, -0.22410660205935795), kwargs = {})
#   %sub_8 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_85, %mul_88), kwargs = {})
#   %mul_89 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, 2), kwargs = {})
#   %pow_28 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_89, 2), kwargs = {})
#   %mul_84 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, -0.22410660205935795), kwargs = {})
#   %pow_21 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_84, 2), kwargs = {})
#   %sub_7 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_21), kwargs = {})
#   %mul_98 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, 4), kwargs = {})
#   %mul_99 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_98, %sub_9), kwargs = {})
#   %sub_11 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_28, %mul_99), kwargs = {})
#   %ge_6 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_11, 0), kwargs = {})
#   %clamp_min_21 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_11, 0), kwargs = {})
#   %sqrt_4 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_21,), kwargs = {})
#   %add_32 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %sqrt_4), kwargs = {})
#   %gt_4 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_32, 1e-12), kwargs = {})
#   %bitwise_and_3 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_6, %gt_4), kwargs = {})
#   %mul_100 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, -2), kwargs = {})
#   %clamp_min_22 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_32, 1e-12), kwargs = {})
#   %div_17 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_100, %clamp_min_22), kwargs = {})
#   %full_default_12 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_30 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_3, %div_17, %full_default_12), kwargs = {})
#   %where_31 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_7, %full_default_13, %where_30), kwargs = {})
#   %minimum_6 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%sub_6, %where_31), kwargs = {})
#   %clamp_max_13 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%minimum_6, 22.72870945945946), kwargs = {})
#   %mul_107 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, 0.5), kwargs = {})
#   %pow_31 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_107, 2), kwargs = {})
#   %pow_32 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%minimum_5, 2), kwargs = {})
#   %mul_108 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_32, 0.4807692307692308), kwargs = {})
#   %mul_109 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_108, %arg2_1), kwargs = {})
#   %pow_33 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_109, 2), kwargs = {})
#   %add_33 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_31, %pow_33), kwargs = {})
#   %mul_83 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 1.0), kwargs = {})
#   %mul_110 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_83, 9.81), kwargs = {})
#   %mul_111 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_110, 0.4807692307692308), kwargs = {})
#   %pow_34 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_111, 2), kwargs = {})
#   %sub_14 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_33, %pow_34), kwargs = {})
#   %ge_11 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_14, 0), kwargs = {})
#   %full_default_17 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_102 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, 0.25), kwargs = {})
#   %pow_30 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_83, 2), kwargs = {})
#   %mul_103 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_30, 9.81), kwargs = {})
#   %mul_104 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_103, 0.4807692307692308), kwargs = {})
#   %mul_105 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_104, 0.22410660205935795), kwargs = {})
#   %sub_13 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_102, %mul_105), kwargs = {})
#   %mul_106 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, 2), kwargs = {})
#   %pow_36 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_106, 2), kwargs = {})
#   %mul_101 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_83, 0.22410660205935795), kwargs = {})
#   %pow_29 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_101, 2), kwargs = {})
#   %sub_12 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_29), kwargs = {})
#   %mul_115 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, 4), kwargs = {})
#   %mul_116 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_115, %sub_14), kwargs = {})
#   %sub_16 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_36, %mul_116), kwargs = {})
#   %ge_10 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_16, 0), kwargs = {})
#   %clamp_min_25 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_16, 0), kwargs = {})
#   %sqrt_6 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_25,), kwargs = {})
#   %add_35 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_106, %sqrt_6), kwargs = {})
#   %gt_7 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_35, 1e-12), kwargs = {})
#   %bitwise_and_5 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_10, %gt_7), kwargs = {})
#   %mul_117 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, -2), kwargs = {})
#   %clamp_min_26 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_35, 1e-12), kwargs = {})
#   %div_19 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_117, %clamp_min_26), kwargs = {})
#   %full_default_16 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_34 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_5, %div_19, %full_default_16), kwargs = {})
#   %where_35 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_11, %full_default_17, %where_34), kwargs = {})
#   %minimum_7 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%clamp_max_13, %where_35), kwargs = {})
#   %abs_16 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%minimum_5,), kwargs = {})
#   %clamp_min_28 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%abs_16, 0.001), kwargs = {})
#   %reciprocal_13 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_28,), kwargs = {})
#   %mul_118 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_13, 7.319), kwargs = {})
#   %clamp_max_14 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_118, 1), kwargs = {})
#   %mul_119 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_14, 7.0), kwargs = {})
#   %pow_40 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%minimum_5, 2), kwargs = {})
#   %mul_129 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_40, 0.5192307692307692), kwargs = {})
#   %mul_130 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_129, %arg2_1), kwargs = {})
#   %pow_41 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_130, 2), kwargs = {})
#   %mul_120 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 0.92), kwargs = {})
#   %mul_131 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_120, 9.81), kwargs = {})
#   %mul_132 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_131, 0.5192307692307692), kwargs = {})
#   %pow_42 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_132, 2), kwargs = {})
#   %sub_21 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_41, %pow_42), kwargs = {})
#   %ge_15 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_21, 0), kwargs = {})
#   %full_default_25 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_20 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_38 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_120, 2), kwargs = {})
#   %mul_124 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_38, 9.81), kwargs = {})
#   %mul_125 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_124, 0.5192307692307692), kwargs = {})
#   %mul_126 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_125, -0.22410660205935795), kwargs = {})
#   %sub_20 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_20, %mul_126), kwargs = {})
#   %mul_127 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, 2), kwargs = {})
#   %pow_44 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_127, 2), kwargs = {})
#   %mul_122 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_120, -0.22410660205935795), kwargs = {})
#   %pow_37 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_122, 2), kwargs = {})
#   %sub_19 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_37), kwargs = {})
#   %mul_136 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, 4), kwargs = {})
#   %mul_137 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_136, %sub_21), kwargs = {})
#   %sub_23 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_44, %mul_137), kwargs = {})
#   %ge_14 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_23, 0), kwargs = {})
#   %clamp_min_31 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_23, 0), kwargs = {})
#   %sqrt_8 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_31,), kwargs = {})
#   %add_38 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_127, %sqrt_8), kwargs = {})
#   %gt_11 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_38, 1e-12), kwargs = {})
#   %bitwise_and_7 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_14, %gt_11), kwargs = {})
#   %mul_138 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, -2), kwargs = {})
#   %clamp_min_32 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_38, 1e-12), kwargs = {})
#   %div_21 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_138, %clamp_min_32), kwargs = {})
#   %full_default_24 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_38 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_7, %div_21, %full_default_24), kwargs = {})
#   %where_39 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_15, %full_default_25, %where_38), kwargs = {})
#   %minimum_8 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%mul_119, %where_39), kwargs = {})
#   %clamp_max_15 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%minimum_8, 22.72870945945946), kwargs = {})
#   %pow_48 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%minimum_5, 2), kwargs = {})
#   %mul_146 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_48, 0.4807692307692308), kwargs = {})
#   %mul_147 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_146, %arg2_1), kwargs = {})
#   %pow_49 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_147, 2), kwargs = {})
#   %mul_121 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 1.0), kwargs = {})
#   %mul_148 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_121, 9.81), kwargs = {})
#   %mul_149 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, 0.4807692307692308), kwargs = {})
#   %pow_50 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_149, 2), kwargs = {})
#   %sub_26 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_49, %pow_50), kwargs = {})
#   %ge_19 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_26, 0), kwargs = {})
#   %full_default_31 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_26 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_46 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_121, 2), kwargs = {})
#   %mul_141 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_46, 9.81), kwargs = {})
#   %mul_142 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_141, 0.4807692307692308), kwargs = {})
#   %mul_143 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_142, 0.22410660205935795), kwargs = {})
#   %sub_25 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_26, %mul_143), kwargs = {})
#   %mul_144 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, 2), kwargs = {})
#   %pow_52 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_144, 2), kwargs = {})
#   %mul_139 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_121, 0.22410660205935795), kwargs = {})
#   %pow_45 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_139, 2), kwargs = {})
#   %sub_24 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_45), kwargs = {})
#   %mul_153 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, 4), kwargs = {})
#   %mul_154 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_153, %sub_26), kwargs = {})
#   %sub_28 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_52, %mul_154), kwargs = {})
#   %ge_18 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_28, 0), kwargs = {})
#   %clamp_min_35 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_28, 0), kwargs = {})
#   %sqrt_10 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_35,), kwargs = {})
#   %add_41 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_144, %sqrt_10), kwargs = {})
#   %gt_14 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_41, 1e-12), kwargs = {})
#   %bitwise_and_9 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_18, %gt_14), kwargs = {})
#   %mul_155 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, -2), kwargs = {})
#   %clamp_min_36 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_41, 1e-12), kwargs = {})
#   %div_23 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_155, %clamp_min_36), kwargs = {})
#   %full_default_30 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_42 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_9, %div_23, %full_default_30), kwargs = {})
#   %where_43 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_19, %full_default_31, %where_42), kwargs = {})
#   %minimum_9 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%clamp_max_15, %where_43), kwargs = {})
#   %minimum_10 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%minimum_7, %minimum_9), kwargs = {})
#   %clamp_min_38 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%minimum_10, 0), kwargs = {})
#   %sub_5 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (-5.0, %add_29), kwargs = {})
#   %ge_5 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_9, 0), kwargs = {})
#   %full_default_11 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_89,), kwargs = {})
#   %pow_27 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg, 2), kwargs = {})
#   %mul_95 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, 4), kwargs = {})
#   %mul_96 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_95, %sub_9), kwargs = {})
#   %sub_10 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_27, %mul_96), kwargs = {})
#   %ge_4 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_10, 0), kwargs = {})
#   %clamp_min_19 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_10, 0), kwargs = {})
#   %sqrt_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_19,), kwargs = {})
#   %add_31 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg, %sqrt_3), kwargs = {})
#   %gt_3 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_31, 1e-12), kwargs = {})
#   %bitwise_and_2 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_4, %gt_3), kwargs = {})
#   %mul_97 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, -2), kwargs = {})
#   %clamp_min_20 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_31, 1e-12), kwargs = {})
#   %div_16 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_97, %clamp_min_20), kwargs = {})
#   %full_default_10 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_28 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_2, %div_16, %full_default_10), kwargs = {})
#   %where_29 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_5, %full_default_11, %where_28), kwargs = {})
#   %neg_1 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_29,), kwargs = {})
#   %maximum : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%sub_5, %neg_1), kwargs = {})
#   %ge_9 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_14, 0), kwargs = {})
#   %full_default_15 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg_2 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_106,), kwargs = {})
#   %pow_35 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg_2, 2), kwargs = {})
#   %mul_112 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, 4), kwargs = {})
#   %mul_113 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %sub_14), kwargs = {})
#   %sub_15 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_35, %mul_113), kwargs = {})
#   %ge_8 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_15, 0), kwargs = {})
#   %clamp_min_23 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_15, 0), kwargs = {})
#   %sqrt_5 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_23,), kwargs = {})
#   %add_34 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_2, %sqrt_5), kwargs = {})
#   %gt_6 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_34, 1e-12), kwargs = {})
#   %bitwise_and_4 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_8, %gt_6), kwargs = {})
#   %mul_114 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, -2), kwargs = {})
#   %clamp_min_24 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_34, 1e-12), kwargs = {})
#   %div_18 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_114, %clamp_min_24), kwargs = {})
#   %full_default_14 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_32 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_4, %div_18, %full_default_14), kwargs = {})
#   %where_33 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_9, %full_default_15, %where_32), kwargs = {})
#   %neg_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_33,), kwargs = {})
#   %maximum_1 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%maximum, %neg_3), kwargs = {})
#   %clamp_min_27 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%maximum_1, -21.045101351351356), kwargs = {})
#   %full_default_19 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], -5.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ge_13 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_21, 0), kwargs = {})
#   %full_default_23 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg_4 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_127,), kwargs = {})
#   %pow_43 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg_4, 2), kwargs = {})
#   %mul_133 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, 4), kwargs = {})
#   %mul_134 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_133, %sub_21), kwargs = {})
#   %sub_22 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_43, %mul_134), kwargs = {})
#   %ge_12 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_22, 0), kwargs = {})
#   %clamp_min_29 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_22, 0), kwargs = {})
#   %sqrt_7 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_29,), kwargs = {})
#   %add_37 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_4, %sqrt_7), kwargs = {})
#   %gt_10 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_37, 1e-12), kwargs = {})
#   %bitwise_and_6 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_12, %gt_10), kwargs = {})
#   %mul_135 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, -2), kwargs = {})
#   %clamp_min_30 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_37, 1e-12), kwargs = {})
#   %div_20 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_135, %clamp_min_30), kwargs = {})
#   %full_default_22 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_36 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_6, %div_20, %full_default_22), kwargs = {})
#   %where_37 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_13, %full_default_23, %where_36), kwargs = {})
#   %neg_5 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_37,), kwargs = {})
#   %maximum_2 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%full_default_19, %neg_5), kwargs = {})
#   %ge_17 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_26, 0), kwargs = {})
#   %full_default_29 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg_6 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_144,), kwargs = {})
#   %pow_51 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg_6, 2), kwargs = {})
#   %mul_150 : Tensor "f32[48, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, 4), kwargs = {})
#   %mul_151 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_150, %sub_26), kwargs = {})
#   %sub_27 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_51, %mul_151), kwargs = {})
#   %ge_16 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_27, 0), kwargs = {})
#   %clamp_min_33 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_27, 0), kwargs = {})
#   %sqrt_9 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_33,), kwargs = {})
#   %add_40 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_6, %sqrt_9), kwargs = {})
#   %gt_13 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_40, 1e-12), kwargs = {})
#   %bitwise_and_8 : Tensor "b8[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_16, %gt_13), kwargs = {})
#   %mul_152 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, -2), kwargs = {})
#   %clamp_min_34 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_40, 1e-12), kwargs = {})
#   %div_22 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_152, %clamp_min_34), kwargs = {})
#   %full_default_28 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([48, 25], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_40 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_8, %div_22, %full_default_28), kwargs = {})
#   %where_41 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_17, %full_default_29, %where_40), kwargs = {})
#   %neg_7 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_41,), kwargs = {})
#   %maximum_3 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%maximum_2, %neg_7), kwargs = {})
#   %clamp_min_37 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%maximum_3, -21.045101351351356), kwargs = {})
#   %maximum_4 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%clamp_min_27, %clamp_min_37), kwargs = {})
#   %neg_8 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%maximum_4,), kwargs = {})
#   %clamp_min_39 : Tensor "f32[48, 25][25, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%neg_8, 0), kwargs = {})
#   return %minimum_5,%mul_99,%ge_6,%clamp_min_21,%gt_4,%clamp_min_22,%div_17,%where_31,%mul_116,%ge_10,%clamp_min_25,%gt_7,%clamp_min_26,%div_19,%where_35,%sub_23,%div_21,%where_39,%sub_28,%div_23,%where_43,%mul_96,%ge_4,%clamp_min_19,%gt_3,%clamp_min_20,%div_16,%where_29,%mul_113,%ge_8,%clamp_min_23,%gt_6,%clamp_min_24,%div_18,%where_33,%sub_22,%div_20,%where_37,%sub_27,%div_22,%where_41,%minimum_10,%clamp_min_39,%clamp_min_38
triton_poi_fused_abs_add_bitwise_and_cat_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_slice_sqrt_sub_tanh_where_zeros_like_2 = async_compile.triton('triton_poi_fused_abs_add_bitwise_and_cat_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_slice_sqrt_sub_tanh_where_zeros_like_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr5': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_bitwise_and_cat_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_slice_sqrt_sub_tanh_where_zeros_like_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr5'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 3, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 57216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_bitwise_and_cat_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_slice_sqrt_sub_tanh_where_zeros_like_2(in_out_ptr0, in_out_ptr1, in_out_ptr5, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 25)
    x1 = xindex // 25
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp27 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr4 + (x2), xmask)
    tmp2 = triton_helpers.minimum(tmp0, tmp1)
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tl.load(in_ptr2 + (24*x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp5 >= tmp8
    tmp12 = tl.full([1], 24, tl.int64)
    tmp13 = tmp5 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (24*x1 + ((-1) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr2 + (1 + 24*x1 + ((-1) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp14, tmp17, tmp18)
    tmp20 = tmp5 >= tmp12
    tmp21 = tl.full([1], 25, tl.int64)
    tmp22 = tmp5 < tmp21
    tmp23 = tl.load(in_ptr2 + (23 + 24*x1), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp14, tmp19, tmp23)
    tmp25 = tl.where(tmp9, tmp10, tmp24)
    tmp26 = triton_helpers.minimum(tmp4, tmp25)
    tmp28 = 0.92
    tmp29 = tmp27 * tmp28
    tmp30 = -0.22410660205935795
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = 0.25
    tmp34 = tmp33 - tmp32
    tmp35 = 4.0
    tmp36 = tmp34 * tmp35
    tmp37 = tl_math.abs(tmp26)
    tmp38 = 20.0
    tmp39 = tmp37 * tmp38
    tmp40 = libdevice.tanh(tmp39)
    tmp41 = 0.1
    tmp42 = tmp40 * tmp41
    tmp43 = tmp26 * tmp26
    tmp44 = 0.01
    tmp45 = tmp43 * tmp44
    tmp46 = tmp42 + tmp45
    tmp47 = 0.5
    tmp48 = tmp46 * tmp47
    tmp49 = tmp48 * tmp48
    tmp50 = 0.5192307692307692
    tmp51 = tmp43 * tmp50
    tmp53 = tmp51 * tmp52
    tmp54 = tmp53 * tmp53
    tmp55 = tmp49 + tmp54
    tmp56 = 9.81
    tmp57 = tmp29 * tmp56
    tmp58 = tmp57 * tmp50
    tmp59 = tmp58 * tmp58
    tmp60 = tmp55 - tmp59
    tmp61 = tmp36 * tmp60
    tmp62 = tmp46 * tmp33
    tmp63 = tmp29 * tmp29
    tmp64 = tmp63 * tmp56
    tmp65 = tmp64 * tmp50
    tmp66 = tmp65 * tmp30
    tmp67 = tmp62 - tmp66
    tmp68 = 2.0
    tmp69 = tmp67 * tmp68
    tmp70 = tmp69 * tmp69
    tmp71 = tmp70 - tmp61
    tmp72 = 0.0
    tmp73 = tmp71 >= tmp72
    tmp74 = triton_helpers.maximum(tmp71, tmp72)
    tmp75 = tl.sqrt_rn(tmp74)
    tmp76 = tmp69 + tmp75
    tmp77 = 1e-12
    tmp78 = tmp76 > tmp77
    tmp79 = triton_helpers.maximum(tmp76, tmp77)
    tmp80 = -2.0
    tmp81 = tmp60 * tmp80
    tmp82 = (tmp81 / tmp79)
    tmp83 = tmp60 >= tmp72
    tmp84 = tmp73 & tmp78
    tmp85 = float("inf")
    tmp86 = tl.where(tmp84, tmp82, tmp85)
    tmp87 = tl.where(tmp83, tmp72, tmp86)
    tmp88 = 1.0
    tmp89 = tmp27 * tmp88
    tmp90 = 0.22410660205935795
    tmp91 = tmp89 * tmp90
    tmp92 = tmp91 * tmp91
    tmp93 = tmp33 - tmp92
    tmp94 = tmp93 * tmp35
    tmp95 = 0.4807692307692308
    tmp96 = tmp43 * tmp95
    tmp97 = tmp96 * tmp52
    tmp98 = tmp97 * tmp97
    tmp99 = tmp49 + tmp98
    tmp100 = tmp89 * tmp56
    tmp101 = tmp100 * tmp95
    tmp102 = tmp101 * tmp101
    tmp103 = tmp99 - tmp102
    tmp104 = tmp94 * tmp103
    tmp105 = tmp89 * tmp89
    tmp106 = tmp105 * tmp56
    tmp107 = tmp106 * tmp95
    tmp108 = tmp107 * tmp90
    tmp109 = tmp62 - tmp108
    tmp110 = tmp109 * tmp68
    tmp111 = tmp110 * tmp110
    tmp112 = tmp111 - tmp104
    tmp113 = tmp112 >= tmp72
    tmp114 = triton_helpers.maximum(tmp112, tmp72)
    tmp115 = tl.sqrt_rn(tmp114)
    tmp116 = tmp110 + tmp115
    tmp117 = tmp116 > tmp77
    tmp118 = triton_helpers.maximum(tmp116, tmp77)
    tmp119 = tmp103 * tmp80
    tmp120 = (tmp119 / tmp118)
    tmp121 = tmp103 >= tmp72
    tmp122 = tmp113 & tmp117
    tmp123 = tl.where(tmp122, tmp120, tmp85)
    tmp124 = tl.where(tmp121, tmp72, tmp123)
    tmp125 = tmp72 - tmp66
    tmp126 = tmp125 * tmp68
    tmp127 = tmp126 * tmp126
    tmp128 = tmp54 - tmp59
    tmp129 = tmp36 * tmp128
    tmp130 = tmp127 - tmp129
    tmp131 = tmp128 * tmp80
    tmp132 = triton_helpers.maximum(tmp130, tmp72)
    tmp133 = tl.sqrt_rn(tmp132)
    tmp134 = tmp126 + tmp133
    tmp135 = triton_helpers.maximum(tmp134, tmp77)
    tmp136 = (tmp131 / tmp135)
    tmp137 = tmp128 >= tmp72
    tmp138 = tmp130 >= tmp72
    tmp139 = tmp134 > tmp77
    tmp140 = tmp138 & tmp139
    tmp141 = tl.where(tmp140, tmp136, tmp85)
    tmp142 = tl.where(tmp137, tmp72, tmp141)
    tmp143 = tmp72 - tmp108
    tmp144 = tmp143 * tmp68
    tmp145 = tmp144 * tmp144
    tmp146 = tmp98 - tmp102
    tmp147 = tmp94 * tmp146
    tmp148 = tmp145 - tmp147
    tmp149 = tmp146 * tmp80
    tmp150 = triton_helpers.maximum(tmp148, tmp72)
    tmp151 = tl.sqrt_rn(tmp150)
    tmp152 = tmp144 + tmp151
    tmp153 = triton_helpers.maximum(tmp152, tmp77)
    tmp154 = (tmp149 / tmp153)
    tmp155 = tmp146 >= tmp72
    tmp156 = tmp148 >= tmp72
    tmp157 = tmp152 > tmp77
    tmp158 = tmp156 & tmp157
    tmp159 = tl.where(tmp158, tmp154, tmp85)
    tmp160 = tl.where(tmp155, tmp72, tmp159)
    tmp161 = -tmp69
    tmp162 = tmp161 * tmp161
    tmp163 = tmp162 - tmp61
    tmp164 = tmp163 >= tmp72
    tmp165 = triton_helpers.maximum(tmp163, tmp72)
    tmp166 = tl.sqrt_rn(tmp165)
    tmp167 = tmp161 + tmp166
    tmp168 = tmp167 > tmp77
    tmp169 = triton_helpers.maximum(tmp167, tmp77)
    tmp170 = (tmp81 / tmp169)
    tmp171 = tmp164 & tmp168
    tmp172 = tl.where(tmp171, tmp170, tmp85)
    tmp173 = tl.where(tmp83, tmp72, tmp172)
    tmp174 = -tmp110
    tmp175 = tmp174 * tmp174
    tmp176 = tmp175 - tmp104
    tmp177 = tmp176 >= tmp72
    tmp178 = triton_helpers.maximum(tmp176, tmp72)
    tmp179 = tl.sqrt_rn(tmp178)
    tmp180 = tmp174 + tmp179
    tmp181 = tmp180 > tmp77
    tmp182 = triton_helpers.maximum(tmp180, tmp77)
    tmp183 = (tmp119 / tmp182)
    tmp184 = tmp177 & tmp181
    tmp185 = tl.where(tmp184, tmp183, tmp85)
    tmp186 = tl.where(tmp121, tmp72, tmp185)
    tmp187 = -tmp126
    tmp188 = tmp187 * tmp187
    tmp189 = tmp188 - tmp129
    tmp190 = triton_helpers.maximum(tmp189, tmp72)
    tmp191 = tl.sqrt_rn(tmp190)
    tmp192 = tmp187 + tmp191
    tmp193 = triton_helpers.maximum(tmp192, tmp77)
    tmp194 = (tmp131 / tmp193)
    tmp195 = tmp189 >= tmp72
    tmp196 = tmp192 > tmp77
    tmp197 = tmp195 & tmp196
    tmp198 = tl.where(tmp197, tmp194, tmp85)
    tmp199 = tl.where(tmp137, tmp72, tmp198)
    tmp200 = -tmp144
    tmp201 = tmp200 * tmp200
    tmp202 = tmp201 - tmp147
    tmp203 = triton_helpers.maximum(tmp202, tmp72)
    tmp204 = tl.sqrt_rn(tmp203)
    tmp205 = tmp200 + tmp204
    tmp206 = triton_helpers.maximum(tmp205, tmp77)
    tmp207 = (tmp149 / tmp206)
    tmp208 = tmp202 >= tmp72
    tmp209 = tmp205 > tmp77
    tmp210 = tmp208 & tmp209
    tmp211 = tl.where(tmp210, tmp207, tmp85)
    tmp212 = tl.where(tmp155, tmp72, tmp211)
    tmp213 = 0.001
    tmp214 = triton_helpers.maximum(tmp37, tmp213)
    tmp215 = tl.full([1], 1, tl.int32)
    tmp216 = (tmp215 / tmp214)
    tmp217 = 7.319
    tmp218 = tmp216 * tmp217
    tmp219 = triton_helpers.minimum(tmp218, tmp88)
    tmp220 = 7.0
    tmp221 = tmp219 * tmp220
    tmp222 = tmp221 - tmp46
    tmp223 = triton_helpers.minimum(tmp222, tmp87)
    tmp224 = 22.72870945945946
    tmp225 = triton_helpers.minimum(tmp223, tmp224)
    tmp226 = triton_helpers.minimum(tmp225, tmp124)
    tmp227 = triton_helpers.minimum(tmp221, tmp142)
    tmp228 = triton_helpers.minimum(tmp227, tmp224)
    tmp229 = triton_helpers.minimum(tmp228, tmp160)
    tmp230 = triton_helpers.minimum(tmp226, tmp229)
    tmp231 = -5.0
    tmp232 = tmp231 - tmp46
    tmp233 = -tmp173
    tmp234 = triton_helpers.maximum(tmp232, tmp233)
    tmp235 = -tmp186
    tmp236 = triton_helpers.maximum(tmp234, tmp235)
    tmp237 = -21.045101351351356
    tmp238 = triton_helpers.maximum(tmp236, tmp237)
    tmp239 = -tmp199
    tmp240 = triton_helpers.maximum(tmp231, tmp239)
    tmp241 = -tmp212
    tmp242 = triton_helpers.maximum(tmp240, tmp241)
    tmp243 = triton_helpers.maximum(tmp242, tmp237)
    tmp244 = triton_helpers.maximum(tmp238, tmp243)
    tmp245 = -tmp244
    tmp246 = triton_helpers.maximum(tmp245, tmp72)
    tmp247 = triton_helpers.maximum(tmp230, tmp72)
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
    tl.store(in_out_ptr5 + (x2), tmp246, xmask)
    tl.store(in_out_ptr1 + (x2), tmp247, xmask)
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
        assert_size_stride(arg0_1, (48, 25), (25, 1))
        assert_size_stride(arg1_1, (48, 1), (1, 1))
        assert_size_stride(arg2_1, (48, 25), (25, 1))
        assert_size_stride(arg3_1, (48, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((48, 25), (25, 1), torch.float32)
            buf4 = buf1; del buf1  # reuse
            buf5 = buf4; del buf4  # reuse
            buf31 = empty_strided_cuda((48, 25), (25, 1), torch.float32)
            buf34 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [curve, mul, grip, mul_4, mul_5, square_1, cc, expand_as, ge_1, zeros_like, square_2, mul_3, square, aa, mul_6, mul_7, disc, ge, bb, clamp_min, sqrt, den, gt, and_, mul_8, clamp_min_1, root, full_like_2, root_1, where_1, curve_1, grip_1, mul_10, mul_11, square_4, cc_1, expand_as_1, ge_3, zeros_like_1, square_5, mul_9, square_3, aa_1, mul_12, mul_13, disc_1, ge_2, bb_1, clamp_min_2, sqrt_1, den_1, gt_1, and__1, mul_14, clamp_min_3, root_2, full_like_4, root_3, where_3, curve_2, clamp_min_4, curve_3, add_4, middle, abs_1, truediv_2, tanh, mul_16, square_6, mul_17, add_5, clamp_min_5, truediv_3, clamp, mul_18, sustainable, low, low_1, high_1, add_6, middle_1, abs_2, truediv_4, tanh_1, mul_20, square_7, mul_21, add_7, clamp_min_6, truediv_5, clamp_1, mul_22, sustainable_1, low_2, high_2, add_8, middle_2, abs_3, truediv_6, tanh_2, mul_24, square_8, mul_25, add_9, clamp_min_7, truediv_7, clamp_2, mul_26, sustainable_2, low_3, high_3, add_10, middle_3, abs_4, truediv_8, tanh_3, mul_28, square_9, mul_29, add_11, clamp_min_8, truediv_9, clamp_3, mul_30, sustainable_3, low_4, high_4, add_12, middle_4, abs_5, truediv_10, tanh_4, mul_32, square_10, mul_33, add_13, clamp_min_9, truediv_11, clamp_4, mul_34, sustainable_4, low_5, high_5, add_14, middle_5, abs_6, truediv_12, tanh_5, mul_36, square_11, mul_37, add_15, clamp_min_10, truediv_13, clamp_5, mul_38, sustainable_5, low_6, high_6, add_16, middle_6, abs_7, truediv_14, tanh_6, mul_40, square_12, mul_41, add_17, clamp_min_11, truediv_15, clamp_6, mul_42, sustainable_6, low_7, high_7, add_18, middle_7, abs_8, truediv_16, tanh_7, mul_44, square_13, mul_45, add_19, clamp_min_12, truediv_17, clamp_7, mul_46, sustainable_7, low_8, high_8, add_20, middle_8, abs_9, truediv_18, tanh_8, mul_48, square_14, mul_49, add_21, clamp_min_13, truediv_19, clamp_8, mul_50, sustainable_8, low_9, high_9, add_22, middle_9, abs_10, truediv_20, tanh_9, mul_52, square_15, mul_53, add_23, clamp_min_14, truediv_21, clamp_9, mul_54, sustainable_9, low_10, high_10, add_24, middle_10, abs_11, truediv_22, tanh_10, mul_56, square_16, mul_57, add_25, clamp_min_15, truediv_23, clamp_10, mul_58, sustainable_10, low_11, high_11, add_26, middle_11, abs_12, truediv_24, tanh_11, mul_60, square_17, mul_61, add_27, clamp_min_16, truediv_25, clamp_11, mul_62, sustainable_11, high_12], Original ATen: [aten.full_like, aten.mul, aten.pow, aten.rsub, aten.expand, aten.ge, aten.zeros_like, aten.add, aten.sub, aten.clamp_min, aten.sqrt, aten.gt, aten.bitwise_and, aten.div, aten.where, aten.minimum, aten.abs, aten.tanh, aten.reciprocal, aten.clamp, aten.le]
            stream0 = get_raw_stream(0)
            triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_expand_full_like_ge_gt_le_minimum_mul_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0.run(buf5, buf34, arg2_1, arg1_1, arg0_1, 1200, stream=stream0)
            buf35 = empty_strided_cuda((48, 24), (24, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem, mul_65, add_4, cap_1, square_18, mul_63, add_28, mul_64, delta, getitem_1, getitem_2, sub_4, abs_13, clamp_min_17, slew_cap], Original ATen: [aten.unsqueeze, aten.mul, aten.add, aten.minimum, aten.pow, aten.atan, aten.slice, aten.sub, aten.abs, aten.clamp_min, aten.div]
            stream0 = get_raw_stream(0)
            triton_poi_fused_abs_add_atan_clamp_min_div_minimum_mul_pow_slice_sub_unsqueeze_1.run(arg3_1, arg0_1, buf5, buf34, arg2_1, buf35, 1152, stream=stream0)
            del arg3_1
            buf36 = buf34; del buf34  # reuse
            buf41 = empty_strided_cuda((48, 25), (25, 1), torch.float32)
            buf42 = buf41; del buf41  # reuse
            buf43 = buf42; del buf42  # reuse
            buf63 = empty_strided_cuda((48, 25), (25, 1), torch.float32)
            buf64 = buf63; del buf63  # reuse
            buf65 = buf64; del buf64  # reuse
            buf57 = buf43; del buf43  # reuse
            buf79 = buf65; del buf65  # reuse
            buf58 = buf57; del buf57  # reuse
            # Topologically Sorted Source Nodes: [add_4, cap_1, getitem_3, getitem_4, getitem_5, minimum_4, getitem_6, cat, cap_2, abs_15, clamp_min_18, truediv_28, clamp_12, power, abs_14, truediv_27, tanh_12, mul_66, square_19, mul_67, r, hi, mul_77, square_22, square_23, mul_78, mul_79, square_24, add_30, grip_2, mul_80, mul_81, square_25, cc_2, ge_7, zeros_like_5, mul_72, square_21, mul_73, mul_74, mul_75, sub_8, bb_2, square_27, mul_71, square_20, aa_2, mul_85, mul_86, disc_3, ge_6, clamp_min_21, sqrt_4, den_3, gt_4, and__3, mul_87, clamp_min_22, root_6, full_like_6, root_7, where_31, hi_1, hi_2, mul_94, square_30, square_31, mul_95, mul_96, square_32, add_33, grip_3, mul_97, mul_98, square_33, cc_3, ge_11, zeros_like_7, mul_89, square_29, mul_90, mul_91, mul_92, sub_13, bb_3, square_35, mul_88, square_28, aa_3, mul_102, mul_103, disc_5, ge_10, clamp_min_25, sqrt_6, den_5, gt_7, and__5, mul_104, clamp_min_26, root_10, full_like_8, root_11, where_35, hi_3, abs_16, clamp_min_27, truediv_33, clamp_15, hi_4, square_39, mul_115, mul_116, add_36, grip_4, mul_117, mul_118, square_41, cc_4, ge_15, zeros_like_11, mul_109, square_37, mul_110, mul_111, mul_112, sub_20, bb_4, square_43, mul_108, square_36, aa_4, mul_122, mul_123, disc_7, ge_14, clamp_min_30, sqrt_8, den_7, gt_11, and__7, mul_124, clamp_min_31, root_14, full_like_10, root_15, where_39, hi_5, hi_6, square_47, mul_132, mul_133, add_39, grip_5, mul_134, mul_135, square_49, cc_5, ge_19, zeros_like_13, mul_126, square_45, mul_127, mul_128, mul_129, sub_25, bb_5, square_51, mul_125, square_44, aa_5, mul_139, mul_140, disc_9, ge_18, clamp_min_34, sqrt_10, den_9, gt_14, and__9, mul_141, clamp_min_35, root_18, full_like_12, root_19, where_43, hi_7, minimum_10, acc, lo, ge_5, zeros_like_4, neg, square_26, mul_82, mul_83, disc_2, ge_4, clamp_min_19, sqrt_3, den_2, gt_3, and__2, mul_84, clamp_min_20, root_4, full_like_5, root_5, where_29, neg_1, lo_1, ge_9, zeros_like_6, neg_2, square_34, mul_99, mul_100, disc_4, ge_8, clamp_min_23, sqrt_5, den_4, gt_6, and__4, mul_101, clamp_min_24, root_8, full_like_7, root_9, where_33, neg_3, lo_2, lo_3, lo_4, ge_13, zeros_like_10, neg_4, square_42, mul_119, mul_120, disc_6, ge_12, clamp_min_28, sqrt_7, den_6, gt_10, and__6, mul_121, clamp_min_29, root_12, full_like_9, root_13, where_37, neg_5, lo_5, ge_17, zeros_like_12, neg_6, square_50, mul_136, mul_137, disc_8, ge_16, clamp_min_32, sqrt_9, den_8, gt_13, and__8, mul_138, clamp_min_33, root_16, full_like_11, root_17, where_41, neg_7, lo_6, lo_7, maximum_4, neg_8, brk], Original ATen: [aten.add, aten.minimum, aten.slice, aten.cat, aten.abs, aten.clamp_min, aten.reciprocal, aten.mul, aten.clamp, aten.div, aten.tanh, aten.pow, aten.sub, aten.ge, aten.zeros_like, aten.rsub, aten.sqrt, aten.gt, aten.bitwise_and, aten.full_like, aten.where, aten.neg, aten.maximum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_abs_add_bitwise_and_cat_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_slice_sqrt_sub_tanh_where_zeros_like_2.run(buf36, buf58, buf79, arg0_1, buf5, buf35, arg1_1, arg2_1, 1200, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            del buf35
        return (buf36, buf5, buf58, buf79, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((48, 25), (25, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((48, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((48, 25), (25, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
