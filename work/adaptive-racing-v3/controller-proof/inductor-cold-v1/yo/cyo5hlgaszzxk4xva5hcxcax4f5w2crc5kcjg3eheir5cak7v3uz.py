# AOT ID: ['9_inference']
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


# kernel path: /home/shchon11/F1tenth/F1tenth_E2E/work/adaptive-racing-v3/controller-proof/inductor-cold-v1/xn/cxn2csjmysgcpjedaqrwszp6h7rphoyeo63i5yu2xktnb443xcur.py
# Topologically Sorted Source Nodes: [square_43, square_41, square_42, sub_26, mul_91, want, square, mul, add, sqrt, minimum, fastest, abs_3, clamp_min_4, truediv_4, clamp, power, abs_2, truediv_3, tanh_1, mul_15, square_8, mul_16, r_1, hi, mul_26, square_11, square_12, mul_27, cap, grip, mul_5, mul_6, mul_7, square_2, abs_1, truediv, tanh, mul_1, square_1, mul_2, r, mul_8, square_3, sub, residual, sqrt_1, square_4, mul_9, clamp_min_1, truediv_1, cap_1, grip_1, mul_10, mul_11, mul_12, square_5, mul_13, square_6, sub_1, residual_1, sqrt_2, square_7, mul_14, clamp_min_3, truediv_2, cap_2, safe_k, mul_28, square_13, add_3, grip_2, mul_29, mul_30, square_14, cc, ge_3, zeros_like_2, mul_21, square_10, mul_22, mul_23, mul_24, sub_5, bb, square_16, mul_20, square_9, aa, mul_34, mul_35, disc_1, ge_2, clamp_min_7, sqrt_4, den_1, gt_2, and__1, mul_36, clamp_min_8, root_2, full_like_2, root_3, where_3, hi_1, hi_2, mul_43, square_19, square_20, mul_44, mul_45, square_21, add_6, grip_3, mul_46, mul_47, square_22, cc_1, ge_7, zeros_like_4, mul_38, square_18, mul_39, mul_40, mul_41, sub_10, bb_1, square_24, mul_37, square_17, aa_1, mul_51, mul_52, disc_3, ge_6, clamp_min_11, sqrt_6, den_3, gt_5, and__3, mul_53, clamp_min_12, root_6, full_like_4, root_7, where_7, hi_3, abs_4, clamp_min_13, truediv_9, clamp_3, hi_4, square_28, mul_64, mul_65, add_9, grip_4, mul_66, mul_67, square_30, cc_2, ge_11, zeros_like_8, mul_58, square_26, mul_59, mul_60, mul_61, sub_17, bb_2, square_32, mul_57, square_25, aa_2, mul_71, mul_72, disc_5, ge_10, clamp_min_16, sqrt_8, den_5, gt_9, and__5, mul_73, clamp_min_17, root_10, full_like_6, root_11, where_11, hi_5, hi_6, square_36, mul_81, mul_82, add_12, grip_5, mul_83, mul_84, square_38, cc_3, ge_15, zeros_like_10, mul_75, square_34, mul_76, mul_77, mul_78, sub_22, bb_3, square_40, mul_74, square_33, aa_3, mul_88, mul_89, disc_7, ge_14, clamp_min_20, sqrt_10, den_7, gt_12, and__7, mul_90, clamp_min_21, root_14, full_like_8, root_15, where_15, hi_7, upper, minimum_9, lo, ge_1, zeros_like_1, neg, square_15, mul_31, mul_32, disc, ge, clamp_min_5, sqrt_3, den, gt_1, and_, mul_33, clamp_min_6, root, full_like_1, root_1, where_1, neg_1, lo_1, ge_5, zeros_like_3, neg_2, square_23, mul_48, mul_49, disc_2, ge_4, clamp_min_9, sqrt_5, den_2, gt_4, and__2, mul_50, clamp_min_10, root_4, full_like_3, root_5, where_5, neg_3, lo_2, lo_3, lo_4, ge_9, zeros_like_7, neg_4, square_31, mul_68, mul_69, disc_4, ge_8, clamp_min_14, sqrt_7, den_4, gt_8, and__4, mul_70, clamp_min_15, root_8, full_like_5, root_9, where_9, neg_5, lo_5, ge_13, zeros_like_9, neg_6, square_39, mul_85, mul_86, disc_6, ge_12, clamp_min_18, sqrt_9, den_6, gt_11, and__6, mul_87, clamp_min_19, root_12, full_like_7, root_13, where_13, neg_7, lo_6, lo_7, lower, ax, mul_92, mul_93, add_15, clamp_min_22, sqrt_11], Original ATen: [aten.pow, aten.sub, aten.mul, aten.div, aten.add, aten.sqrt, aten.minimum, aten.maximum, aten.abs, aten.clamp_min, aten.reciprocal, aten.clamp, aten.tanh, aten.full_like, aten.ge, aten.zeros_like, aten.rsub, aten.gt, aten.bitwise_and, aten.where, aten.neg]
# Source node to ATen node mapping:
#   aa => sub_4
#   aa_1 => sub_9
#   aa_2 => sub_16
#   aa_3 => sub_21
#   abs_1 => abs_1
#   abs_2 => abs_2
#   abs_3 => abs_3
#   abs_4 => abs_4
#   add => add
#   add_12 => pow_38
#   add_15 => add_15
#   add_3 => add_3
#   add_6 => add_6
#   add_9 => pow_30
#   and_ => bitwise_and
#   and__1 => bitwise_and_1
#   and__2 => bitwise_and_2
#   and__3 => bitwise_and_3
#   and__4 => bitwise_and_4
#   and__5 => bitwise_and_5
#   and__6 => bitwise_and_6
#   and__7 => bitwise_and_7
#   ax => maximum_6
#   bb => mul_26
#   bb_1 => mul_43
#   bb_2 => mul_64
#   bb_3 => mul_81
#   cap => full_default
#   cap_1 => minimum_1
#   cap_2 => minimum_2
#   cc => sub_6
#   cc_1 => sub_11
#   cc_2 => sub_18
#   cc_3 => sub_23
#   clamp => clamp_max
#   clamp_3 => clamp_max_2
#   clamp_min_1 => clamp_min_1
#   clamp_min_10 => clamp_min_10
#   clamp_min_11 => clamp_min_11
#   clamp_min_12 => clamp_min_12
#   clamp_min_13 => clamp_min_14
#   clamp_min_14 => clamp_min_15
#   clamp_min_15 => clamp_min_16
#   clamp_min_16 => clamp_min_17
#   clamp_min_17 => clamp_min_18
#   clamp_min_18 => clamp_min_19
#   clamp_min_19 => clamp_min_20
#   clamp_min_20 => clamp_min_21
#   clamp_min_21 => clamp_min_22
#   clamp_min_22 => clamp_min_24
#   clamp_min_3 => clamp_min_3
#   clamp_min_4 => clamp_min_4
#   clamp_min_5 => clamp_min_5
#   clamp_min_6 => clamp_min_6
#   clamp_min_7 => clamp_min_7
#   clamp_min_8 => clamp_min_8
#   clamp_min_9 => clamp_min_9
#   den => add_4
#   den_1 => add_5
#   den_2 => add_7
#   den_3 => add_8
#   den_4 => add_10
#   den_5 => add_11
#   den_6 => add_13
#   den_7 => add_14
#   disc => sub_7
#   disc_1 => sub_8
#   disc_2 => sub_12
#   disc_3 => sub_13
#   disc_4 => sub_19
#   disc_5 => sub_20
#   disc_6 => sub_24
#   disc_7 => sub_25
#   fastest => maximum
#   full_like_1 => full_default_1
#   full_like_2 => full_default_3
#   full_like_3 => full_default_5
#   full_like_4 => full_default_7
#   full_like_5 => full_default_13
#   full_like_6 => full_default_15
#   full_like_7 => full_default_19
#   full_like_8 => full_default_21
#   ge => ge
#   ge_1 => ge_1
#   ge_10 => ge_10
#   ge_11 => ge_11
#   ge_12 => ge_12
#   ge_13 => ge_13
#   ge_14 => ge_14
#   ge_15 => ge_15
#   ge_2 => ge_2
#   ge_3 => ge_3
#   ge_4 => ge_4
#   ge_5 => ge_5
#   ge_6 => ge_6
#   ge_7 => ge_7
#   ge_8 => ge_8
#   ge_9 => ge_9
#   grip => mul_3
#   grip_1 => mul_4
#   grip_2 => mul_19
#   grip_3 => mul_20
#   grip_4 => mul_57
#   grip_5 => mul_58
#   gt_1 => gt_1
#   gt_11 => gt_11
#   gt_12 => gt_12
#   gt_2 => gt_2
#   gt_4 => gt_4
#   gt_5 => gt_5
#   gt_8 => gt_8
#   gt_9 => gt_9
#   hi => sub_3
#   hi_1 => minimum_4
#   hi_2 => clamp_max_1
#   hi_3 => minimum_5
#   hi_4 => mul_56
#   hi_5 => minimum_6
#   hi_6 => clamp_max_3
#   hi_7 => minimum_7
#   lo => sub_2
#   lo_1 => maximum_1
#   lo_2 => maximum_2
#   lo_3 => clamp_min_13
#   lo_4 => full_default_10
#   lo_5 => maximum_3
#   lo_6 => maximum_4
#   lo_7 => clamp_min_23
#   lower => maximum_5
#   minimum => minimum
#   minimum_9 => minimum_9
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_2 => mul_2
#   mul_20 => mul_21
#   mul_21 => mul_22
#   mul_22 => mul_23
#   mul_23 => mul_24
#   mul_24 => mul_25
#   mul_26 => mul_27
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
#   mul_43 => mul_44
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
#   mul_57 => mul_59
#   mul_58 => full_default_11
#   mul_59 => mul_61
#   mul_6 => mul_6
#   mul_60 => mul_62
#   mul_61 => mul_63
#   mul_64 => mul_66
#   mul_65 => mul_67
#   mul_66 => mul_68
#   mul_67 => mul_69
#   mul_68 => mul_70
#   mul_69 => mul_71
#   mul_7 => mul_7
#   mul_70 => mul_72
#   mul_71 => mul_73
#   mul_72 => mul_74
#   mul_73 => mul_75
#   mul_74 => mul_76
#   mul_75 => full_default_17
#   mul_76 => mul_78
#   mul_77 => mul_79
#   mul_78 => mul_80
#   mul_8 => mul_8
#   mul_81 => mul_83
#   mul_82 => mul_84
#   mul_83 => mul_85
#   mul_84 => mul_86
#   mul_85 => mul_87
#   mul_86 => mul_88
#   mul_87 => mul_89
#   mul_88 => mul_90
#   mul_89 => mul_91
#   mul_9 => mul_9
#   mul_90 => mul_92
#   mul_91 => mul_93
#   mul_92 => mul_94
#   mul_93 => mul_95
#   neg => neg
#   neg_1 => neg_1
#   neg_2 => neg_2
#   neg_3 => neg_3
#   neg_4 => neg_4
#   neg_5 => neg_5
#   neg_6 => neg_6
#   neg_7 => neg_7
#   power => mul_18
#   r => add_1
#   r_1 => add_2
#   residual => clamp_min
#   residual_1 => clamp_min_2
#   root => div_4
#   root_1 => where
#   root_10 => div_9
#   root_11 => where_10
#   root_12 => div_10
#   root_13 => where_12
#   root_14 => div_11
#   root_15 => where_14
#   root_2 => div_5
#   root_3 => where_2
#   root_4 => div_6
#   root_5 => where_4
#   root_6 => div_7
#   root_7 => where_6
#   root_8 => div_8
#   root_9 => where_8
#   safe_k => minimum_3
#   sqrt => sqrt
#   sqrt_1 => sqrt_1
#   sqrt_10 => sqrt_10
#   sqrt_11 => sqrt_11
#   sqrt_2 => sqrt_2
#   sqrt_3 => sqrt_3
#   sqrt_4 => sqrt_4
#   sqrt_5 => sqrt_5
#   sqrt_6 => sqrt_6
#   sqrt_7 => sqrt_7
#   sqrt_8 => sqrt_8
#   sqrt_9 => sqrt_9
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
#   square_25 => pow_26
#   square_26 => pow_27
#   square_28 => pow_29
#   square_3 => pow_4
#   square_30 => pow_31
#   square_31 => pow_32
#   square_32 => pow_33
#   square_33 => pow_34
#   square_34 => pow_35
#   square_36 => pow_37
#   square_38 => pow_39
#   square_39 => pow_40
#   square_4 => pow_5
#   square_40 => pow_41
#   square_41 => pow_42
#   square_42 => pow_43
#   square_43 => pow_44
#   square_5 => pow_6
#   square_6 => pow_7
#   square_7 => pow_8
#   square_8 => pow_9
#   square_9 => pow_10
#   sub => sub
#   sub_1 => sub_1
#   sub_10 => sub_10
#   sub_17 => sub_17
#   sub_22 => sub_22
#   sub_26 => sub_26
#   sub_5 => sub_5
#   tanh => tanh
#   tanh_1 => tanh_1
#   truediv => div
#   truediv_1 => div_1
#   truediv_2 => div_2
#   truediv_3 => div_3
#   truediv_4 => mul_17, reciprocal
#   truediv_9 => mul_55, reciprocal_1
#   upper => minimum_8
#   want => div_12
#   where_1 => where_1
#   where_11 => where_11
#   where_13 => where_13
#   where_15 => where_15
#   where_3 => where_3
#   where_5 => where_5
#   where_7 => where_7
#   where_9 => where_9
#   zeros_like_1 => full_default_2
#   zeros_like_10 => full_default_22
#   zeros_like_2 => full_default_4
#   zeros_like_3 => full_default_6
#   zeros_like_4 => full_default_8
#   zeros_like_7 => full_default_14
#   zeros_like_8 => full_default_16
#   zeros_like_9 => full_default_20
# Graph fragment:
#   %arg0_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "f32[1][25]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg1_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg3_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %minimum_3 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=minimum_3]
#   %sub_6 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=sub_6]
#   %mul_26 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=mul_26]
#   %sub_11 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=sub_11]
#   %mul_43 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=mul_43]
#   %sub_18 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=sub_18]
#   %sub_23 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=sub_23]
#   %minimum_8 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=minimum_8]
#   %maximum_5 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=maximum_5]
#   %pow_44 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg0_1, 2), kwargs = {})
#   %pow_42 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg2_1, 2), kwargs = {})
#   %pow_43 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg0_1, 2), kwargs = {})
#   %sub_26 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_42, %pow_43), kwargs = {})
#   %mul_93 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 2), kwargs = {})
#   %div_12 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_26, %mul_93), kwargs = {})
#   %pow_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg0_1, 2), kwargs = {})
#   %mul : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 14.0), kwargs = {})
#   %add : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %mul), kwargs = {})
#   %sqrt : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %minimum : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%arg2_1, %sqrt), kwargs = {})
#   %maximum : Tensor "f32[1][1]cuda:0"[num_users=12] = call_function[target=torch.ops.aten.maximum.default](args = (%arg0_1, %minimum), kwargs = {})
#   %abs_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%maximum,), kwargs = {})
#   %clamp_min_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%abs_3, 0.001), kwargs = {})
#   %reciprocal : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_4,), kwargs = {})
#   %mul_17 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 7.319), kwargs = {})
#   %clamp_max : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_17, 1), kwargs = {})
#   %mul_18 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max, 7.0), kwargs = {})
#   %abs_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%maximum,), kwargs = {})
#   %div_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_2, 0.05), kwargs = {})
#   %tanh_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_3,), kwargs = {})
#   %mul_15 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_1, 0.1), kwargs = {})
#   %pow_9 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%maximum, 2), kwargs = {})
#   %mul_16 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_9, 0.01), kwargs = {})
#   %add_2 : Tensor "f32[1][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %mul_16), kwargs = {})
#   %sub_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_18, %add_2), kwargs = {})
#   %mul_27 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 0.5), kwargs = {})
#   %pow_12 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_27, 2), kwargs = {})
#   %pow_13 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%maximum, 2), kwargs = {})
#   %mul_28 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_13, 0.5192307692307692), kwargs = {})
#   %full_default : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, 0.92), kwargs = {})
#   %mul_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, 0.85), kwargs = {})
#   %mul_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, 9.81), kwargs = {})
#   %mul_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, 0.5192307692307692), kwargs = {})
#   %pow_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_7, 2), kwargs = {})
#   %abs_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%maximum,), kwargs = {})
#   %div : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%abs_1, 0.05), kwargs = {})
#   %tanh : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 0.1), kwargs = {})
#   %pow_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%maximum, 2), kwargs = {})
#   %mul_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_2, 0.01), kwargs = {})
#   %add_1 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %mul_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.5), kwargs = {})
#   %pow_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_8, 2), kwargs = {})
#   %sub : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_3, %pow_4), kwargs = {})
#   %clamp_min : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0), kwargs = {})
#   %sqrt_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min,), kwargs = {})
#   %pow_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%maximum, 2), kwargs = {})
#   %mul_9 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_5, 0.5192307692307692), kwargs = {})
#   %clamp_min_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_9, 1e-06), kwargs = {})
#   %div_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sqrt_1, %clamp_min_1), kwargs = {})
#   %minimum_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %div_1), kwargs = {})
#   %mul_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, 1.0), kwargs = {})
#   %mul_10 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, 0.85), kwargs = {})
#   %mul_11 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, 9.81), kwargs = {})
#   %mul_12 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, 0.4807692307692308), kwargs = {})
#   %pow_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_12, 2), kwargs = {})
#   %mul_13 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.5), kwargs = {})
#   %pow_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_13, 2), kwargs = {})
#   %sub_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_6, %pow_7), kwargs = {})
#   %clamp_min_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_1, 0), kwargs = {})
#   %sqrt_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_2,), kwargs = {})
#   %pow_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%maximum, 2), kwargs = {})
#   %mul_14 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_8, 0.4807692307692308), kwargs = {})
#   %clamp_min_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_14, 1e-06), kwargs = {})
#   %div_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sqrt_2, %clamp_min_3), kwargs = {})
#   %minimum_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%minimum_1, %div_2), kwargs = {})
#   %minimum_3 : Tensor "f32[1][1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.minimum.default](args = (%arg4_1, %minimum_2), kwargs = {})
#   %mul_29 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %minimum_3), kwargs = {})
#   %pow_14 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_29, 2), kwargs = {})
#   %add_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_12, %pow_14), kwargs = {})
#   %mul_19 : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, 0.92), kwargs = {})
#   %mul_30 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, 9.81), kwargs = {})
#   %mul_31 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, 0.5192307692307692), kwargs = {})
#   %pow_15 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_31, 2), kwargs = {})
#   %sub_6 : Tensor "f32[1][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %pow_15), kwargs = {})
#   %ge_3 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_6, 0), kwargs = {})
#   %full_default_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_22 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 0.25), kwargs = {})
#   %pow_11 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_19, 2), kwargs = {})
#   %mul_23 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_11, 9.81), kwargs = {})
#   %mul_24 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, 0.5192307692307692), kwargs = {})
#   %mul_25 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, -0.22410660205935795), kwargs = {})
#   %sub_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_22, %mul_25), kwargs = {})
#   %mul_26 : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, 2), kwargs = {})
#   %pow_17 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_26, 2), kwargs = {})
#   %mul_21 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, -0.22410660205935795), kwargs = {})
#   %pow_10 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_21, 2), kwargs = {})
#   %sub_4 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_10), kwargs = {})
#   %mul_35 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, 4), kwargs = {})
#   %mul_36 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, %sub_6), kwargs = {})
#   %sub_8 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_17, %mul_36), kwargs = {})
#   %ge_2 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_8, 0), kwargs = {})
#   %clamp_min_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_8, 0), kwargs = {})
#   %sqrt_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_7,), kwargs = {})
#   %add_5 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %sqrt_4), kwargs = {})
#   %gt_2 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_5, 1e-12), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_2, %gt_2), kwargs = {})
#   %mul_37 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, -2), kwargs = {})
#   %clamp_min_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_5, 1e-12), kwargs = {})
#   %div_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_37, %clamp_min_8), kwargs = {})
#   %full_default_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_1, %div_5, %full_default_3), kwargs = {})
#   %where_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_3, %full_default_4, %where_2), kwargs = {})
#   %minimum_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%sub_3, %where_3), kwargs = {})
#   %clamp_max_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%minimum_4, 22.72870945945946), kwargs = {})
#   %mul_44 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 0.5), kwargs = {})
#   %pow_20 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_44, 2), kwargs = {})
#   %pow_21 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%maximum, 2), kwargs = {})
#   %mul_45 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_21, 0.4807692307692308), kwargs = {})
#   %mul_46 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_45, %minimum_3), kwargs = {})
#   %pow_22 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_46, 2), kwargs = {})
#   %add_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_20, %pow_22), kwargs = {})
#   %mul_20 : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, 1.0), kwargs = {})
#   %mul_47 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, 9.81), kwargs = {})
#   %mul_48 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_47, 0.4807692307692308), kwargs = {})
#   %pow_23 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_48, 2), kwargs = {})
#   %sub_11 : Tensor "f32[1][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %pow_23), kwargs = {})
#   %ge_7 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_11, 0), kwargs = {})
#   %full_default_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_39 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 0.25), kwargs = {})
#   %pow_19 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_20, 2), kwargs = {})
#   %mul_40 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_19, 9.81), kwargs = {})
#   %mul_41 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, 0.4807692307692308), kwargs = {})
#   %mul_42 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, 0.22410660205935795), kwargs = {})
#   %sub_10 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_39, %mul_42), kwargs = {})
#   %mul_43 : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, 2), kwargs = {})
#   %pow_25 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_43, 2), kwargs = {})
#   %mul_38 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, 0.22410660205935795), kwargs = {})
#   %pow_18 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_38, 2), kwargs = {})
#   %sub_9 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_18), kwargs = {})
#   %mul_52 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, 4), kwargs = {})
#   %mul_53 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %sub_11), kwargs = {})
#   %sub_13 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_25, %mul_53), kwargs = {})
#   %ge_6 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_13, 0), kwargs = {})
#   %clamp_min_11 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_13, 0), kwargs = {})
#   %sqrt_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_11,), kwargs = {})
#   %add_8 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %sqrt_6), kwargs = {})
#   %gt_5 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_8, 1e-12), kwargs = {})
#   %bitwise_and_3 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_6, %gt_5), kwargs = {})
#   %mul_54 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, -2), kwargs = {})
#   %clamp_min_12 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_8, 1e-12), kwargs = {})
#   %div_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_54, %clamp_min_12), kwargs = {})
#   %full_default_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_3, %div_7, %full_default_7), kwargs = {})
#   %where_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_7, %full_default_8, %where_6), kwargs = {})
#   %minimum_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%clamp_max_1, %where_7), kwargs = {})
#   %abs_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%maximum,), kwargs = {})
#   %clamp_min_14 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%abs_4, 0.001), kwargs = {})
#   %reciprocal_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_14,), kwargs = {})
#   %mul_55 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 7.319), kwargs = {})
#   %clamp_max_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%mul_55, 1), kwargs = {})
#   %mul_56 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_2, 7.0), kwargs = {})
#   %pow_29 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%maximum, 2), kwargs = {})
#   %mul_66 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_29, 0.5192307692307692), kwargs = {})
#   %mul_67 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_66, %minimum_3), kwargs = {})
#   %pow_30 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_67, 2), kwargs = {})
#   %mul_57 : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, 0.92), kwargs = {})
#   %mul_68 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_57, 9.81), kwargs = {})
#   %mul_69 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_68, 0.5192307692307692), kwargs = {})
#   %pow_31 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_69, 2), kwargs = {})
#   %sub_18 : Tensor "f32[1][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_30, %pow_31), kwargs = {})
#   %ge_11 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_18, 0), kwargs = {})
#   %full_default_16 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_11 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_27 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_57, 2), kwargs = {})
#   %mul_61 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_27, 9.81), kwargs = {})
#   %mul_62 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, 0.5192307692307692), kwargs = {})
#   %mul_63 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_62, -0.22410660205935795), kwargs = {})
#   %sub_17 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_11, %mul_63), kwargs = {})
#   %mul_64 : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, 2), kwargs = {})
#   %pow_33 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_64, 2), kwargs = {})
#   %mul_59 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_57, -0.22410660205935795), kwargs = {})
#   %pow_26 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_59, 2), kwargs = {})
#   %sub_16 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_26), kwargs = {})
#   %mul_73 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, 4), kwargs = {})
#   %mul_74 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %sub_18), kwargs = {})
#   %sub_20 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_33, %mul_74), kwargs = {})
#   %ge_10 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_20, 0), kwargs = {})
#   %clamp_min_17 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_20, 0), kwargs = {})
#   %sqrt_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_17,), kwargs = {})
#   %add_11 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_64, %sqrt_8), kwargs = {})
#   %gt_9 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_11, 1e-12), kwargs = {})
#   %bitwise_and_5 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_10, %gt_9), kwargs = {})
#   %mul_75 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, -2), kwargs = {})
#   %clamp_min_18 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_11, 1e-12), kwargs = {})
#   %div_9 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_75, %clamp_min_18), kwargs = {})
#   %full_default_15 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_10 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_5, %div_9, %full_default_15), kwargs = {})
#   %where_11 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_11, %full_default_16, %where_10), kwargs = {})
#   %minimum_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%mul_56, %where_11), kwargs = {})
#   %clamp_max_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%minimum_6, 22.72870945945946), kwargs = {})
#   %pow_37 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%maximum, 2), kwargs = {})
#   %mul_83 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_37, 0.4807692307692308), kwargs = {})
#   %mul_84 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_83, %minimum_3), kwargs = {})
#   %pow_38 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_84, 2), kwargs = {})
#   %mul_58 : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, 1.0), kwargs = {})
#   %mul_85 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, 9.81), kwargs = {})
#   %mul_86 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, 0.4807692307692308), kwargs = {})
#   %pow_39 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_86, 2), kwargs = {})
#   %sub_23 : Tensor "f32[1][1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_38, %pow_39), kwargs = {})
#   %ge_15 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_23, 0), kwargs = {})
#   %full_default_22 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_17 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_35 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_58, 2), kwargs = {})
#   %mul_78 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_35, 9.81), kwargs = {})
#   %mul_79 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_78, 0.4807692307692308), kwargs = {})
#   %mul_80 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, 0.22410660205935795), kwargs = {})
#   %sub_22 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_17, %mul_80), kwargs = {})
#   %mul_81 : Tensor "f32[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, 2), kwargs = {})
#   %pow_41 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_81, 2), kwargs = {})
#   %mul_76 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, 0.22410660205935795), kwargs = {})
#   %pow_34 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_76, 2), kwargs = {})
#   %sub_21 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (0.25, %pow_34), kwargs = {})
#   %mul_90 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, 4), kwargs = {})
#   %mul_91 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_90, %sub_23), kwargs = {})
#   %sub_25 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_41, %mul_91), kwargs = {})
#   %ge_14 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_25, 0), kwargs = {})
#   %clamp_min_21 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_25, 0), kwargs = {})
#   %sqrt_10 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_21,), kwargs = {})
#   %add_14 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_81, %sqrt_10), kwargs = {})
#   %gt_12 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_14, 1e-12), kwargs = {})
#   %bitwise_and_7 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_14, %gt_12), kwargs = {})
#   %mul_92 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, -2), kwargs = {})
#   %clamp_min_22 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_14, 1e-12), kwargs = {})
#   %div_11 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_92, %clamp_min_22), kwargs = {})
#   %full_default_21 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_14 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_7, %div_11, %full_default_21), kwargs = {})
#   %where_15 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_15, %full_default_22, %where_14), kwargs = {})
#   %minimum_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%clamp_max_3, %where_15), kwargs = {})
#   %minimum_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%minimum_5, %minimum_7), kwargs = {})
#   %minimum_9 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%div_12, %minimum_8), kwargs = {})
#   %sub_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (-5.0, %add_2), kwargs = {})
#   %ge_1 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_6, 0), kwargs = {})
#   %full_default_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_26,), kwargs = {})
#   %pow_16 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg, 2), kwargs = {})
#   %mul_32 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, 4), kwargs = {})
#   %mul_33 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %sub_6), kwargs = {})
#   %sub_7 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_16, %mul_33), kwargs = {})
#   %ge : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_7, 0), kwargs = {})
#   %clamp_min_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_7, 0), kwargs = {})
#   %sqrt_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_5,), kwargs = {})
#   %add_4 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg, %sqrt_3), kwargs = {})
#   %gt_1 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_4, 1e-12), kwargs = {})
#   %bitwise_and : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge, %gt_1), kwargs = {})
#   %mul_34 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, -2), kwargs = {})
#   %clamp_min_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_4, 1e-12), kwargs = {})
#   %div_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_34, %clamp_min_6), kwargs = {})
#   %full_default_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and, %div_4, %full_default_1), kwargs = {})
#   %where_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_1, %full_default_2, %where), kwargs = {})
#   %neg_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_1,), kwargs = {})
#   %maximum_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%sub_2, %neg_1), kwargs = {})
#   %ge_5 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_11, 0), kwargs = {})
#   %full_default_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg_2 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_43,), kwargs = {})
#   %pow_24 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg_2, 2), kwargs = {})
#   %mul_49 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, 4), kwargs = {})
#   %mul_50 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %sub_11), kwargs = {})
#   %sub_12 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_24, %mul_50), kwargs = {})
#   %ge_4 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_12, 0), kwargs = {})
#   %clamp_min_9 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_12, 0), kwargs = {})
#   %sqrt_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_9,), kwargs = {})
#   %add_7 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_2, %sqrt_5), kwargs = {})
#   %gt_4 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_7, 1e-12), kwargs = {})
#   %bitwise_and_2 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_4, %gt_4), kwargs = {})
#   %mul_51 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, -2), kwargs = {})
#   %clamp_min_10 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_7, 1e-12), kwargs = {})
#   %div_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_51, %clamp_min_10), kwargs = {})
#   %full_default_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_2, %div_6, %full_default_5), kwargs = {})
#   %where_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_5, %full_default_6, %where_4), kwargs = {})
#   %neg_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_5,), kwargs = {})
#   %maximum_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%maximum_1, %neg_3), kwargs = {})
#   %clamp_min_13 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%maximum_2, -21.045101351351356), kwargs = {})
#   %full_default_10 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], -5.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ge_9 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_18, 0), kwargs = {})
#   %full_default_14 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg_4 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_64,), kwargs = {})
#   %pow_32 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg_4, 2), kwargs = {})
#   %mul_70 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, 4), kwargs = {})
#   %mul_71 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_70, %sub_18), kwargs = {})
#   %sub_19 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_32, %mul_71), kwargs = {})
#   %ge_8 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_19, 0), kwargs = {})
#   %clamp_min_15 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_19, 0), kwargs = {})
#   %sqrt_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_15,), kwargs = {})
#   %add_10 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_4, %sqrt_7), kwargs = {})
#   %gt_8 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_10, 1e-12), kwargs = {})
#   %bitwise_and_4 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_8, %gt_8), kwargs = {})
#   %mul_72 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, -2), kwargs = {})
#   %clamp_min_16 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_10, 1e-12), kwargs = {})
#   %div_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_72, %clamp_min_16), kwargs = {})
#   %full_default_13 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_8 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_4, %div_8, %full_default_13), kwargs = {})
#   %where_9 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_9, %full_default_14, %where_8), kwargs = {})
#   %neg_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_9,), kwargs = {})
#   %maximum_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%full_default_10, %neg_5), kwargs = {})
#   %ge_13 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_23, 0), kwargs = {})
#   %full_default_20 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %neg_6 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_81,), kwargs = {})
#   %pow_40 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%neg_6, 2), kwargs = {})
#   %mul_87 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, 4), kwargs = {})
#   %mul_88 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_87, %sub_23), kwargs = {})
#   %sub_24 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_40, %mul_88), kwargs = {})
#   %ge_12 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sub_24, 0), kwargs = {})
#   %clamp_min_19 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_24, 0), kwargs = {})
#   %sqrt_9 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_19,), kwargs = {})
#   %add_13 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_6, %sqrt_9), kwargs = {})
#   %gt_11 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_13, 1e-12), kwargs = {})
#   %bitwise_and_6 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_12, %gt_11), kwargs = {})
#   %mul_89 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, -2), kwargs = {})
#   %clamp_min_20 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_13, 1e-12), kwargs = {})
#   %div_10 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_89, %clamp_min_20), kwargs = {})
#   %full_default_19 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_12 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_6, %div_10, %full_default_19), kwargs = {})
#   %where_13 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_13, %full_default_20, %where_12), kwargs = {})
#   %neg_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%where_13,), kwargs = {})
#   %maximum_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%maximum_3, %neg_7), kwargs = {})
#   %clamp_min_23 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%maximum_4, -21.045101351351356), kwargs = {})
#   %maximum_5 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%clamp_min_13, %clamp_min_23), kwargs = {})
#   %maximum_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%minimum_9, %maximum_5), kwargs = {})
#   %mul_94 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%maximum_6, 2), kwargs = {})
#   %mul_95 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_94, %arg1_1), kwargs = {})
#   %add_15 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_44, %mul_95), kwargs = {})
#   %clamp_min_24 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_15, 0), kwargs = {})
#   %sqrt_11 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min_24,), kwargs = {})
#   return %mul_26,%mul_43,%minimum_3,%sub_6,%sub_11,%sub_18,%sub_23,%minimum_8,%maximum_5,%sqrt_11
triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0 = async_compile.triton('triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp25 = tl.load(in_ptr3 + (0))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp49 = tl.load(in_ptr4 + (0))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp4 = tmp1 * tmp1
    tmp7 = 14.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp4 + tmp8
    tmp10 = tl.sqrt_rn(tmp9)
    tmp11 = triton_helpers.minimum(tmp3, tmp10)
    tmp12 = triton_helpers.maximum(tmp1, tmp11)
    tmp13 = tl_math.abs(tmp12)
    tmp14 = 20.0
    tmp15 = tmp13 * tmp14
    tmp16 = libdevice.tanh(tmp15)
    tmp17 = 0.1
    tmp18 = tmp16 * tmp17
    tmp19 = tmp12 * tmp12
    tmp20 = 0.01
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = 0.25
    tmp24 = tmp22 * tmp23
    tmp27 = 0.92
    tmp28 = tmp26 * tmp27
    tmp29 = tmp28 * tmp28
    tmp30 = 9.81
    tmp31 = tmp29 * tmp30
    tmp32 = 0.5192307692307692
    tmp33 = tmp31 * tmp32
    tmp34 = -0.22410660205935795
    tmp35 = tmp33 * tmp34
    tmp36 = tmp24 - tmp35
    tmp37 = 2.0
    tmp38 = tmp36 * tmp37
    tmp39 = 1.0
    tmp40 = tmp26 * tmp39
    tmp41 = tmp40 * tmp40
    tmp42 = tmp41 * tmp30
    tmp43 = 0.4807692307692308
    tmp44 = tmp42 * tmp43
    tmp45 = 0.22410660205935795
    tmp46 = tmp44 * tmp45
    tmp47 = tmp24 - tmp46
    tmp48 = tmp47 * tmp37
    tmp51 = 0.85
    tmp52 = tmp28 * tmp51
    tmp53 = tmp52 * tmp30
    tmp54 = tmp53 * tmp32
    tmp55 = tmp54 * tmp54
    tmp56 = 0.5
    tmp57 = tmp22 * tmp56
    tmp58 = tmp57 * tmp57
    tmp59 = tmp55 - tmp58
    tmp60 = 0.0
    tmp61 = triton_helpers.maximum(tmp59, tmp60)
    tmp62 = tl.sqrt_rn(tmp61)
    tmp63 = tmp19 * tmp32
    tmp64 = 1e-06
    tmp65 = triton_helpers.maximum(tmp63, tmp64)
    tmp66 = (tmp62 / tmp65)
    tmp67 = float("inf")
    tmp68 = triton_helpers.minimum(tmp67, tmp66)
    tmp69 = tmp40 * tmp51
    tmp70 = tmp69 * tmp30
    tmp71 = tmp70 * tmp43
    tmp72 = tmp71 * tmp71
    tmp73 = tmp72 - tmp58
    tmp74 = triton_helpers.maximum(tmp73, tmp60)
    tmp75 = tl.sqrt_rn(tmp74)
    tmp76 = tmp19 * tmp43
    tmp77 = triton_helpers.maximum(tmp76, tmp64)
    tmp78 = (tmp75 / tmp77)
    tmp79 = triton_helpers.minimum(tmp68, tmp78)
    tmp80 = triton_helpers.minimum(tmp50, tmp79)
    tmp81 = tmp63 * tmp80
    tmp82 = tmp81 * tmp81
    tmp83 = tmp58 + tmp82
    tmp84 = tmp28 * tmp30
    tmp85 = tmp84 * tmp32
    tmp86 = tmp85 * tmp85
    tmp87 = tmp83 - tmp86
    tmp88 = tmp76 * tmp80
    tmp89 = tmp88 * tmp88
    tmp90 = tmp58 + tmp89
    tmp91 = tmp40 * tmp30
    tmp92 = tmp91 * tmp43
    tmp93 = tmp92 * tmp92
    tmp94 = tmp90 - tmp93
    tmp95 = tmp82 - tmp86
    tmp96 = tmp89 - tmp93
    tmp97 = 0.001
    tmp98 = triton_helpers.maximum(tmp13, tmp97)
    tmp99 = tl.full([1], 1, tl.int32)
    tmp100 = (tmp99 / tmp98)
    tmp101 = 7.319
    tmp102 = tmp100 * tmp101
    tmp103 = triton_helpers.minimum(tmp102, tmp39)
    tmp104 = 7.0
    tmp105 = tmp103 * tmp104
    tmp106 = tmp105 - tmp22
    tmp107 = tmp87 >= tmp60
    tmp108 = tmp38 * tmp38
    tmp109 = tmp28 * tmp34
    tmp110 = tmp109 * tmp109
    tmp111 = tmp23 - tmp110
    tmp112 = 4.0
    tmp113 = tmp111 * tmp112
    tmp114 = tmp113 * tmp87
    tmp115 = tmp108 - tmp114
    tmp116 = tmp115 >= tmp60
    tmp117 = triton_helpers.maximum(tmp115, tmp60)
    tmp118 = tl.sqrt_rn(tmp117)
    tmp119 = tmp38 + tmp118
    tmp120 = 1e-12
    tmp121 = tmp119 > tmp120
    tmp122 = tmp116 & tmp121
    tmp123 = -2.0
    tmp124 = tmp87 * tmp123
    tmp125 = triton_helpers.maximum(tmp119, tmp120)
    tmp126 = (tmp124 / tmp125)
    tmp127 = tl.where(tmp122, tmp126, tmp67)
    tmp128 = tl.where(tmp107, tmp60, tmp127)
    tmp129 = triton_helpers.minimum(tmp106, tmp128)
    tmp130 = 22.72870945945946
    tmp131 = triton_helpers.minimum(tmp129, tmp130)
    tmp132 = tmp94 >= tmp60
    tmp133 = tmp48 * tmp48
    tmp134 = tmp40 * tmp45
    tmp135 = tmp134 * tmp134
    tmp136 = tmp23 - tmp135
    tmp137 = tmp136 * tmp112
    tmp138 = tmp137 * tmp94
    tmp139 = tmp133 - tmp138
    tmp140 = tmp139 >= tmp60
    tmp141 = triton_helpers.maximum(tmp139, tmp60)
    tmp142 = tl.sqrt_rn(tmp141)
    tmp143 = tmp48 + tmp142
    tmp144 = tmp143 > tmp120
    tmp145 = tmp140 & tmp144
    tmp146 = tmp94 * tmp123
    tmp147 = triton_helpers.maximum(tmp143, tmp120)
    tmp148 = (tmp146 / tmp147)
    tmp149 = tl.where(tmp145, tmp148, tmp67)
    tmp150 = tl.where(tmp132, tmp60, tmp149)
    tmp151 = triton_helpers.minimum(tmp131, tmp150)
    tmp152 = tmp95 >= tmp60
    tmp153 = tmp60 - tmp35
    tmp154 = tmp153 * tmp37
    tmp155 = tmp154 * tmp154
    tmp156 = tmp113 * tmp95
    tmp157 = tmp155 - tmp156
    tmp158 = tmp157 >= tmp60
    tmp159 = triton_helpers.maximum(tmp157, tmp60)
    tmp160 = tl.sqrt_rn(tmp159)
    tmp161 = tmp154 + tmp160
    tmp162 = tmp161 > tmp120
    tmp163 = tmp158 & tmp162
    tmp164 = tmp95 * tmp123
    tmp165 = triton_helpers.maximum(tmp161, tmp120)
    tmp166 = (tmp164 / tmp165)
    tmp167 = tl.where(tmp163, tmp166, tmp67)
    tmp168 = tl.where(tmp152, tmp60, tmp167)
    tmp169 = triton_helpers.minimum(tmp105, tmp168)
    tmp170 = triton_helpers.minimum(tmp169, tmp130)
    tmp171 = tmp96 >= tmp60
    tmp172 = tmp60 - tmp46
    tmp173 = tmp172 * tmp37
    tmp174 = tmp173 * tmp173
    tmp175 = tmp137 * tmp96
    tmp176 = tmp174 - tmp175
    tmp177 = tmp176 >= tmp60
    tmp178 = triton_helpers.maximum(tmp176, tmp60)
    tmp179 = tl.sqrt_rn(tmp178)
    tmp180 = tmp173 + tmp179
    tmp181 = tmp180 > tmp120
    tmp182 = tmp177 & tmp181
    tmp183 = tmp96 * tmp123
    tmp184 = triton_helpers.maximum(tmp180, tmp120)
    tmp185 = (tmp183 / tmp184)
    tmp186 = tl.where(tmp182, tmp185, tmp67)
    tmp187 = tl.where(tmp171, tmp60, tmp186)
    tmp188 = triton_helpers.minimum(tmp170, tmp187)
    tmp189 = triton_helpers.minimum(tmp151, tmp188)
    tmp190 = -5.0
    tmp191 = tmp190 - tmp22
    tmp192 = -tmp38
    tmp193 = tmp192 * tmp192
    tmp194 = tmp193 - tmp114
    tmp195 = tmp194 >= tmp60
    tmp196 = triton_helpers.maximum(tmp194, tmp60)
    tmp197 = tl.sqrt_rn(tmp196)
    tmp198 = tmp192 + tmp197
    tmp199 = tmp198 > tmp120
    tmp200 = tmp195 & tmp199
    tmp201 = triton_helpers.maximum(tmp198, tmp120)
    tmp202 = (tmp124 / tmp201)
    tmp203 = tl.where(tmp200, tmp202, tmp67)
    tmp204 = tl.where(tmp107, tmp60, tmp203)
    tmp205 = -tmp204
    tmp206 = triton_helpers.maximum(tmp191, tmp205)
    tmp207 = -tmp48
    tmp208 = tmp207 * tmp207
    tmp209 = tmp208 - tmp138
    tmp210 = tmp209 >= tmp60
    tmp211 = triton_helpers.maximum(tmp209, tmp60)
    tmp212 = tl.sqrt_rn(tmp211)
    tmp213 = tmp207 + tmp212
    tmp214 = tmp213 > tmp120
    tmp215 = tmp210 & tmp214
    tmp216 = triton_helpers.maximum(tmp213, tmp120)
    tmp217 = (tmp146 / tmp216)
    tmp218 = tl.where(tmp215, tmp217, tmp67)
    tmp219 = tl.where(tmp132, tmp60, tmp218)
    tmp220 = -tmp219
    tmp221 = triton_helpers.maximum(tmp206, tmp220)
    tmp222 = -21.045101351351356
    tmp223 = triton_helpers.maximum(tmp221, tmp222)
    tmp224 = -tmp154
    tmp225 = tmp224 * tmp224
    tmp226 = tmp225 - tmp156
    tmp227 = tmp226 >= tmp60
    tmp228 = triton_helpers.maximum(tmp226, tmp60)
    tmp229 = tl.sqrt_rn(tmp228)
    tmp230 = tmp224 + tmp229
    tmp231 = tmp230 > tmp120
    tmp232 = tmp227 & tmp231
    tmp233 = triton_helpers.maximum(tmp230, tmp120)
    tmp234 = (tmp164 / tmp233)
    tmp235 = tl.where(tmp232, tmp234, tmp67)
    tmp236 = tl.where(tmp152, tmp60, tmp235)
    tmp237 = -tmp236
    tmp238 = triton_helpers.maximum(tmp190, tmp237)
    tmp239 = -tmp173
    tmp240 = tmp239 * tmp239
    tmp241 = tmp240 - tmp175
    tmp242 = tmp241 >= tmp60
    tmp243 = triton_helpers.maximum(tmp241, tmp60)
    tmp244 = tl.sqrt_rn(tmp243)
    tmp245 = tmp239 + tmp244
    tmp246 = tmp245 > tmp120
    tmp247 = tmp242 & tmp246
    tmp248 = triton_helpers.maximum(tmp245, tmp120)
    tmp249 = (tmp183 / tmp248)
    tmp250 = tl.where(tmp247, tmp249, tmp67)
    tmp251 = tl.where(tmp171, tmp60, tmp250)
    tmp252 = -tmp251
    tmp253 = triton_helpers.maximum(tmp238, tmp252)
    tmp254 = triton_helpers.maximum(tmp253, tmp222)
    tmp255 = triton_helpers.maximum(tmp223, tmp254)
    tmp256 = tmp3 * tmp3
    tmp257 = tmp256 - tmp4
    tmp258 = tmp6 * tmp37
    tmp259 = (tmp257 / tmp258)
    tmp260 = triton_helpers.minimum(tmp259, tmp189)
    tmp261 = triton_helpers.maximum(tmp260, tmp255)
    tmp262 = tmp261 * tmp37
    tmp263 = tmp262 * tmp6
    tmp264 = tmp4 + tmp263
    tmp265 = triton_helpers.maximum(tmp264, tmp60)
    tmp266 = tl.sqrt_rn(tmp265)
    tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp80, None)
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp266, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, ), (1, ))
        assert_size_stride(arg1_1, (1, ), (1, ))
        assert_size_stride(arg2_1, (1, ), (25, ))
        assert_size_stride(arg3_1, (1, ), (1, ))
        assert_size_stride(arg4_1, (1, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, ), (1, ), torch.float32)
            buf7 = empty_strided_cuda((1, ), (1, ), torch.float32)
            buf9 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [square_43, square_41, square_42, sub_26, mul_91, want, square, mul, add, sqrt, minimum, fastest, abs_3, clamp_min_4, truediv_4, clamp, power, abs_2, truediv_3, tanh_1, mul_15, square_8, mul_16, r_1, hi, mul_26, square_11, square_12, mul_27, cap, grip, mul_5, mul_6, mul_7, square_2, abs_1, truediv, tanh, mul_1, square_1, mul_2, r, mul_8, square_3, sub, residual, sqrt_1, square_4, mul_9, clamp_min_1, truediv_1, cap_1, grip_1, mul_10, mul_11, mul_12, square_5, mul_13, square_6, sub_1, residual_1, sqrt_2, square_7, mul_14, clamp_min_3, truediv_2, cap_2, safe_k, mul_28, square_13, add_3, grip_2, mul_29, mul_30, square_14, cc, ge_3, zeros_like_2, mul_21, square_10, mul_22, mul_23, mul_24, sub_5, bb, square_16, mul_20, square_9, aa, mul_34, mul_35, disc_1, ge_2, clamp_min_7, sqrt_4, den_1, gt_2, and__1, mul_36, clamp_min_8, root_2, full_like_2, root_3, where_3, hi_1, hi_2, mul_43, square_19, square_20, mul_44, mul_45, square_21, add_6, grip_3, mul_46, mul_47, square_22, cc_1, ge_7, zeros_like_4, mul_38, square_18, mul_39, mul_40, mul_41, sub_10, bb_1, square_24, mul_37, square_17, aa_1, mul_51, mul_52, disc_3, ge_6, clamp_min_11, sqrt_6, den_3, gt_5, and__3, mul_53, clamp_min_12, root_6, full_like_4, root_7, where_7, hi_3, abs_4, clamp_min_13, truediv_9, clamp_3, hi_4, square_28, mul_64, mul_65, add_9, grip_4, mul_66, mul_67, square_30, cc_2, ge_11, zeros_like_8, mul_58, square_26, mul_59, mul_60, mul_61, sub_17, bb_2, square_32, mul_57, square_25, aa_2, mul_71, mul_72, disc_5, ge_10, clamp_min_16, sqrt_8, den_5, gt_9, and__5, mul_73, clamp_min_17, root_10, full_like_6, root_11, where_11, hi_5, hi_6, square_36, mul_81, mul_82, add_12, grip_5, mul_83, mul_84, square_38, cc_3, ge_15, zeros_like_10, mul_75, square_34, mul_76, mul_77, mul_78, sub_22, bb_3, square_40, mul_74, square_33, aa_3, mul_88, mul_89, disc_7, ge_14, clamp_min_20, sqrt_10, den_7, gt_12, and__7, mul_90, clamp_min_21, root_14, full_like_8, root_15, where_15, hi_7, upper, minimum_9, lo, ge_1, zeros_like_1, neg, square_15, mul_31, mul_32, disc, ge, clamp_min_5, sqrt_3, den, gt_1, and_, mul_33, clamp_min_6, root, full_like_1, root_1, where_1, neg_1, lo_1, ge_5, zeros_like_3, neg_2, square_23, mul_48, mul_49, disc_2, ge_4, clamp_min_9, sqrt_5, den_2, gt_4, and__2, mul_50, clamp_min_10, root_4, full_like_3, root_5, where_5, neg_3, lo_2, lo_3, lo_4, ge_9, zeros_like_7, neg_4, square_31, mul_68, mul_69, disc_4, ge_8, clamp_min_14, sqrt_7, den_4, gt_8, and__4, mul_70, clamp_min_15, root_8, full_like_5, root_9, where_9, neg_5, lo_5, ge_13, zeros_like_9, neg_6, square_39, mul_85, mul_86, disc_6, ge_12, clamp_min_18, sqrt_9, den_6, gt_11, and__6, mul_87, clamp_min_19, root_12, full_like_7, root_13, where_13, neg_7, lo_6, lo_7, lower, ax, mul_92, mul_93, add_15, clamp_min_22, sqrt_11], Original ATen: [aten.pow, aten.sub, aten.mul, aten.div, aten.add, aten.sqrt, aten.minimum, aten.maximum, aten.abs, aten.clamp_min, aten.reciprocal, aten.clamp, aten.tanh, aten.full_like, aten.ge, aten.zeros_like, aten.rsub, aten.gt, aten.bitwise_and, aten.where, aten.neg]
            stream0 = get_raw_stream(0)
            triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0.run(buf9, arg0_1, arg2_1, arg1_1, arg3_1, arg4_1, buf0, 1, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            del arg3_1
            del arg4_1
        return (buf9, buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, ), (25, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
