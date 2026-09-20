
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
