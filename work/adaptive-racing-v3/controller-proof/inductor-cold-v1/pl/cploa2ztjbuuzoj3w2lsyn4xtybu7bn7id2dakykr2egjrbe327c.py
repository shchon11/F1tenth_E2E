
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'out_ptr11': '*fp32', 'out_ptr12': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_atan_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_le_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_sqrt_sub_tan_tanh_where_zeros_like_0', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 4, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 15}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_atan_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_le_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_sqrt_sub_tan_tanh_where_zeros_like_0(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr11, out_ptr12, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp10 = tl.load(in_ptr1 + (3 + 6*x0), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr1 + (4 + 6*x0), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr2 + (2*x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.92
    tmp2 = tmp0 * tmp1
    tmp3 = 0.85
    tmp4 = tmp2 * tmp3
    tmp5 = 9.81
    tmp6 = tmp4 * tmp5
    tmp7 = 0.5192307692307692
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 * tmp8
    tmp11 = tl_math.abs(tmp10)
    tmp12 = 20.0
    tmp13 = tmp11 * tmp12
    tmp14 = libdevice.tanh(tmp13)
    tmp15 = 0.1
    tmp16 = tmp14 * tmp15
    tmp17 = tmp10 * tmp10
    tmp18 = 0.01
    tmp19 = tmp17 * tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = 0.5
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tmp9 - tmp23
    tmp25 = 0.0
    tmp26 = triton_helpers.maximum(tmp24, tmp25)
    tmp27 = tl.sqrt_rn(tmp26)
    tmp28 = tmp17 * tmp7
    tmp29 = 1e-06
    tmp30 = triton_helpers.maximum(tmp28, tmp29)
    tmp31 = (tmp27 / tmp30)
    tmp32 = 1.0
    tmp33 = tmp0 * tmp32
    tmp34 = tmp33 * tmp3
    tmp35 = tmp34 * tmp5
    tmp36 = 0.4807692307692308
    tmp37 = tmp35 * tmp36
    tmp38 = tmp37 * tmp37
    tmp39 = tmp38 - tmp23
    tmp40 = triton_helpers.maximum(tmp39, tmp25)
    tmp41 = tl.sqrt_rn(tmp40)
    tmp42 = tmp17 * tmp36
    tmp43 = triton_helpers.maximum(tmp42, tmp29)
    tmp44 = (tmp41 / tmp43)
    tmp45 = 0.003
    tmp46 = tmp17 * tmp45
    tmp47 = 0.3302
    tmp48 = tmp46 + tmp47
    tmp49 = float("inf")
    tmp50 = triton_helpers.minimum(tmp49, tmp31)
    tmp51 = triton_helpers.minimum(tmp50, tmp44)
    tmp52 = tmp48 * tmp51
    tmp53 = libdevice.atan(tmp52)
    tmp54 = 0.4189
    tmp55 = triton_helpers.minimum(tmp53, tmp54)
    tmp56 = -tmp55
    tmp58 = 0.16000000000000003
    tmp59 = tmp57 - tmp58
    tmp60 = triton_helpers.maximum(tmp56, tmp59)
    tmp61 = tmp57 + tmp58
    tmp62 = triton_helpers.minimum(tmp55, tmp61)
    tmp63 = tmp60 <= tmp62
    tmp65 = triton_helpers.minimum(tmp64, tmp62)
    tmp66 = triton_helpers.maximum(tmp65, tmp60)
    tmp67 = triton_helpers.minimum(tmp25, tmp61)
    tmp68 = triton_helpers.maximum(tmp67, tmp59)
    tmp69 = -0.4189
    tmp70 = triton_helpers.maximum(tmp68, tmp69)
    tmp71 = triton_helpers.minimum(tmp70, tmp54)
    tmp72 = tl.where(tmp63, tmp66, tmp71)
    tmp73 = libdevice.tan(tmp72)
    tmp74 = (tmp73 / tmp48)
    tmp75 = tmp28 * tmp74
    tmp76 = tmp75 * tmp75
    tmp77 = tmp23 + tmp76
    tmp78 = tmp2 * tmp5
    tmp79 = tmp78 * tmp7
    tmp80 = tmp79 * tmp79
    tmp81 = tmp77 - tmp80
    tmp82 = 0.25
    tmp83 = tmp20 * tmp82
    tmp84 = tmp2 * tmp2
    tmp85 = tmp84 * tmp5
    tmp86 = tmp85 * tmp7
    tmp87 = -0.22410660205935795
    tmp88 = tmp86 * tmp87
    tmp89 = tmp83 - tmp88
    tmp90 = 2.0
    tmp91 = tmp89 * tmp90
    tmp92 = tmp91 * tmp91
    tmp93 = tmp2 * tmp87
    tmp94 = tmp93 * tmp93
    tmp95 = tmp82 - tmp94
    tmp96 = 4.0
    tmp97 = tmp95 * tmp96
    tmp98 = tmp97 * tmp81
    tmp99 = tmp92 - tmp98
    tmp100 = triton_helpers.maximum(tmp99, tmp25)
    tmp101 = tl.sqrt_rn(tmp100)
    tmp102 = tmp91 + tmp101
    tmp103 = 0.001
    tmp104 = triton_helpers.maximum(tmp11, tmp103)
    tmp105 = tl.full([1], 1, tl.int32)
    tmp106 = (tmp105 / tmp104)
    tmp107 = 7.319
    tmp108 = tmp106 * tmp107
    tmp109 = triton_helpers.minimum(tmp108, tmp32)
    tmp110 = 7.0
    tmp111 = tmp109 * tmp110
    tmp112 = tmp111 - tmp20
    tmp113 = tmp81 >= tmp25
    tmp114 = tmp99 >= tmp25
    tmp115 = 1e-12
    tmp116 = tmp102 > tmp115
    tmp117 = tmp114 & tmp116
    tmp118 = -2.0
    tmp119 = tmp81 * tmp118
    tmp120 = triton_helpers.maximum(tmp102, tmp115)
    tmp121 = (tmp119 / tmp120)
    tmp122 = tl.where(tmp117, tmp121, tmp49)
    tmp123 = tl.where(tmp113, tmp25, tmp122)
    tmp124 = triton_helpers.minimum(tmp112, tmp123)
    tmp125 = tmp42 * tmp74
    tmp126 = tmp125 * tmp125
    tmp127 = tmp23 + tmp126
    tmp128 = tmp33 * tmp5
    tmp129 = tmp128 * tmp36
    tmp130 = tmp129 * tmp129
    tmp131 = tmp127 - tmp130
    tmp132 = tmp33 * tmp33
    tmp133 = tmp132 * tmp5
    tmp134 = tmp133 * tmp36
    tmp135 = 0.22410660205935795
    tmp136 = tmp134 * tmp135
    tmp137 = tmp83 - tmp136
    tmp138 = tmp137 * tmp90
    tmp139 = tmp138 * tmp138
    tmp140 = tmp33 * tmp135
    tmp141 = tmp140 * tmp140
    tmp142 = tmp82 - tmp141
    tmp143 = tmp142 * tmp96
    tmp144 = tmp143 * tmp131
    tmp145 = tmp139 - tmp144
    tmp146 = triton_helpers.maximum(tmp145, tmp25)
    tmp147 = tl.sqrt_rn(tmp146)
    tmp148 = tmp138 + tmp147
    tmp149 = -tmp138
    tmp150 = tmp149 * tmp149
    tmp151 = tmp150 - tmp144
    tmp152 = triton_helpers.maximum(tmp151, tmp25)
    tmp153 = tl.sqrt_rn(tmp152)
    tmp154 = tmp149 + tmp153
    tmp155 = -tmp91
    tmp156 = tmp155 * tmp155
    tmp157 = tmp156 - tmp98
    tmp158 = triton_helpers.maximum(tmp157, tmp25)
    tmp159 = tl.sqrt_rn(tmp158)
    tmp160 = tmp155 + tmp159
    tmp161 = -5.0
    tmp162 = tmp161 - tmp20
    tmp163 = tmp157 >= tmp25
    tmp164 = tmp160 > tmp115
    tmp165 = tmp163 & tmp164
    tmp166 = triton_helpers.maximum(tmp160, tmp115)
    tmp167 = (tmp119 / tmp166)
    tmp168 = tl.where(tmp165, tmp167, tmp49)
    tmp169 = tl.where(tmp113, tmp25, tmp168)
    tmp170 = -tmp169
    tmp171 = triton_helpers.maximum(tmp162, tmp170)
    tmp172 = tmp131 >= tmp25
    tmp173 = tmp151 >= tmp25
    tmp174 = tmp154 > tmp115
    tmp175 = tmp173 & tmp174
    tmp176 = tmp131 * tmp118
    tmp177 = triton_helpers.maximum(tmp154, tmp115)
    tmp178 = (tmp176 / tmp177)
    tmp179 = tl.where(tmp175, tmp178, tmp49)
    tmp180 = tl.where(tmp172, tmp25, tmp179)
    tmp181 = -tmp180
    tmp182 = triton_helpers.maximum(tmp171, tmp181)
    tmp183 = -21.045101351351356
    tmp184 = triton_helpers.maximum(tmp182, tmp183)
    tmp185 = 22.72870945945946
    tmp186 = triton_helpers.minimum(tmp124, tmp185)
    tmp187 = tmp145 >= tmp25
    tmp188 = tmp148 > tmp115
    tmp189 = tmp187 & tmp188
    tmp190 = triton_helpers.maximum(tmp148, tmp115)
    tmp191 = (tmp176 / tmp190)
    tmp192 = tl.where(tmp189, tmp191, tmp49)
    tmp193 = tl.where(tmp172, tmp25, tmp192)
    tmp194 = triton_helpers.minimum(tmp186, tmp193)
    tmp195 = 12.0
    tmp196 = tmp195 - tmp10
    tmp197 = tmp196 * tmp12
    tmp198 = triton_helpers.minimum(tmp194, tmp197)
    tmp199 = triton_helpers.maximum(tmp198, tmp184)
    tmp200 = 1e-05
    tmp201 = tmp81 > tmp200
    tmp202 = tl.full([1], False, tl.int1)
    tmp203 = tmp202 | tmp201
    tmp204 = tmp131 > tmp200
    tmp205 = tmp203 | tmp204
    tmp206 = tmp184 > tmp194
    tmp207 = tmp205 | tmp206
    tmp208 = tmp60 > tmp62
    tmp209 = tmp207 | tmp208
    tl.store(out_ptr2 + (x0), tmp72, xmask)
    tl.store(in_out_ptr1 + (x0), tmp184, xmask)
    tl.store(out_ptr11 + (x0), tmp199, xmask)
    tl.store(out_ptr12 + (x0), tmp209, xmask)
