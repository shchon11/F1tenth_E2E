
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*i1', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_slice_sqrt_sub_tan_tanh_unsqueeze_where_zeros_like_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 7, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 28800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_slice_sqrt_sub_tan_tanh_unsqueeze_where_zeros_like_0(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
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
