
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*i1', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_slice_sqrt_sub_tan_tanh_unsqueeze_where_zeros_like_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 7, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_bitwise_and_bitwise_or_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_select_slice_sqrt_sub_tan_tanh_unsqueeze_where_zeros_like_0(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + 6*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (2*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (0))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp131 = tl.load(in_ptr1 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tmp186 = tl.load(in_ptr0 + (4 + 6*x0), xmask, eviction_policy='evict_last')
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
    tmp28 = 0.92
    tmp29 = tmp27 * tmp28
    tmp30 = 9.81
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31 * tmp14
    tmp33 = tmp32 * tmp32
    tmp34 = tmp25 - tmp33
    tmp35 = 0.4807692307692308
    tmp36 = tmp7 * tmp35
    tmp37 = tmp36 * tmp22
    tmp38 = tmp37 * tmp37
    tmp39 = tmp13 + tmp38
    tmp40 = 1.0
    tmp41 = tmp27 * tmp40
    tmp42 = tmp41 * tmp30
    tmp43 = tmp42 * tmp35
    tmp44 = tmp43 * tmp43
    tmp45 = tmp39 - tmp44
    tmp46 = 0.25
    tmp47 = tmp10 * tmp46
    tmp48 = tmp29 * tmp29
    tmp49 = tmp48 * tmp30
    tmp50 = tmp49 * tmp14
    tmp51 = -0.22410660205935795
    tmp52 = tmp50 * tmp51
    tmp53 = tmp47 - tmp52
    tmp54 = 2.0
    tmp55 = tmp53 * tmp54
    tmp56 = -tmp55
    tmp57 = tmp56 * tmp56
    tmp58 = tmp29 * tmp51
    tmp59 = tmp58 * tmp58
    tmp60 = tmp46 - tmp59
    tmp61 = 4.0
    tmp62 = tmp60 * tmp61
    tmp63 = tmp62 * tmp34
    tmp64 = tmp57 - tmp63
    tmp65 = 0.0
    tmp66 = triton_helpers.maximum(tmp64, tmp65)
    tmp67 = tl.sqrt_rn(tmp66)
    tmp68 = tmp56 + tmp67
    tmp69 = -5.0
    tmp70 = tmp69 - tmp10
    tmp71 = tmp34 >= tmp65
    tmp72 = tmp64 >= tmp65
    tmp73 = 1e-12
    tmp74 = tmp68 > tmp73
    tmp75 = tmp72 & tmp74
    tmp76 = -2.0
    tmp77 = tmp34 * tmp76
    tmp78 = triton_helpers.maximum(tmp68, tmp73)
    tmp79 = (tmp77 / tmp78)
    tmp80 = float("inf")
    tmp81 = tl.where(tmp75, tmp79, tmp80)
    tmp82 = tl.where(tmp71, tmp65, tmp81)
    tmp83 = -tmp82
    tmp84 = triton_helpers.maximum(tmp70, tmp83)
    tmp85 = tmp41 * tmp41
    tmp86 = tmp85 * tmp30
    tmp87 = tmp86 * tmp35
    tmp88 = 0.22410660205935795
    tmp89 = tmp87 * tmp88
    tmp90 = tmp47 - tmp89
    tmp91 = tmp90 * tmp54
    tmp92 = -tmp91
    tmp93 = tmp92 * tmp92
    tmp94 = tmp41 * tmp88
    tmp95 = tmp94 * tmp94
    tmp96 = tmp46 - tmp95
    tmp97 = tmp96 * tmp61
    tmp98 = tmp97 * tmp45
    tmp99 = tmp93 - tmp98
    tmp100 = triton_helpers.maximum(tmp99, tmp65)
    tmp101 = tl.sqrt_rn(tmp100)
    tmp102 = tmp92 + tmp101
    tmp103 = tmp55 * tmp55
    tmp104 = tmp103 - tmp63
    tmp105 = triton_helpers.maximum(tmp104, tmp65)
    tmp106 = tl.sqrt_rn(tmp105)
    tmp107 = tmp55 + tmp106
    tmp108 = 0.001
    tmp109 = triton_helpers.maximum(tmp1, tmp108)
    tmp110 = tl.full([1], 1, tl.int32)
    tmp111 = (tmp110 / tmp109)
    tmp112 = 7.319
    tmp113 = tmp111 * tmp112
    tmp114 = triton_helpers.minimum(tmp113, tmp40)
    tmp115 = 7.0
    tmp116 = tmp114 * tmp115
    tmp117 = tmp116 - tmp10
    tmp118 = tmp104 >= tmp65
    tmp119 = tmp107 > tmp73
    tmp120 = tmp118 & tmp119
    tmp121 = triton_helpers.maximum(tmp107, tmp73)
    tmp122 = (tmp77 / tmp121)
    tmp123 = tl.where(tmp120, tmp122, tmp80)
    tmp124 = tl.where(tmp71, tmp65, tmp123)
    tmp125 = triton_helpers.minimum(tmp117, tmp124)
    tmp126 = tmp91 * tmp91
    tmp127 = tmp126 - tmp98
    tmp128 = triton_helpers.maximum(tmp127, tmp65)
    tmp129 = tl.sqrt_rn(tmp128)
    tmp130 = tmp91 + tmp129
    tmp132 = tmp131 + tmp10
    tmp133 = tmp132 * tmp11
    tmp134 = tmp133 * tmp133
    tmp135 = tmp134 + tmp24
    tmp136 = tl.sqrt_rn(tmp135)
    tmp137 = tmp131 * tmp51
    tmp138 = 5.093653846153845
    tmp139 = tmp137 + tmp138
    tmp140 = triton_helpers.maximum(tmp139, tmp65)
    tmp141 = tmp29 * tmp140
    tmp142 = tmp136 - tmp141
    tmp143 = tmp134 + tmp38
    tmp144 = tl.sqrt_rn(tmp143)
    tmp145 = tmp131 * tmp88
    tmp146 = 4.716346153846154
    tmp147 = tmp145 + tmp146
    tmp148 = triton_helpers.maximum(tmp147, tmp65)
    tmp149 = tmp41 * tmp148
    tmp150 = tmp144 - tmp149
    tmp151 = tmp45 >= tmp65
    tmp152 = tmp99 >= tmp65
    tmp153 = tmp102 > tmp73
    tmp154 = tmp152 & tmp153
    tmp155 = tmp45 * tmp76
    tmp156 = triton_helpers.maximum(tmp102, tmp73)
    tmp157 = (tmp155 / tmp156)
    tmp158 = tl.where(tmp154, tmp157, tmp80)
    tmp159 = tl.where(tmp151, tmp65, tmp158)
    tmp160 = -tmp159
    tmp161 = triton_helpers.maximum(tmp84, tmp160)
    tmp162 = -21.045101351351356
    tmp163 = triton_helpers.maximum(tmp161, tmp162)
    tmp164 = 22.72870945945946
    tmp165 = triton_helpers.minimum(tmp125, tmp164)
    tmp166 = tmp127 >= tmp65
    tmp167 = tmp130 > tmp73
    tmp168 = tmp166 & tmp167
    tmp169 = triton_helpers.maximum(tmp130, tmp73)
    tmp170 = (tmp155 / tmp169)
    tmp171 = tl.where(tmp168, tmp170, tmp80)
    tmp172 = tl.where(tmp151, tmp65, tmp171)
    tmp173 = triton_helpers.minimum(tmp165, tmp172)
    tmp174 = 1e-05
    tmp175 = tmp34 > tmp174
    tmp176 = tl.full([1], False, tl.int1)
    tmp177 = tmp176 | tmp175
    tmp178 = tmp45 > tmp174
    tmp179 = tmp177 | tmp178
    tmp180 = tmp163 > tmp173
    tmp181 = tmp179 | tmp180
    tmp182 = tmp163 - tmp131
    tmp183 = tmp131 - tmp173
    tmp184 = triton_helpers.maximum(tmp182, tmp183)
    tmp185 = triton_helpers.maximum(tmp184, tmp65)
    tmp187 = tmp16 - tmp186
    tmp188 = tl_math.abs(tmp187)
    tmp189 = 0.16000000000000003
    tmp190 = tmp188 - tmp189
    tmp191 = triton_helpers.maximum(tmp190, tmp65)
    tl.store(out_ptr8 + (x0), tmp142, xmask)
    tl.store(out_ptr9 + (x0), tmp150, xmask)
    tl.store(in_out_ptr0 + (x0), tmp163, xmask)
    tl.store(in_out_ptr1 + (x0), tmp173, xmask)
    tl.store(out_ptr10 + (x0), tmp181, xmask)
    tl.store(out_ptr11 + (x0), tmp185, xmask)
    tl.store(out_ptr12 + (x0), tmp191, xmask)
