
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
