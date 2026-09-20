
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr5': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_bitwise_and_cat_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_slice_sqrt_sub_tanh_where_zeros_like_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr5'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 3, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2384}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_bitwise_and_cat_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_slice_sqrt_sub_tanh_where_zeros_like_2(in_out_ptr0, in_out_ptr1, in_out_ptr5, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50
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
