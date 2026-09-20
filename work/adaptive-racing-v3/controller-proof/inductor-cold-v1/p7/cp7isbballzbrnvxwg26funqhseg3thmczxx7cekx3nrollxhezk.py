
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
