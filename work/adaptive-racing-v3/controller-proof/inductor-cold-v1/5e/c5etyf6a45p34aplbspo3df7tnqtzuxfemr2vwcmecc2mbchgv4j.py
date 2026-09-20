
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
