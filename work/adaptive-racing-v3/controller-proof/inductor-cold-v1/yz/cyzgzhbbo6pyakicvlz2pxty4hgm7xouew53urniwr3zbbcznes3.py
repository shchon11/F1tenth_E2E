
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr3': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr3'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_full_like_ge_gt_maximum_minimum_mul_neg_pow_reciprocal_rsub_sqrt_sub_tanh_where_zeros_like_0(in_out_ptr0, in_out_ptr3, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp10 = tl.load(in_ptr1 + (x0), xmask)
    tmp11 = tl.load(in_ptr2 + (x0), xmask)
    tmp13 = tl.load(in_ptr3 + (x0), xmask)
    tmp42 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = 0.92
    tmp2 = tmp0 * tmp1
    tmp3 = 0.85
    tmp4 = tmp2 * tmp3
    tmp5 = 9.81
    tmp6 = tmp4 * tmp5
    tmp7 = 0.5192307692307692
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 * tmp8
    tmp12 = tmp10 * tmp10
    tmp14 = 14.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp12 + tmp15
    tmp17 = tl.sqrt_rn(tmp16)
    tmp18 = triton_helpers.minimum(tmp11, tmp17)
    tmp19 = triton_helpers.maximum(tmp10, tmp18)
    tmp20 = tl_math.abs(tmp19)
    tmp21 = 20.0
    tmp22 = tmp20 * tmp21
    tmp23 = libdevice.tanh(tmp22)
    tmp24 = 0.1
    tmp25 = tmp23 * tmp24
    tmp26 = tmp19 * tmp19
    tmp27 = 0.01
    tmp28 = tmp26 * tmp27
    tmp29 = tmp25 + tmp28
    tmp30 = 0.5
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tmp9 - tmp32
    tmp34 = 1.0
    tmp35 = tmp0 * tmp34
    tmp36 = tmp35 * tmp3
    tmp37 = tmp36 * tmp5
    tmp38 = 0.4807692307692308
    tmp39 = tmp37 * tmp38
    tmp40 = tmp39 * tmp39
    tmp41 = tmp40 - tmp32
    tmp43 = 0.0
    tmp44 = triton_helpers.maximum(tmp33, tmp43)
    tmp45 = tl.sqrt_rn(tmp44)
    tmp46 = tmp26 * tmp7
    tmp47 = 1e-06
    tmp48 = triton_helpers.maximum(tmp46, tmp47)
    tmp49 = (tmp45 / tmp48)
    tmp50 = float("inf")
    tmp51 = triton_helpers.minimum(tmp50, tmp49)
    tmp52 = triton_helpers.maximum(tmp41, tmp43)
    tmp53 = tl.sqrt_rn(tmp52)
    tmp54 = tmp26 * tmp38
    tmp55 = triton_helpers.maximum(tmp54, tmp47)
    tmp56 = (tmp53 / tmp55)
    tmp57 = triton_helpers.minimum(tmp51, tmp56)
    tmp58 = triton_helpers.minimum(tmp42, tmp57)
    tmp59 = tmp46 * tmp58
    tmp60 = tmp59 * tmp59
    tmp61 = tmp32 + tmp60
    tmp62 = tmp2 * tmp5
    tmp63 = tmp62 * tmp7
    tmp64 = tmp63 * tmp63
    tmp65 = tmp61 - tmp64
    tmp66 = tmp54 * tmp58
    tmp67 = tmp66 * tmp66
    tmp68 = tmp32 + tmp67
    tmp69 = tmp35 * tmp5
    tmp70 = tmp69 * tmp38
    tmp71 = tmp70 * tmp70
    tmp72 = tmp68 - tmp71
    tmp73 = tmp60 - tmp64
    tmp74 = tmp67 - tmp71
    tmp75 = 0.25
    tmp76 = tmp29 * tmp75
    tmp77 = tmp2 * tmp2
    tmp78 = tmp77 * tmp5
    tmp79 = tmp78 * tmp7
    tmp80 = -0.22410660205935795
    tmp81 = tmp79 * tmp80
    tmp82 = tmp76 - tmp81
    tmp83 = tmp35 * tmp35
    tmp84 = tmp83 * tmp5
    tmp85 = tmp84 * tmp38
    tmp86 = 0.22410660205935795
    tmp87 = tmp85 * tmp86
    tmp88 = tmp76 - tmp87
    tmp89 = 2.0
    tmp90 = tmp82 * tmp89
    tmp91 = tmp90 * tmp90
    tmp92 = tmp2 * tmp80
    tmp93 = tmp92 * tmp92
    tmp94 = tmp75 - tmp93
    tmp95 = 4.0
    tmp96 = tmp94 * tmp95
    tmp97 = tmp96 * tmp65
    tmp98 = tmp91 - tmp97
    tmp99 = tmp98 >= tmp43
    tmp100 = triton_helpers.maximum(tmp98, tmp43)
    tmp101 = tl.sqrt_rn(tmp100)
    tmp102 = tmp90 + tmp101
    tmp103 = 1e-12
    tmp104 = tmp102 > tmp103
    tmp105 = tmp99 & tmp104
    tmp106 = -2.0
    tmp107 = tmp65 * tmp106
    tmp108 = triton_helpers.maximum(tmp102, tmp103)
    tmp109 = (tmp107 / tmp108)
    tmp110 = tl.where(tmp105, tmp109, tmp50)
    tmp111 = -tmp90
    tmp112 = tmp111 * tmp111
    tmp113 = tmp112 - tmp97
    tmp114 = tmp113 >= tmp43
    tmp115 = triton_helpers.maximum(tmp113, tmp43)
    tmp116 = tl.sqrt_rn(tmp115)
    tmp117 = tmp111 + tmp116
    tmp118 = tmp117 > tmp103
    tmp119 = tmp114 & tmp118
    tmp120 = triton_helpers.maximum(tmp117, tmp103)
    tmp121 = (tmp107 / tmp120)
    tmp122 = tl.where(tmp119, tmp121, tmp50)
    tmp123 = tmp88 * tmp89
    tmp124 = tmp123 * tmp123
    tmp125 = tmp35 * tmp86
    tmp126 = tmp125 * tmp125
    tmp127 = tmp75 - tmp126
    tmp128 = tmp127 * tmp95
    tmp129 = tmp128 * tmp72
    tmp130 = tmp124 - tmp129
    tmp131 = tmp130 >= tmp43
    tmp132 = triton_helpers.maximum(tmp130, tmp43)
    tmp133 = tl.sqrt_rn(tmp132)
    tmp134 = tmp123 + tmp133
    tmp135 = tmp134 > tmp103
    tmp136 = tmp131 & tmp135
    tmp137 = tmp72 * tmp106
    tmp138 = triton_helpers.maximum(tmp134, tmp103)
    tmp139 = (tmp137 / tmp138)
    tmp140 = tl.where(tmp136, tmp139, tmp50)
    tmp141 = -tmp123
    tmp142 = tmp141 * tmp141
    tmp143 = tmp142 - tmp129
    tmp144 = tmp143 >= tmp43
    tmp145 = triton_helpers.maximum(tmp143, tmp43)
    tmp146 = tl.sqrt_rn(tmp145)
    tmp147 = tmp141 + tmp146
    tmp148 = tmp147 > tmp103
    tmp149 = tmp144 & tmp148
    tmp150 = triton_helpers.maximum(tmp147, tmp103)
    tmp151 = (tmp137 / tmp150)
    tmp152 = tl.where(tmp149, tmp151, tmp50)
    tmp153 = tmp43 - tmp81
    tmp154 = tmp153 * tmp89
    tmp155 = tmp154 * tmp154
    tmp156 = tmp96 * tmp73
    tmp157 = tmp155 - tmp156
    tmp158 = tmp157 >= tmp43
    tmp159 = triton_helpers.maximum(tmp157, tmp43)
    tmp160 = tl.sqrt_rn(tmp159)
    tmp161 = tmp154 + tmp160
    tmp162 = tmp161 > tmp103
    tmp163 = tmp158 & tmp162
    tmp164 = tmp73 * tmp106
    tmp165 = triton_helpers.maximum(tmp161, tmp103)
    tmp166 = (tmp164 / tmp165)
    tmp167 = tmp43 - tmp87
    tmp168 = tmp167 * tmp89
    tmp169 = tmp168 * tmp168
    tmp170 = tmp128 * tmp74
    tmp171 = tmp169 - tmp170
    tmp172 = tmp171 >= tmp43
    tmp173 = triton_helpers.maximum(tmp171, tmp43)
    tmp174 = tl.sqrt_rn(tmp173)
    tmp175 = tmp168 + tmp174
    tmp176 = tmp175 > tmp103
    tmp177 = tmp172 & tmp176
    tmp178 = tmp74 * tmp106
    tmp179 = triton_helpers.maximum(tmp175, tmp103)
    tmp180 = (tmp178 / tmp179)
    tmp181 = -tmp154
    tmp182 = tmp181 * tmp181
    tmp183 = tmp182 - tmp156
    tmp184 = triton_helpers.maximum(tmp183, tmp43)
    tmp185 = tl.sqrt_rn(tmp184)
    tmp186 = tmp181 + tmp185
    tmp187 = tmp186 > tmp103
    tmp188 = triton_helpers.maximum(tmp186, tmp103)
    tmp189 = tmp183 >= tmp43
    tmp190 = tmp189 & tmp187
    tmp191 = (tmp164 / tmp188)
    tmp192 = tl.where(tmp190, tmp191, tmp50)
    tmp193 = -tmp168
    tmp194 = tmp193 * tmp193
    tmp195 = tmp194 - tmp170
    tmp196 = triton_helpers.maximum(tmp195, tmp43)
    tmp197 = tl.sqrt_rn(tmp196)
    tmp198 = tmp193 + tmp197
    tmp199 = tmp198 > tmp103
    tmp200 = triton_helpers.maximum(tmp198, tmp103)
    tmp201 = tmp195 >= tmp43
    tmp202 = tmp201 & tmp199
    tmp203 = (tmp178 / tmp200)
    tmp204 = tl.where(tmp202, tmp203, tmp50)
    tmp205 = 0.001
    tmp206 = triton_helpers.maximum(tmp20, tmp205)
    tmp207 = tl.full([1], 1, tl.int32)
    tmp208 = (tmp207 / tmp206)
    tmp209 = 7.319
    tmp210 = tmp208 * tmp209
    tmp211 = triton_helpers.minimum(tmp210, tmp34)
    tmp212 = 7.0
    tmp213 = tmp211 * tmp212
    tmp214 = tmp213 - tmp29
    tmp215 = tmp65 >= tmp43
    tmp216 = tl.where(tmp215, tmp43, tmp110)
    tmp217 = triton_helpers.minimum(tmp214, tmp216)
    tmp218 = tmp73 >= tmp43
    tmp219 = tl.where(tmp163, tmp166, tmp50)
    tmp220 = tl.where(tmp218, tmp43, tmp219)
    tmp221 = triton_helpers.minimum(tmp213, tmp220)
    tmp222 = 22.72870945945946
    tmp223 = triton_helpers.minimum(tmp221, tmp222)
    tmp224 = tmp11 * tmp11
    tmp225 = tmp224 - tmp12
    tmp226 = tmp13 * tmp89
    tmp227 = (tmp225 / tmp226)
    tmp228 = triton_helpers.minimum(tmp217, tmp222)
    tmp229 = tmp72 >= tmp43
    tmp230 = tl.where(tmp229, tmp43, tmp140)
    tmp231 = triton_helpers.minimum(tmp228, tmp230)
    tmp232 = tmp74 >= tmp43
    tmp233 = tl.where(tmp177, tmp180, tmp50)
    tmp234 = tl.where(tmp232, tmp43, tmp233)
    tmp235 = triton_helpers.minimum(tmp223, tmp234)
    tmp236 = triton_helpers.minimum(tmp231, tmp235)
    tmp237 = triton_helpers.minimum(tmp227, tmp236)
    tmp238 = -5.0
    tmp239 = tmp238 - tmp29
    tmp240 = tl.where(tmp215, tmp43, tmp122)
    tmp241 = -tmp240
    tmp242 = triton_helpers.maximum(tmp239, tmp241)
    tmp243 = tl.where(tmp229, tmp43, tmp152)
    tmp244 = -tmp243
    tmp245 = triton_helpers.maximum(tmp242, tmp244)
    tmp246 = -21.045101351351356
    tmp247 = triton_helpers.maximum(tmp245, tmp246)
    tmp248 = tl.where(tmp218, tmp43, tmp192)
    tmp249 = -tmp248
    tmp250 = triton_helpers.maximum(tmp238, tmp249)
    tmp251 = tl.where(tmp232, tmp43, tmp204)
    tmp252 = -tmp251
    tmp253 = triton_helpers.maximum(tmp250, tmp252)
    tmp254 = triton_helpers.maximum(tmp253, tmp246)
    tmp255 = triton_helpers.maximum(tmp247, tmp254)
    tmp256 = triton_helpers.maximum(tmp237, tmp255)
    tmp257 = tmp256 * tmp89
    tmp258 = tmp257 * tmp13
    tmp259 = tmp12 + tmp258
    tmp260 = triton_helpers.maximum(tmp259, tmp43)
    tmp261 = tl.sqrt_rn(tmp260)
    tl.store(in_out_ptr0 + (x0), tmp58, xmask)
    tl.store(in_out_ptr3 + (x0), tmp261, xmask)
