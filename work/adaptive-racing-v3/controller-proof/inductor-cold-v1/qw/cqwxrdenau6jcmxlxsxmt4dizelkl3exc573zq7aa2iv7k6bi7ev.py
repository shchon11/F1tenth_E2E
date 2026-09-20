
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_le_maximum_minimum_neg_select_stack_sub_where_zeros_like_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_le_maximum_minimum_neg_select_stack_sub_where_zeros_like_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.where(tmp4, tmp6, 0.0)
    tmp8 = -tmp7
    tmp9 = tl.load(in_ptr1 + (4))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp11 = tl.where(tmp4, tmp10, 0.0)
    tmp12 = 0.16000000000000003
    tmp13 = tmp11 - tmp12
    tmp14 = triton_helpers.maximum(tmp8, tmp13)
    tmp15 = tmp11 + tmp12
    tmp16 = triton_helpers.minimum(tmp7, tmp15)
    tmp17 = tmp14 <= tmp16
    tmp18 = tl.load(in_ptr2 + (0))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp20 = tl.where(tmp4, tmp19, 0.0)
    tmp21 = triton_helpers.minimum(tmp20, tmp16)
    tmp22 = triton_helpers.maximum(tmp21, tmp14)
    tmp23 = 0.0
    tmp24 = triton_helpers.minimum(tmp23, tmp15)
    tmp25 = triton_helpers.maximum(tmp24, tmp13)
    tmp26 = -0.4189
    tmp27 = triton_helpers.maximum(tmp25, tmp26)
    tmp28 = 0.4189
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = tl.where(tmp17, tmp22, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp4, tmp30, tmp31)
    tmp33 = tmp0 >= tmp3
    tmp34 = tl.full([1], 2, tl.int64)
    tmp35 = tmp0 < tmp34
    tmp36 = tl.load(in_ptr2 + (1))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp38 = tl.where(tmp33, tmp37, 0.0)
    tmp39 = tl.load(in_ptr3 + (0))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp41 = tl.where(tmp33, tmp40, 0.0)
    tmp42 = triton_helpers.minimum(tmp38, tmp41)
    tmp43 = tl.load(in_ptr4 + (0))
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK])
    tmp45 = tl.where(tmp33, tmp44, 0.0)
    tmp46 = triton_helpers.maximum(tmp42, tmp45)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp33, tmp46, tmp47)
    tmp49 = tl.where(tmp4, tmp32, tmp48)
    tl.store(out_ptr0 + (x0), tmp49, xmask)
