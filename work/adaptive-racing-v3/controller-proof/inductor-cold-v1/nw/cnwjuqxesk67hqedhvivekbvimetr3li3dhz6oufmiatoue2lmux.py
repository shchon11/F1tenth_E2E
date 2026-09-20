
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_atan_clamp_min_div_minimum_mul_pow_slice_sub_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_atan_clamp_min_div_minimum_mul_pow_slice_sub_unsqueeze_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 24
    x0 = (xindex % 24)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (1 + x0 + 25*x1), xmask)
    tmp4 = tl.load(in_ptr2 + (1 + x0 + 25*x1), xmask)
    tmp6 = tl.load(in_ptr3 + (1 + x0 + 25*x1), xmask)
    tmp13 = tl.load(in_ptr4 + (1 + x0 + 25*x1), xmask)
    tmp16 = tl.load(in_ptr1 + (x0 + 25*x1), xmask)
    tmp17 = tl.load(in_ptr2 + (x0 + 25*x1), xmask)
    tmp19 = tl.load(in_ptr3 + (x0 + 25*x1), xmask)
    tmp24 = tl.load(in_ptr4 + (x0 + 25*x1), xmask)
    tmp1 = 3.2
    tmp2 = tmp0 * tmp1
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp7 = triton_helpers.minimum(tmp5, tmp6)
    tmp8 = tmp7 * tmp7
    tmp9 = 0.003
    tmp10 = tmp8 * tmp9
    tmp11 = 0.3302
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 * tmp13
    tmp15 = libdevice.atan(tmp14)
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp20 * tmp20
    tmp22 = tmp21 * tmp9
    tmp23 = tmp22 + tmp11
    tmp25 = tmp23 * tmp24
    tmp26 = libdevice.atan(tmp25)
    tmp27 = tmp15 - tmp26
    tmp28 = tl_math.abs(tmp27)
    tmp29 = 1e-06
    tmp30 = triton_helpers.maximum(tmp28, tmp29)
    tmp31 = (tmp2 / tmp30)
    tl.store(out_ptr0 + (x2), tmp31, xmask)
