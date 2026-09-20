
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=34, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_atan_clamp_min_div_minimum_mul_pow_slice_sub_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'C4AE5D0BE9AAD72C2766AB07DD996C7AD9C23AFA1F39358C56BEF986E95FF01C', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_atan_clamp_min_div_minimum_mul_pow_slice_sub_unsqueeze_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (1 + x0), xmask)
    tmp5 = tl.load(in_ptr2 + (1 + x0), xmask)
    tmp7 = tl.load(in_ptr3 + (1 + x0), xmask)
    tmp14 = tl.load(in_ptr4 + (1 + x0), xmask)
    tmp17 = tl.load(in_ptr1 + (x0), xmask)
    tmp18 = tl.load(in_ptr2 + (x0), xmask)
    tmp20 = tl.load(in_ptr3 + (x0), xmask)
    tmp25 = tl.load(in_ptr4 + (x0), xmask)
    tmp2 = 3.2
    tmp3 = tmp1 * tmp2
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 * tmp8
    tmp10 = 0.003
    tmp11 = tmp9 * tmp10
    tmp12 = 0.3302
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = libdevice.atan(tmp15)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp21 * tmp21
    tmp23 = tmp22 * tmp10
    tmp24 = tmp23 + tmp12
    tmp26 = tmp24 * tmp25
    tmp27 = libdevice.atan(tmp26)
    tmp28 = tmp16 - tmp27
    tmp29 = tl_math.abs(tmp28)
    tmp30 = 1e-06
    tmp31 = triton_helpers.maximum(tmp29, tmp30)
    tmp32 = (tmp3 / tmp31)
    tl.store(out_ptr0 + (x0), tmp32, xmask)
