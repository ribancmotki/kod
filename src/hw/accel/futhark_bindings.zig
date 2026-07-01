pub const struct_futhark_context_config = opaque {};
pub const struct_futhark_context = opaque {};
pub const struct_futhark_f16_1d = opaque {};
pub const struct_futhark_f16_2d = opaque {};
pub const struct_futhark_f16_3d = opaque {};
pub const struct_futhark_f32_1d = opaque {};
pub const struct_futhark_f32_2d = opaque {};
pub const struct_futhark_f32_3d = opaque {};
pub const struct_futhark_u64_1d = opaque {};
pub const struct_futhark_i64_1d = opaque {};
pub const struct_futhark_opaque_tup9 = opaque {};
pub const struct_futhark_opaque_tup5_grad_full = opaque {};

pub extern "c" fn futhark_context_config_new() ?*struct_futhark_context_config;
pub extern "c" fn futhark_context_config_free(cfg: ?*struct_futhark_context_config) void;
pub extern "c" fn futhark_context_config_set_device(cfg: ?*struct_futhark_context_config, device: [*:0]const u8) void;
pub extern "c" fn futhark_context_config_set_platform(cfg: ?*struct_futhark_context_config, platform: c_int) void;
pub extern "c" fn futhark_context_config_set_default_group_size(cfg: ?*struct_futhark_context_config, size: c_int) void;
pub extern "c" fn futhark_context_config_set_default_num_groups(cfg: ?*struct_futhark_context_config, num: c_int) void;
pub extern "c" fn futhark_context_config_set_default_tile_size(cfg: ?*struct_futhark_context_config, size: c_int) void;

pub extern "c" fn futhark_context_new(cfg: ?*struct_futhark_context_config) ?*struct_futhark_context;
pub extern "c" fn futhark_context_free(ctx: ?*struct_futhark_context) void;
pub extern "c" fn futhark_context_sync(ctx: ?*struct_futhark_context) c_int;
pub extern "c" fn futhark_context_get_error(ctx: ?*struct_futhark_context) ?[*:0]const u8;
pub extern "c" fn futhark_context_clear_caches(ctx: ?*struct_futhark_context) c_int;

pub extern "c" fn futhark_new_f16_1d(ctx: ?*struct_futhark_context, data: ?[*]const u16, dim0: i64) ?*struct_futhark_f16_1d;
pub extern "c" fn futhark_new_f16_2d(ctx: ?*struct_futhark_context, data: ?[*]const u16, dim0: i64, dim1: i64) ?*struct_futhark_f16_2d;
pub extern "c" fn futhark_new_f16_3d(ctx: ?*struct_futhark_context, data: ?[*]const u16, dim0: i64, dim1: i64, dim2: i64) ?*struct_futhark_f16_3d;
pub extern "c" fn futhark_new_f16_2d_from_f32(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64) ?*struct_futhark_f16_2d;
pub extern "c" fn futhark_new_f16_3d_from_f32(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64, dim2: i64) ?*struct_futhark_f16_3d;

pub extern "c" fn futhark_free_f16_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_1d) c_int;
pub extern "c" fn futhark_free_f16_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d) c_int;
pub extern "c" fn futhark_free_f16_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_3d) c_int;

pub extern "c" fn futhark_values_f16_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_1d, data: ?[*]u16) c_int;
pub extern "c" fn futhark_values_f16_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d, data: ?[*]u16) c_int;
pub extern "c" fn futhark_values_f16_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_3d, data: ?[*]u16) c_int;
pub extern "c" fn futhark_values_f16_2d_to_f32(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_f16_3d_to_f32(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_3d, data: ?[*]f32) c_int;

pub extern "c" fn futhark_values_raw_f16_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_1d) ?*anyopaque;
pub extern "c" fn futhark_values_raw_f16_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d) ?*anyopaque;
pub extern "c" fn futhark_shape_f16_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_1d, dims: ?[*]i64) c_int;
pub extern "c" fn futhark_shape_f16_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d, dims: ?[*]i64) c_int;

pub extern "c" fn futhark_new_f32_1d(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64) ?*struct_futhark_f32_1d;
pub extern "c" fn futhark_new_f32_2d(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64) ?*struct_futhark_f32_2d;
pub extern "c" fn futhark_new_f32_3d(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64, dim2: i64) ?*struct_futhark_f32_3d;
pub extern "c" fn futhark_new_u64_1d(ctx: ?*struct_futhark_context, data: ?[*]const u64, dim0: i64) ?*struct_futhark_u64_1d;
pub extern "c" fn futhark_new_i64_1d(ctx: ?*struct_futhark_context, data: ?[*]const i64, dim0: i64) ?*struct_futhark_i64_1d;

pub extern "c" fn futhark_free_f32_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_1d) void;
pub extern "c" fn futhark_free_f32_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_2d) void;
pub extern "c" fn futhark_free_f32_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_3d) void;
pub extern "c" fn futhark_free_u64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_u64_1d) void;
pub extern "c" fn futhark_free_i64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_i64_1d) void;

pub extern "c" fn futhark_values_f32_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_1d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_f32_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_2d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_f32_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_3d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_u64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_u64_1d, data: ?[*]u64) c_int;
pub extern "c" fn futhark_values_i64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_i64_1d, data: ?[*]i64) c_int;

pub extern "c" fn futhark_entry_matmul(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_2d, a: ?*struct_futhark_f32_2d, b: ?*struct_futhark_f32_2d) c_int;
pub extern "c" fn futhark_entry_batch_matmul(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_3d, a: ?*struct_futhark_f32_3d, b: ?*struct_futhark_f32_3d) c_int;
pub extern "c" fn futhark_entry_dot(ctx: ?*struct_futhark_context, out: ?*f32, a: ?*struct_futhark_f32_1d, b: ?*struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_entry_clip_fisher(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, fisher: ?*struct_futhark_f32_1d, clip_val: f32) c_int;
pub extern "c" fn futhark_entry_reduce_gradients(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, gradients: ?*struct_futhark_f32_2d) c_int;
pub extern "c" fn futhark_entry_rank_segments(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, query_hash: u64, segment_hashes: ?*struct_futhark_u64_1d, base_scores: ?*struct_futhark_f32_1d) c_int;

pub extern "c" fn futhark_entry_rsf_forward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    input: ?*struct_futhark_f16_2d,
    weights_s: ?*struct_futhark_f16_2d,
    weights_t: ?*struct_futhark_f16_2d,
    s_bias: ?*struct_futhark_f16_1d,
    t_bias: ?*struct_futhark_f16_1d,
    clip_min: u16,
    clip_max: u16,
) c_int;

pub extern "c" fn futhark_entry_rsf_backward(
    ctx: ?*struct_futhark_context,
    out_grad_ws: ?*?*struct_futhark_f16_2d,
    out_grad_wt: ?*?*struct_futhark_f16_2d,
    out_grad_sb: ?*?*struct_futhark_f16_1d,
    out_grad_tb: ?*?*struct_futhark_f16_1d,
    input: ?*struct_futhark_f16_2d,
    grad_output: ?*struct_futhark_f16_2d,
    weights_s: ?*struct_futhark_f16_2d,
    weights_t: ?*struct_futhark_f16_2d,
    s_bias: ?*struct_futhark_f16_1d,
    t_bias: ?*struct_futhark_f16_1d,
    clip_min: u16,
    clip_max: u16,
) c_int;

pub extern "c" fn futhark_entry_scale_weights_inplace(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f16_2d, weights: ?*struct_futhark_f16_2d, scale: u16) c_int;

pub extern "c" fn futhark_free_opaque_tup9_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_f16(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup9,
) c_int;

pub extern "c" fn futhark_project_opaque_tup9_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_f16_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    obj: ?*const struct_futhark_opaque_tup9,
) c_int;

pub extern "c" fn futhark_project_opaque_tup9_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_f16_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    obj: ?*const struct_futhark_opaque_tup9,
) c_int;

pub extern "c" fn futhark_project_opaque_tup9_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_f16_2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_1d,
    obj: ?*const struct_futhark_opaque_tup9,
) c_int;

pub extern "c" fn futhark_project_opaque_tup9_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_f16_3(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_1d,
    obj: ?*const struct_futhark_opaque_tup9,
) c_int;

pub extern "c" fn futhark_project_opaque_tup9_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_f16_4(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    obj: ?*const struct_futhark_opaque_tup9,
) c_int;

pub extern "c" fn futhark_project_opaque_tup9_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_f16_5(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    obj: ?*const struct_futhark_opaque_tup9,
) c_int;

pub extern "c" fn futhark_project_opaque_tup9_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_f16_6(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_1d,
    obj: ?*const struct_futhark_opaque_tup9,
) c_int;

pub extern "c" fn futhark_project_opaque_tup9_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_f16_7(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_1d,
    obj: ?*const struct_futhark_opaque_tup9,
) c_int;

pub extern "c" fn futhark_project_opaque_tup9_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_f16_8(
    ctx: ?*struct_futhark_context,
    out: ?*u16,
    obj: ?*const struct_futhark_opaque_tup9,
) c_int;

pub extern "c" fn futhark_entry_training_step(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup9,
    in0_inputs: ?*const struct_futhark_f16_3d,
    in1_targets: ?*const struct_futhark_f16_3d,
    in2_weights_s: ?*const struct_futhark_f16_2d,
    in3_weights_t: ?*const struct_futhark_f16_2d,
    in4_s_bias: ?*const struct_futhark_f16_1d,
    in5_t_bias: ?*const struct_futhark_f16_1d,
    in6_velocity_s: ?*const struct_futhark_f16_2d,
    in7_velocity_t: ?*const struct_futhark_f16_2d,
    in8_velocity_sb: ?*const struct_futhark_f16_1d,
    in9_velocity_tb: ?*const struct_futhark_f16_1d,
    in10_lr: u16,
    in11_momentum: u16,
    in12_clip_min: u16,
    in13_clip_max: u16,
) c_int;

// === Multi-layer backprop primitives ===
// `batch_forward` returns the full-sample output tensor [batch][seq][half*2] so it can be cached
// as activations between layers.
pub extern "c" fn futhark_entry_batch_forward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    in0_inputs: ?*const struct_futhark_f16_3d,
    in1_weights_s: ?*const struct_futhark_f16_2d,
    in2_weights_t: ?*const struct_futhark_f16_2d,
    in3_s_bias: ?*const struct_futhark_f16_1d,
    in4_t_bias: ?*const struct_futhark_f16_1d,
    in5_clip_min: u16,
    in6_clip_max: u16,
) c_int;

// `batch_compute_loss` returns the scalar mean-squared-error loss in f16 (accumulated in f32).
pub extern "c" fn futhark_entry_batch_compute_loss(
    ctx: ?*struct_futhark_context,
    out: ?*u16,
    in0_outputs: ?*const struct_futhark_f16_3d,
    in1_targets: ?*const struct_futhark_f16_3d,
) c_int;

// `compute_initial_grad_l2` returns 2*(output-target) per element, used as the dL/dY seed at the
// top of the multi-layer stack.
pub extern "c" fn futhark_entry_compute_initial_grad_l2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    in0_outputs: ?*const struct_futhark_f16_3d,
    in1_targets: ?*const struct_futhark_f16_3d,
) c_int;

// `batch_gradients_full` returns (grad_ws, grad_wt, grad_sb, grad_tb, grad_input) as an opaque tup5.
// grad_input is [batch][seq][half*2] -- the dL/dX of the current layer, fed as dL/dY of the previous one.
pub extern "c" fn futhark_entry_batch_gradients_full(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup5_grad_full,
    in0_inputs: ?*const struct_futhark_f16_3d,
    in1_grad_outputs: ?*const struct_futhark_f16_3d,
    in2_weights_s: ?*const struct_futhark_f16_2d,
    in3_weights_t: ?*const struct_futhark_f16_2d,
    in4_s_bias: ?*const struct_futhark_f16_1d,
    in5_t_bias: ?*const struct_futhark_f16_1d,
    in6_clip_min: u16,
    in7_clip_max: u16,
) c_int;

pub extern "c" fn futhark_free_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup5_grad_full,
) c_int;

pub extern "c" fn futhark_project_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    obj: ?*const struct_futhark_opaque_tup5_grad_full,
) c_int;

pub extern "c" fn futhark_project_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    obj: ?*const struct_futhark_opaque_tup5_grad_full,
) c_int;

pub extern "c" fn futhark_project_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16_2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_1d,
    obj: ?*const struct_futhark_opaque_tup5_grad_full,
) c_int;

pub extern "c" fn futhark_project_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16_3(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_1d,
    obj: ?*const struct_futhark_opaque_tup5_grad_full,
) c_int;

pub extern "c" fn futhark_project_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16_4(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    obj: ?*const struct_futhark_opaque_tup5_grad_full,
) c_int;

// SFD updates per-layer (already exist as entry points in main.fut, declared here for completeness).
pub extern "c" fn futhark_entry_sfd_update_half(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup2_2d,
    in0_weights: ?*const struct_futhark_f16_2d,
    in1_gradients: ?*const struct_futhark_f16_2d,
    in2_lr: u16,
    in3_momentum: u16,
    in4_velocity: ?*const struct_futhark_f16_2d,
) c_int;

pub extern "c" fn futhark_entry_sfd_update_bias(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup2_1d,
    in0_bias: ?*const struct_futhark_f16_1d,
    in1_gradients: ?*const struct_futhark_f16_1d,
    in2_lr: u16,
    in3_momentum: u16,
    in4_velocity: ?*const struct_futhark_f16_1d,
) c_int;

pub const struct_futhark_opaque_tup2_2d = opaque {};
pub const struct_futhark_opaque_tup2_1d = opaque {};

pub extern "c" fn futhark_entry_oftb_forward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    inputs: ?*const struct_futhark_f16_3d,
) c_int;

pub extern "c" fn futhark_entry_oftb_backward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    grad_outputs: ?*const struct_futhark_f16_3d,
) c_int;

pub extern "c" fn futhark_entry_batch_oftb_forward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    inputs: ?*const struct_futhark_f16_3d,
) c_int;

pub extern "c" fn futhark_entry_batch_oftb_backward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    grad_outputs: ?*const struct_futhark_f16_3d,
) c_int;

pub extern "c" fn futhark_entry_embedding_forward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    tokens: ?*const struct_futhark_i64_1d,
    weight: ?*const struct_futhark_f16_2d,
) c_int;

pub extern "c" fn futhark_entry_embedding_backward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    tokens: ?*const struct_futhark_i64_1d,
    grad_output: ?*const struct_futhark_f16_2d,
    grad_weight: ?*const struct_futhark_f16_2d,
) c_int;

pub extern "c" fn futhark_entry_embedding_update(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    weight: ?*const struct_futhark_f16_2d,
    grad_weight: ?*const struct_futhark_f16_2d,
    lr: u16,
) c_int;

pub extern "c" fn futhark_free_opaque_tup2_arr2d_f16_arr2d_f16(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup2_2d,
) c_int;

pub extern "c" fn futhark_project_opaque_tup2_arr2d_f16_arr2d_f16_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    obj: ?*const struct_futhark_opaque_tup2_2d,
) c_int;

pub extern "c" fn futhark_project_opaque_tup2_arr2d_f16_arr2d_f16_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    obj: ?*const struct_futhark_opaque_tup2_2d,
) c_int;

pub extern "c" fn futhark_free_opaque_tup2_arr1d_f16_arr1d_f16(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup2_1d,
) c_int;

pub extern "c" fn futhark_project_opaque_tup2_arr1d_f16_arr1d_f16_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_1d,
    obj: ?*const struct_futhark_opaque_tup2_1d,
) c_int;

pub extern "c" fn futhark_project_opaque_tup2_arr1d_f16_arr1d_f16_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_1d,
    obj: ?*const struct_futhark_opaque_tup2_1d,
) c_int;

