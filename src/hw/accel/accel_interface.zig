const std = @import("std");
const cuda = @import("cuda_bindings.zig");
const futhark = @import("futhark_bindings.zig");
const core_tensor = @import("../../core/tensor.zig");
const core_memory = @import("../../core/memory.zig");

pub const gpu_enabled: bool = @import("build_options").gpu_acceleration;

pub const AccelError = error{
    FutharkConfigFailed,
    FutharkContextFailed,
    FutharkSyncFailed,
    FutharkArrayNewFailed,
    FutharkValuesFailed,
    FutharkForwardFailed,
    FutharkTrainingStepFailed,
    FutharkScaleWeightsFailed,
    FutharkShapeFailed,
    FutharkComputeLossFailed,
    FutharkBackwardFailed,
    FutharkSFDUpdateFailed,
    CudaHostAllocFailed,
    CudaFreeFailed,
    NullPointer,
    InvalidDimensions,
    AllocationFailed,
    PartialRowCleanup,
};

pub const WeightKind = enum {
    weights_s,
    weights_t,
    s_bias,
    t_bias,
    velocity_s,
    velocity_t,
    velocity_sb,
    velocity_tb,
};

pub const FutharkContext = struct {
    ctx: ?*futhark.struct_futhark_context,
    cfg: ?*futhark.struct_futhark_context_config,

    const Self = @This();

    pub fn init() AccelError!Self {
        const cfg = futhark.futhark_context_config_new();
        if (cfg == null) return AccelError.FutharkConfigFailed;

        // These knobs only exist in GPU backends (cuda/opencl/hip). The sequential
        // C backend used by the inference server and CPU benchmarks does not
        // define them, so we gate them behind the comptime gpu_enabled flag.
        if (comptime gpu_enabled) {
            futhark.futhark_context_config_set_device(cfg, "");
            futhark.futhark_context_config_set_default_group_size(cfg, 256);
            futhark.futhark_context_config_set_default_num_groups(cfg, 128);
            futhark.futhark_context_config_set_default_tile_size(cfg, 32);
        }

        const ctx = futhark.futhark_context_new(cfg);
        if (ctx == null) {
            futhark.futhark_context_config_free(cfg);
            return AccelError.FutharkContextFailed;
        }

        if (futhark.futhark_context_sync(ctx) != 0) {
            futhark.futhark_context_free(ctx);
            futhark.futhark_context_config_free(cfg);
            return AccelError.FutharkSyncFailed;
        }

        return Self{ .ctx = ctx, .cfg = cfg };
    }

    pub fn deinit(self: *Self) void {
        if (self.ctx) |ctx| {
            const err_str = futhark.futhark_context_get_error(ctx);
            if (err_str != null) {
                _ = futhark.futhark_context_clear_caches(ctx);
            }
            futhark.futhark_context_free(ctx);
            self.ctx = null;
        }
        if (self.cfg) |cfg| {
            futhark.futhark_context_config_free(cfg);
            self.cfg = null;
        }
    }

    pub fn sync(self: *Self) AccelError!void {
        if (self.ctx == null) return AccelError.NullPointer;
        if (futhark.futhark_context_sync(self.ctx) != 0) {
            return AccelError.FutharkSyncFailed;
        }
    }

    pub fn getDataPointer(self: *Self, array: *FutharkArray2DF16) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;

        const raw_ptr = futhark.futhark_values_raw_f16_2d(self.ctx, array.arr);
        if (raw_ptr == null) {
            return AccelError.NullPointer;
        }

        return raw_ptr.?;
    }

    pub fn getDataPointer1D(self: *Self, array: *FutharkArray1DF16) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;

        const raw_ptr = futhark.futhark_values_raw_f16_1d(self.ctx, array.arr);
        if (raw_ptr == null) {
            return AccelError.NullPointer;
        }

        return raw_ptr.?;
    }
};

fn get1DDevicePtr(ctx: *FutharkContext, array: *FutharkArray1DF16) AccelError!*anyopaque {
    return ctx.getDataPointer1D(array);
}

pub const PinnedMemory = struct {
    ptr: ?*anyopaque,
    size: usize,

    const Self = @This();

    pub fn alloc(size: usize) AccelError!Self {
        if (size == 0) {
            return Self{ .ptr = null, .size = 0 };
        }

        var ptr: ?*anyopaque = null;
        const err = cuda.cudaHostAlloc(&ptr, size, cuda.cudaHostAllocDefault);
        if (err != cuda.cudaSuccess) {
            return AccelError.CudaHostAllocFailed;
        }

        return Self{
            .ptr = ptr,
            .size = size,
        };
    }

    pub fn free(self: *Self) void {
        if (self.ptr) |p| {
            _ = cuda.cudaFreeHost(p);
            self.ptr = null;
            self.size = 0;
        }
    }

    pub fn asSlice(self: *Self, comptime T: type) ?[]T {
        if (self.ptr == null) return null;
        if (self.size == 0) return &[_]T{};
        const count = self.size / @sizeOf(T);
        if (count == 0) return &[_]T{};
        const aligned: [*]T = @ptrCast(@alignCast(self.ptr.?));
        return aligned[0..count];
    }
};

pub const FutharkArray1DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_1d,
    len: usize,

    const Self = @This();

    pub fn newFromFlat(ctx: *FutharkContext, flat_data: []const f16, length: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (length == 0) return AccelError.InvalidDimensions;
        if (flat_data.len != length) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_1d(
            ctx.ctx,
            @ptrCast(flat_data.ptr),
            @intCast(length),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .len = length };
    }

    pub fn newZeros(ctx: *FutharkContext, length: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (length == 0) return AccelError.InvalidDimensions;

        const zeros = allocator.alloc(f16, length) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);

        const arr = futhark.futhark_new_f16_1d(
            ctx.ctx,
            @ptrCast(zeros.ptr),
            @intCast(length),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .len = length };
    }

    pub fn values1D(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f16 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        if (self.len == 0) return AccelError.InvalidDimensions;

        const buf = allocator.alloc(f16, self.len) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);

        const result = futhark.futhark_values_f16_1d(ctx.ctx, self.arr, @ptrCast(buf.ptr));
        if (result != 0) {
            allocator.free(buf);
            return AccelError.FutharkValuesFailed;
        }

        const sync_result = futhark.futhark_context_sync(ctx.ctx);
        if (sync_result != 0) {
            allocator.free(buf);
            return AccelError.FutharkSyncFailed;
        }

        return buf;
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }
};

pub const FutharkArray2DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn new(ctx: *FutharkContext, data: []const []const f16) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;

        const rows = data.len;
        const cols = data[0].len;
        if (cols == 0) return AccelError.InvalidDimensions;

        for (data) |row| {
            if (row.len != cols) return AccelError.InvalidDimensions;
        }

        const total = rows * cols;
        var flat_data = std.ArrayList(f16).init(std.heap.page_allocator);
        defer flat_data.deinit();

        flat_data.ensureTotalCapacity(total) catch return AccelError.AllocationFailed;

        for (data) |row| {
            flat_data.appendSlice(row) catch return AccelError.AllocationFailed;
        }

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(flat_data.items.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newFromFlat(ctx: *FutharkContext, flat_data: []const f16, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (flat_data.len != rows * cols) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(flat_data.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;

        const total = rows * cols;
        const zeros = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(zeros.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_2d(ctx.ctx, arr);
            self.arr = null;
            self.rows = 0;
            self.cols = 0;
        }
    }

    pub fn values(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![][]f16 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;

        const rows = self.rows;
        const cols = self.cols;

        if (rows == 0 or cols == 0) {
            return allocator.alloc([]f16, 0) catch return AccelError.AllocationFailed;
        }

        const flat = allocator.alloc(f16, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(flat);

        if (futhark.futhark_values_f16_2d(ctx.ctx, self.arr, @ptrCast(flat.ptr)) != 0) {
            return AccelError.FutharkValuesFailed;
        }

        const result = allocator.alloc([]f16, rows) catch return AccelError.AllocationFailed;
        var i: usize = 0;
        while (i < rows) : (i += 1) {
            result[i] = allocator.alloc(f16, cols) catch {
                var j: usize = 0;
                while (j < i) : (j += 1) {
                    allocator.free(result[j]);
                }
                allocator.free(result);
                return AccelError.PartialRowCleanup;
            };
            @memcpy(result[i], flat[i * cols .. (i + 1) * cols]);
        }

        return result;
    }
};

pub const FutharkArray3DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_3d,
    dim0: usize,
    dim1: usize,
    dim2: usize,

    const Self = @This();

    pub fn newFromFlat(ctx: *FutharkContext, flat: []const f16, d0: usize, d1: usize, d2: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (d0 == 0 or d1 == 0 or d2 == 0) return AccelError.InvalidDimensions;
        if (flat.len != d0 * d1 * d2) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_3d(
            ctx.ctx,
            @ptrCast(flat.ptr),
            @intCast(d0),
            @intCast(d1),
            @intCast(d2),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .dim0 = d0, .dim1 = d1, .dim2 = d2 };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_3d(ctx.ctx, arr);
            self.arr = null;
            self.dim0 = 0;
            self.dim1 = 0;
            self.dim2 = 0;
        }
    }
};

pub const FutharkArray2DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (tensor.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = tensor.shape.dims[0];
        const cols = tensor.shape.dims[1];
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, tensor.data.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newFromFlat(ctx: *FutharkContext, data: []const f32, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, data.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const zeros = allocator.alloc(f32, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, zeros.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            futhark.futhark_free_f32_2d(ctx.ctx, arr);
            self.arr = null;
            self.rows = 0;
            self.cols = 0;
        }
    }

    pub fn toTensor(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const shape = [_]usize{ self.rows, self.cols };
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_2d(ctx.ctx, self.arr, tensor.data.ptr) != 0) {
            tensor.deinit();
            return AccelError.FutharkValuesFailed;
        }
        return tensor;
    }
};

pub const FutharkArray1DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_1d,
    len: usize,

    const Self = @This();

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (tensor.shape.dims.len != 1) return AccelError.InvalidDimensions;
        const n = tensor.shape.dims[0];
        if (n == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_1d(ctx.ctx, tensor.data.ptr, @intCast(n));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = n };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            futhark.futhark_free_f32_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }

    pub fn toTensor(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const shape = [_]usize{self.len};
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_1d(ctx.ctx, self.arr, tensor.data.ptr) != 0) {
            tensor.deinit();
            return AccelError.FutharkValuesFailed;
        }
        return tensor;
    }
};

// One RSF layer = (W_s, W_t, s_bias, t_bias) + their SFD momentum buffers.
// Stored as opaque Futhark GPU handles.
pub const RSFLayer = struct {
    weights_s: FutharkArray2DF16,
    weights_t: FutharkArray2DF16,
    s_bias: FutharkArray1DF16,
    t_bias: FutharkArray1DF16,
    velocity_s: FutharkArray2DF16,
    velocity_t: FutharkArray2DF16,
    velocity_sb: FutharkArray1DF16,
    velocity_tb: FutharkArray1DF16,

    pub fn free(self: *RSFLayer, ctx: *FutharkContext) void {
        self.velocity_tb.free(ctx);
        self.velocity_sb.free(ctx);
        self.velocity_t.free(ctx);
        self.velocity_s.free(ctx);
        self.t_bias.free(ctx);
        self.s_bias.free(ctx);
        self.weights_t.free(ctx);
        self.weights_s.free(ctx);
    }
};

pub const RSFAccelerator = struct {
    ctx: FutharkContext,
    layers: []RSFLayer,
    layers_owner: std.mem.Allocator,
    model_dim: usize,
    num_layers: usize,
    clip_min: f16,
    clip_max: f16,
    initialized: bool,

    const Self = @This();

    pub fn init(model_dim: usize) AccelError!Self {
        return initMultiLayer(model_dim, 1, std.heap.page_allocator);
    }

    pub fn initMultiLayer(model_dim: usize, num_layers: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (model_dim == 0) return AccelError.InvalidDimensions;
        if (model_dim % 2 != 0) return AccelError.InvalidDimensions;
        if (num_layers == 0) return AccelError.InvalidDimensions;
        const half: usize = model_dim / 2;

        var ctx = try FutharkContext.init();
        errdefer ctx.deinit();

        // All ranks use the SAME deterministic seed so that the model is
        // consistent across the cluster BEFORE any all-reduce happens.
        // Per-layer seeds are derived from a base seed and the layer index
        // to break symmetry across layers (otherwise every layer would
        // initialize to the same values).
        const base_seed: u64 = 0x4A41494445204E4F; // "JAIDE NO"
        const init_stddev: f32 = 0.02;

        var layers = allocator.alloc(RSFLayer, num_layers) catch return AccelError.AllocationFailed;
        errdefer allocator.free(layers);

        const total: usize = half * half;
        const ws_buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer allocator.free(ws_buf);
        const wt_buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer allocator.free(wt_buf);

        var layers_built: usize = 0;
        errdefer {
            var idx: usize = 0;
            while (idx < layers_built) : (idx += 1) {
                layers[idx].free(&ctx);
            }
        }

        var layer_idx: usize = 0;
        while (layer_idx < num_layers) : (layer_idx += 1) {
            const layer_seed: u64 = base_seed +% (@as(u64, @intCast(layer_idx)) *% 0x9E3779B97F4A7C15);
            var rng = std.Random.DefaultPrng.init(layer_seed);
            const rnd = rng.random();
            for (ws_buf) |*v| {
                const r = rnd.floatNorm(f32) * init_stddev;
                v.* = @floatCast(r);
            }
            for (wt_buf) |*v| {
                const r = rnd.floatNorm(f32) * init_stddev;
                v.* = @floatCast(r);
            }

            const weights_s = try FutharkArray2DF16.newFromFlat(&ctx, ws_buf, half, half);
            const weights_t = try FutharkArray2DF16.newFromFlat(&ctx, wt_buf, half, half);
            const s_bias = try FutharkArray1DF16.newZeros(&ctx, half, allocator);
            const t_bias = try FutharkArray1DF16.newZeros(&ctx, half, allocator);
            const velocity_s = try FutharkArray2DF16.newZeros(&ctx, half, half, allocator);
            const velocity_t = try FutharkArray2DF16.newZeros(&ctx, half, half, allocator);
            const velocity_sb = try FutharkArray1DF16.newZeros(&ctx, half, allocator);
            const velocity_tb = try FutharkArray1DF16.newZeros(&ctx, half, allocator);

            layers[layer_idx] = .{
                .weights_s = weights_s,
                .weights_t = weights_t,
                .s_bias = s_bias,
                .t_bias = t_bias,
                .velocity_s = velocity_s,
                .velocity_t = velocity_t,
                .velocity_sb = velocity_sb,
                .velocity_tb = velocity_tb,
            };
            layers_built += 1;
        }

        return Self{
            .ctx = ctx,
            .layers = layers,
            .layers_owner = allocator,
            .model_dim = model_dim,
            .num_layers = num_layers,
            .clip_min = @as(f16, -2.0),
            .clip_max = @as(f16, 2.0),
            .initialized = true,
        };
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;

        var i: usize = self.layers.len;
        while (i > 0) {
            i -= 1;
            self.layers[i].free(&self.ctx);
        }
        self.layers_owner.free(self.layers);
        self.ctx.deinit();
        self.initialized = false;
    }

    pub fn forward(self: *Self, input: *FutharkArray2DF16) AccelError!FutharkArray2DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (input.arr == null) return AccelError.NullPointer;
        if (self.layers.len == 0) return AccelError.NullPointer;

        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        // Chain through layers: current_arr is the live 2D activation.
        // We allocate fresh outputs from rsf_forward and free intermediates as we go.
        var current_arr: ?*futhark.struct_futhark_f16_2d = input.arr;
        const rows = input.rows;
        const cols = input.cols;

        var li: usize = 0;
        while (li < self.layers.len) : (li += 1) {
            const layer = &self.layers[li];
            if (layer.weights_s.arr == null or layer.weights_t.arr == null) return AccelError.NullPointer;
            if (layer.s_bias.arr == null or layer.t_bias.arr == null) return AccelError.NullPointer;

            var next_arr: ?*futhark.struct_futhark_f16_2d = null;
            const result = futhark.futhark_entry_rsf_forward(
                self.ctx.ctx,
                &next_arr,
                current_arr,
                layer.weights_s.arr,
                layer.weights_t.arr,
                layer.s_bias.arr,
                layer.t_bias.arr,
                clip_min_bits,
                clip_max_bits,
            );
            if (result != 0) {
                if (li > 0) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, current_arr);
                return AccelError.FutharkForwardFailed;
            }
            if (next_arr == null) {
                if (li > 0) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, current_arr);
                return AccelError.NullPointer;
            }

            // Free previous intermediate (but never the caller's input).
            if (li > 0) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, current_arr);
            current_arr = next_arr;
        }

        return FutharkArray2DF16{ .arr = current_arr, .rows = rows, .cols = cols };
    }

    // Multi-layer training step.
    // 1) Forward: run batch_forward through each layer, caching every intermediate
    //    3D activation tensor on the GPU.
    // 2) Loss: batch_compute_loss(final_output, target) (accumulator is f32).
    // 3) Initial gradient: 2*(final_output - target) via compute_initial_grad_l2.
    // 4) Backward: for each layer from N-1 down to 0, call batch_gradients_full to get
    //    (grad_ws, grad_wt, grad_sb, grad_tb, grad_input). Apply sfd_update_half /
    //    sfd_update_bias to the layer's weights, and feed grad_input back as the
    //    grad_output of the previous layer.
    pub fn trainingStep(
        self: *Self,
        inputs: *FutharkArray3DF16,
        targets: *FutharkArray3DF16,
        learning_rate: f16,
        momentum: f16,
    ) AccelError!f16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (inputs.arr == null or targets.arr == null) return AccelError.NullPointer;
        if (self.layers.len == 0) return AccelError.NullPointer;

        const lr_bits: u16 = @bitCast(learning_rate);
        const momentum_bits: u16 = @bitCast(momentum);
        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        const n_layers = self.layers.len;
        const step_alloc = std.heap.page_allocator;
        // activations[0] = inputs (caller-owned), activations[1..n_layers] = layer outputs (we own).
        var activations = step_alloc.alloc(?*futhark.struct_futhark_f16_3d, n_layers + 1) catch return AccelError.AllocationFailed;
        defer step_alloc.free(activations);
        var owned = step_alloc.alloc(bool, n_layers + 1) catch return AccelError.AllocationFailed;
        defer step_alloc.free(owned);
        for (activations) |*a| a.* = null;
        for (owned) |*o| o.* = false;
        activations[0] = inputs.arr;
        owned[0] = false;

        // free all owned activations on early exit
        var early_err: ?AccelError = null;
        errdefer {
            var idx: usize = 0;
            while (idx < activations.len) : (idx += 1) {
                if (owned[idx] and activations[idx] != null) {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, activations[idx]);
                }
            }
        }

        // ----- Forward pass -----
        var li: usize = 0;
        while (li < n_layers) : (li += 1) {
            const layer = &self.layers[li];
            if (layer.weights_s.arr == null or layer.weights_t.arr == null) return AccelError.NullPointer;
            if (layer.s_bias.arr == null or layer.t_bias.arr == null) return AccelError.NullPointer;

            var next_act: ?*futhark.struct_futhark_f16_3d = null;
            const rc = futhark.futhark_entry_batch_forward(
                self.ctx.ctx,
                &next_act,
                activations[li],
                layer.weights_s.arr,
                layer.weights_t.arr,
                layer.s_bias.arr,
                layer.t_bias.arr,
                clip_min_bits,
                clip_max_bits,
            );
            if (rc != 0 or next_act == null) {
                const err_str = futhark.futhark_context_get_error(self.ctx.ctx);
                if (err_str) |s| std.debug.print("[Futhark batch_forward L{d} error] {s}\n", .{ li, std.mem.span(s) });
                early_err = AccelError.FutharkForwardFailed;
                return early_err.?;
            }
            // OFTB butterfly mixing after RSF coupling layer
            var oftb_out: ?*futhark.struct_futhark_f16_3d = null;
            const oftb_rc = futhark.futhark_entry_batch_oftb_forward(
                self.ctx.ctx,
                &oftb_out,
                next_act,
            );
            if (oftb_rc != 0 or oftb_out == null) {
                early_err = AccelError.FutharkForwardFailed;
                return early_err.?;
            }
            // Replace next_act with OFTB output
            _ = futhark.futhark_free_f16_3d(self.ctx.ctx, next_act);
            next_act = oftb_out;
            activations[li + 1] = next_act;
            owned[li + 1] = true;
        }

        // ----- Loss on final output -----
        var loss_bits: u16 = 0;
        const loss_rc = futhark.futhark_entry_batch_compute_loss(
            self.ctx.ctx,
            &loss_bits,
            activations[n_layers],
            targets.arr,
        );
        if (loss_rc != 0) {
            const err_str = futhark.futhark_context_get_error(self.ctx.ctx);
            if (err_str) |s| std.debug.print("[Futhark batch_compute_loss error] {s}\n", .{std.mem.span(s)});
            return AccelError.FutharkComputeLossFailed;
        }
        const loss_f16: f16 = @bitCast(loss_bits);

        // ----- Initial gradient seed: dL/dY_final = 2*(Y_final - target) -----
        var grad_out: ?*futhark.struct_futhark_f16_3d = null;
        const gseed_rc = futhark.futhark_entry_compute_initial_grad_l2(
            self.ctx.ctx,
            &grad_out,
            activations[n_layers],
            targets.arr,
        );
        if (gseed_rc != 0 or grad_out == null) {
            const err_str = futhark.futhark_context_get_error(self.ctx.ctx);
            if (err_str) |s| std.debug.print("[Futhark initial_grad_l2 error] {s}\n", .{std.mem.span(s)});
            return AccelError.FutharkBackwardFailed;
        }

        // ----- Backward pass (top-down) -----
        var lb: usize = n_layers;
        while (lb > 0) {
            lb -= 1;
            const layer = &self.layers[lb];

            // OFTB backward: transform gradient through butterfly mixing
            var oftb_grad: ?*futhark.struct_futhark_f16_3d = null;
            const oftb_bwd_rc = futhark.futhark_entry_batch_oftb_backward(
                self.ctx.ctx,
                &oftb_grad,
                grad_out,
            );
            _ = futhark.futhark_free_f16_3d(self.ctx.ctx, grad_out);
            grad_out = null;
            if (oftb_bwd_rc != 0 or oftb_grad == null) {
                return AccelError.FutharkBackwardFailed;
            }
            grad_out = oftb_grad;

            // batch_gradients_full(activations[lb], grad_out, layer weights)
            var grad_tup: ?*futhark.struct_futhark_opaque_tup5_grad_full = null;
            const bg_rc = futhark.futhark_entry_batch_gradients_full(
                self.ctx.ctx,
                &grad_tup,
                activations[lb],
                grad_out,
                layer.weights_s.arr,
                layer.weights_t.arr,
                layer.s_bias.arr,
                layer.t_bias.arr,
                clip_min_bits,
                clip_max_bits,
            );
            // grad_out is consumed by Futhark (its dL/dY for this layer); free our handle.
            _ = futhark.futhark_free_f16_3d(self.ctx.ctx, grad_out);
            grad_out = null;

            if (bg_rc != 0 or grad_tup == null) {
                const err_str = futhark.futhark_context_get_error(self.ctx.ctx);
                if (err_str) |s| std.debug.print("[Futhark batch_gradients_full L{d} error] {s}\n", .{ lb, std.mem.span(s) });
                return AccelError.FutharkBackwardFailed;
            }

            var grad_ws: ?*futhark.struct_futhark_f16_2d = null;
            var grad_wt: ?*futhark.struct_futhark_f16_2d = null;
            var grad_sb: ?*futhark.struct_futhark_f16_1d = null;
            var grad_tb: ?*futhark.struct_futhark_f16_1d = null;
            var grad_in: ?*futhark.struct_futhark_f16_3d = null;

            const proj0 = futhark.futhark_project_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16_0(self.ctx.ctx, &grad_ws, grad_tup);
            const proj1 = futhark.futhark_project_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16_1(self.ctx.ctx, &grad_wt, grad_tup);
            const proj2 = futhark.futhark_project_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16_2(self.ctx.ctx, &grad_sb, grad_tup);
            const proj3 = futhark.futhark_project_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16_3(self.ctx.ctx, &grad_tb, grad_tup);
            const proj4 = futhark.futhark_project_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16_4(self.ctx.ctx, &grad_in, grad_tup);
            _ = futhark.futhark_free_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16(self.ctx.ctx, grad_tup);

            if (proj0 != 0 or proj1 != 0 or proj2 != 0 or proj3 != 0 or proj4 != 0 or
                grad_ws == null or grad_wt == null or grad_sb == null or grad_tb == null or grad_in == null) {
                if (grad_ws != null) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, grad_ws);
                if (grad_wt != null) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, grad_wt);
                if (grad_sb != null) _ = futhark.futhark_free_f16_1d(self.ctx.ctx, grad_sb);
                if (grad_tb != null) _ = futhark.futhark_free_f16_1d(self.ctx.ctx, grad_tb);
                if (grad_in != null) _ = futhark.futhark_free_f16_3d(self.ctx.ctx, grad_in);
                return AccelError.FutharkBackwardFailed;
            }

            // ----- SFD updates for this layer's W_s, W_t, s_bias, t_bias -----
            try sfdUpdateMat(self, &layer.weights_s, &layer.velocity_s, grad_ws, lr_bits, momentum_bits);
            try sfdUpdateMat(self, &layer.weights_t, &layer.velocity_t, grad_wt, lr_bits, momentum_bits);
            try sfdUpdateBias(self, &layer.s_bias, &layer.velocity_sb, grad_sb, lr_bits, momentum_bits);
            try sfdUpdateBias(self, &layer.t_bias, &layer.velocity_tb, grad_tb, lr_bits, momentum_bits);

            _ = futhark.futhark_free_f16_2d(self.ctx.ctx, grad_ws);
            _ = futhark.futhark_free_f16_2d(self.ctx.ctx, grad_wt);
            _ = futhark.futhark_free_f16_1d(self.ctx.ctx, grad_sb);
            _ = futhark.futhark_free_f16_1d(self.ctx.ctx, grad_tb);

            // grad_in becomes dL/dY for the previous (lower) layer.
            grad_out = grad_in;
        }

        // Discard final grad_in (it is dL/d(model_input), not needed for training).
        if (grad_out != null) {
            _ = futhark.futhark_free_f16_3d(self.ctx.ctx, grad_out);
            grad_out = null;
        }

        // Free intermediate activations (everything except the caller's inputs).
        var fi: usize = 0;
        while (fi < activations.len) : (fi += 1) {
            if (owned[fi] and activations[fi] != null) {
                _ = futhark.futhark_free_f16_3d(self.ctx.ctx, activations[fi]);
                activations[fi] = null;
                owned[fi] = false;
            }
        }

        return loss_f16;
    }

    fn sfdUpdateMat(
        self: *Self,
        weights: *FutharkArray2DF16,
        velocity: *FutharkArray2DF16,
        gradients: ?*futhark.struct_futhark_f16_2d,
        lr_bits: u16,
        momentum_bits: u16,
    ) AccelError!void {
        if (weights.arr == null or velocity.arr == null) return AccelError.NullPointer;
        var out_tup: ?*futhark.struct_futhark_opaque_tup2_2d = null;
        const rc = futhark.futhark_entry_sfd_update_half(
            self.ctx.ctx,
            &out_tup,
            weights.arr,
            gradients,
            lr_bits,
            momentum_bits,
            velocity.arr,
        );
        if (rc != 0 or out_tup == null) {
            const err_str = futhark.futhark_context_get_error(self.ctx.ctx);
            if (err_str) |s| std.debug.print("[Futhark sfd_update_half error] {s}\n", .{std.mem.span(s)});
            return AccelError.FutharkSFDUpdateFailed;
        }
        var new_w: ?*futhark.struct_futhark_f16_2d = null;
        var new_v: ?*futhark.struct_futhark_f16_2d = null;
        _ = futhark.futhark_project_opaque_tup2_arr2d_f16_arr2d_f16_0(self.ctx.ctx, &new_w, out_tup);
        _ = futhark.futhark_project_opaque_tup2_arr2d_f16_arr2d_f16_1(self.ctx.ctx, &new_v, out_tup);
        _ = futhark.futhark_free_opaque_tup2_arr2d_f16_arr2d_f16(self.ctx.ctx, out_tup);
        if (new_w == null or new_v == null) return AccelError.FutharkSFDUpdateFailed;
        const old_w = weights.arr;
        const old_v = velocity.arr;
        weights.arr = new_w;
        velocity.arr = new_v;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_w);
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_v);
    }

    fn sfdUpdateBias(
        self: *Self,
        bias: *FutharkArray1DF16,
        velocity: *FutharkArray1DF16,
        gradients: ?*futhark.struct_futhark_f16_1d,
        lr_bits: u16,
        momentum_bits: u16,
    ) AccelError!void {
        if (bias.arr == null or velocity.arr == null) return AccelError.NullPointer;
        var out_tup: ?*futhark.struct_futhark_opaque_tup2_1d = null;
        const rc = futhark.futhark_entry_sfd_update_bias(
            self.ctx.ctx,
            &out_tup,
            bias.arr,
            gradients,
            lr_bits,
            momentum_bits,
            velocity.arr,
        );
        if (rc != 0 or out_tup == null) {
            const err_str = futhark.futhark_context_get_error(self.ctx.ctx);
            if (err_str) |s| std.debug.print("[Futhark sfd_update_bias error] {s}\n", .{std.mem.span(s)});
            return AccelError.FutharkSFDUpdateFailed;
        }
        var new_b: ?*futhark.struct_futhark_f16_1d = null;
        var new_v: ?*futhark.struct_futhark_f16_1d = null;
        _ = futhark.futhark_project_opaque_tup2_arr1d_f16_arr1d_f16_0(self.ctx.ctx, &new_b, out_tup);
        _ = futhark.futhark_project_opaque_tup2_arr1d_f16_arr1d_f16_1(self.ctx.ctx, &new_v, out_tup);
        _ = futhark.futhark_free_opaque_tup2_arr1d_f16_arr1d_f16(self.ctx.ctx, out_tup);
        if (new_b == null or new_v == null) return AccelError.FutharkSFDUpdateFailed;
        const old_b = bias.arr;
        const old_v = velocity.arr;
        bias.arr = new_b;
        velocity.arr = new_v;
        _ = futhark.futhark_free_f16_1d(self.ctx.ctx, old_b);
        _ = futhark.futhark_free_f16_1d(self.ctx.ctx, old_v);
    }

    pub fn scaleWeights(self: *Self, scale_factor: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (scale_factor == @as(f16, 0.0)) return AccelError.InvalidDimensions;

        const scale_bits: u16 = @bitCast(scale_factor);
        for (self.layers) |*layer| {
            if (layer.weights_s.arr == null or layer.weights_t.arr == null) return AccelError.NullPointer;

            var new_ws: ?*futhark.struct_futhark_f16_2d = null;
            const result_s = futhark.futhark_entry_scale_weights_inplace(
                self.ctx.ctx,
                &new_ws,
                layer.weights_s.arr,
                scale_bits,
            );
            if (result_s != 0) return AccelError.FutharkScaleWeightsFailed;
            if (new_ws != null) {
                const old = layer.weights_s.arr;
                layer.weights_s.arr = new_ws;
                _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
            }

            var new_wt: ?*futhark.struct_futhark_f16_2d = null;
            const result_t = futhark.futhark_entry_scale_weights_inplace(
                self.ctx.ctx,
                &new_wt,
                layer.weights_t.arr,
                scale_bits,
            );
            if (result_t != 0) return AccelError.FutharkScaleWeightsFailed;
            if (new_wt != null) {
                const old = layer.weights_t.arr;
                layer.weights_t.arr = new_wt;
                _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
            }
        }
    }

    pub fn sync(self: *Self) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        return self.ctx.sync();
    }

    pub fn numLayers(self: *const Self) usize {
        return self.num_layers;
    }

    pub fn layerPtr(self: *Self, layer_idx: usize) AccelError!*RSFLayer {
        if (!self.initialized) return AccelError.NullPointer;
        if (layer_idx >= self.layers.len) return AccelError.InvalidDimensions;
        return &self.layers[layer_idx];
    }

    pub fn setLayerWeightsS(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        layer.weights_s.free(&self.ctx);
        layer.weights_s = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }

    pub fn setLayerWeightsT(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        layer.weights_t.free(&self.ctx);
        layer.weights_t = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }

    pub fn setLayerSBias(self: *Self, layer_idx: usize, data: []const f16, length: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (length == 0) return AccelError.InvalidDimensions;
        if (data.len != length) return AccelError.InvalidDimensions;
        layer.s_bias.free(&self.ctx);
        layer.s_bias = try FutharkArray1DF16.newFromFlat(&self.ctx, data, length);
    }

    pub fn setLayerTBias(self: *Self, layer_idx: usize, data: []const f16, length: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (length == 0) return AccelError.InvalidDimensions;
        if (data.len != length) return AccelError.InvalidDimensions;
        layer.t_bias.free(&self.ctx);
        layer.t_bias = try FutharkArray1DF16.newFromFlat(&self.ctx, data, length);
    }

    pub fn setLayerVelocityS(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        layer.velocity_s.free(&self.ctx);
        layer.velocity_s = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }

    pub fn setLayerVelocityT(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        layer.velocity_t.free(&self.ctx);
        layer.velocity_t = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }

    pub fn setLayerVelocitySB(self: *Self, layer_idx: usize, data: []const f16, length: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (length == 0) return AccelError.InvalidDimensions;
        if (data.len != length) return AccelError.InvalidDimensions;
        layer.velocity_sb.free(&self.ctx);
        layer.velocity_sb = try FutharkArray1DF16.newFromFlat(&self.ctx, data, length);
    }

    pub fn setLayerVelocityTB(self: *Self, layer_idx: usize, data: []const f16, length: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (length == 0) return AccelError.InvalidDimensions;
        if (data.len != length) return AccelError.InvalidDimensions;
        layer.velocity_tb.free(&self.ctx);
        layer.velocity_tb = try FutharkArray1DF16.newFromFlat(&self.ctx, data, length);
    }

    pub fn readLayerWeightsFlat(self: *Self, layer_idx: usize, kind: WeightKind, allocator: std.mem.Allocator) AccelError![]f16 {
        const layer = try self.layerPtr(layer_idx);
        return switch (kind) {
            .weights_s => readMatFlat(self, &layer.weights_s, allocator),
            .weights_t => readMatFlat(self, &layer.weights_t, allocator),
            .velocity_s => readMatFlat(self, &layer.velocity_s, allocator),
            .velocity_t => readMatFlat(self, &layer.velocity_t, allocator),
            .s_bias => readBiasFlat(self, &layer.s_bias, allocator),
            .t_bias => readBiasFlat(self, &layer.t_bias, allocator),
            .velocity_sb => readBiasFlat(self, &layer.velocity_sb, allocator),
            .velocity_tb => readBiasFlat(self, &layer.velocity_tb, allocator),
        };
    }

    /// Returns the raw GPU device pointer for the requested per-layer tensor.
    /// Used by the distributed trainer to feed NCCL allReduce directly
    /// without round-tripping through host memory.
    /// Element count is the second return value (half*half for matrices,
    /// half for biases).
    pub fn getLayerDevicePtr(
        self: *Self,
        layer_idx: usize,
        kind: WeightKind,
    ) AccelError!struct { ptr: *anyopaque, count: usize } {
        if (!self.initialized) return AccelError.NullPointer;
        const layer = try self.layerPtr(layer_idx);
        const half = self.model_dim / 2;
        return switch (kind) {
            .weights_s => .{ .ptr = try self.ctx.getDataPointer(&layer.weights_s), .count = half * half },
            .weights_t => .{ .ptr = try self.ctx.getDataPointer(&layer.weights_t), .count = half * half },
            .velocity_s => .{ .ptr = try self.ctx.getDataPointer(&layer.velocity_s), .count = half * half },
            .velocity_t => .{ .ptr = try self.ctx.getDataPointer(&layer.velocity_t), .count = half * half },
            .s_bias => .{ .ptr = try get1DDevicePtr(&self.ctx, &layer.s_bias), .count = half },
            .t_bias => .{ .ptr = try get1DDevicePtr(&self.ctx, &layer.t_bias), .count = half },
            .velocity_sb => .{ .ptr = try get1DDevicePtr(&self.ctx, &layer.velocity_sb), .count = half },
            .velocity_tb => .{ .ptr = try get1DDevicePtr(&self.ctx, &layer.velocity_tb), .count = half },
        };
    }

    fn readMatFlat(self: *Self, mat: *FutharkArray2DF16, allocator: std.mem.Allocator) AccelError![]f16 {
        const rows = try mat.values(&self.ctx, allocator);
        defer {
            for (rows) |row| allocator.free(row);
            allocator.free(rows);
        }
        const half = self.model_dim / 2;
        if (rows.len != half) return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, half, half) catch return AccelError.AllocationFailed;
        var flat = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        var idx: usize = 0;
        for (rows) |row| {
            if (row.len != half) return AccelError.InvalidDimensions;
            for (row) |v| {
                flat[idx] = v;
                idx += 1;
            }
        }
        return flat;
    }

    fn readBiasFlat(self: *Self, bias: *FutharkArray1DF16, allocator: std.mem.Allocator) AccelError![]f16 {
        const vals = try bias.values1D(&self.ctx, allocator);
        const half = self.model_dim / 2;
        if (vals.len != half) {
            allocator.free(vals);
            return AccelError.InvalidDimensions;
        }
        return vals;
    }

    pub fn setClipRange(self: *Self, clip_min_val: f16, clip_max_val: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (clip_min_val >= clip_max_val) return AccelError.InvalidDimensions;
        self.clip_min = clip_min_val;
        self.clip_max = clip_max_val;
    }

    pub fn forwardFromTensor(self: *Self, input: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (!self.initialized) return AccelError.NullPointer;
        if (input.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = input.shape.dims[0];
        const cols = input.shape.dims[1];
        const f16_data = allocator.alloc(f16, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(f16_data);
        {
            var i: usize = 0;
            while (i < input.data.len) : (i += 1) {
                const v = input.data[i];
                f16_data[i] = @floatCast(v);
            }
        }
        var f16_input = try FutharkArray2DF16.newFromFlat(&self.ctx, f16_data, rows, cols);
        defer f16_input.free(&self.ctx);
        var output = try self.forward(&f16_input);
        defer output.free(&self.ctx);
        const shape = [_]usize{ output.rows, output.cols };
        var result = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        const out_f16 = allocator.alloc(f16, output.rows * output.cols) catch {
            result.deinit();
            return AccelError.AllocationFailed;
        };
        defer allocator.free(out_f16);
        if (futhark.futhark_values_f16_2d(self.ctx.ctx, output.arr, @ptrCast(out_f16.ptr)) != 0) {
            result.deinit();
            return AccelError.FutharkValuesFailed;
        }
        {
            var i: usize = 0;
            while (i < out_f16.len) : (i += 1) {
                const v = out_f16[i];
                result.data[i] = @floatCast(v);
            }
        }
        return result;
    }
};

pub const GPUOps = struct {
    ctx: FutharkContext,

    const Self = @This();

    pub fn init() AccelError!Self {
        return Self{ .ctx = try FutharkContext.init() };
    }

    pub fn deinit(self: *Self) void {
        self.ctx.deinit();
    }

    pub fn matmul(self: *Self, a: *const core_tensor.Tensor, b: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        var fa = try FutharkArray2DF32.fromTensor(&self.ctx, a);
        defer fa.free(&self.ctx);
        var fb = try FutharkArray2DF32.fromTensor(&self.ctx, b);
        defer fb.free(&self.ctx);

        var out_arr: ?*futhark.struct_futhark_f32_2d = null;
        if (futhark.futhark_entry_matmul(self.ctx.ctx, &out_arr, fa.arr, fb.arr) != 0) {
            return AccelError.FutharkForwardFailed;
        }
        if (out_arr == null) return AccelError.NullPointer;

        var result = FutharkArray2DF32{ .arr = out_arr, .rows = a.shape.dims[0], .cols = b.shape.dims[1] };
        defer result.free(&self.ctx);
        return result.toTensor(&self.ctx, allocator);
    }
};

pub const FutharkArray1DI64 = struct {
    arr: ?*futhark.struct_futhark_i64_1d,
    len: usize,

    const Self = @This();

    pub fn newFromSlice(ctx: *FutharkContext, data: []const i64) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_i64_1d(ctx.ctx, data.ptr, @intCast(data.len));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = data.len };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            futhark.futhark_free_i64_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }
};

pub const EmbeddingAccelerator = struct {
    ctx: *FutharkContext,
    weight: FutharkArray2DF16,
    grad_weight: FutharkArray2DF16,
    vocab_size: usize,
    dim: usize,
    initialized: bool,

    const Self = @This();

    pub fn init(ctx: *FutharkContext, vocab_size: usize, dim: usize, seed: u64) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (vocab_size == 0 or dim == 0) return AccelError.InvalidDimensions;

        var rng = std.Random.DefaultPrng.init(seed);
        const rnd = rng.random();
        const total = vocab_size * dim;
        const weight_data = std.heap.page_allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(weight_data);
        for (weight_data) |*v| {
            v.* = @floatCast((rnd.float(f32) - 0.5) * 0.02);
        }

        const weight = try FutharkArray2DF16.newFromFlat(ctx, weight_data, vocab_size, dim);
        errdefer weight.free(ctx);
        const grad_weight = try FutharkArray2DF16.newZeros(ctx, vocab_size, dim, std.heap.page_allocator);

        return Self{
            .ctx = ctx,
            .weight = weight,
            .grad_weight = grad_weight,
            .vocab_size = vocab_size,
            .dim = dim,
            .initialized = true,
        };
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;
        self.grad_weight.free(self.ctx);
        self.weight.free(self.ctx);
        self.initialized = false;
    }

    pub fn forward(self: *Self, tokens: []const u32) AccelError!FutharkArray2DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;

        const token_i64s = std.heap.page_allocator.alloc(i64, tokens.len) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(token_i64s);
        for (token_i64s, 0..) |*t, i| {
            t.* = @intCast(@min(tokens[i], @as(u32, @intCast(self.vocab_size - 1))));
        }

        const tok_arr = try FutharkArray1DI64.newFromSlice(self.ctx, token_i64s);
        errdefer tok_arr.free(self.ctx);

        var out: ?*futhark.struct_futhark_f16_2d = null;
        const rc = futhark.futhark_entry_embedding_forward(
            self.ctx.ctx,
            &out,
            tok_arr.arr,
            self.weight.arr,
        );
        tok_arr.free(self.ctx);

        if (rc != 0 or out == null) return AccelError.FutharkForwardFailed;
        return FutharkArray2DF16{ .arr = out, .rows = tokens.len, .cols = self.dim };
    }

    pub fn backwardAndUpdate(self: *Self, tokens: []const u32, grad_output: *FutharkArray2DF16, lr: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;

        const token_i64s = std.heap.page_allocator.alloc(i64, tokens.len) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(token_i64s);
        for (token_i64s, 0..) |*t, i| {
            t.* = @intCast(@min(tokens[i], @as(u32, @intCast(self.vocab_size - 1))));
        }

        const tok_arr = try FutharkArray1DI64.newFromSlice(self.ctx, token_i64s);
        defer tok_arr.free(self.ctx);

        var new_grad: ?*futhark.struct_futhark_f16_2d = null;
        const bwd_rc = futhark.futhark_entry_embedding_backward(
            self.ctx.ctx,
            &new_grad,
            tok_arr.arr,
            grad_output.arr,
            self.grad_weight.arr,
        );
        if (bwd_rc != 0 or new_grad == null) return AccelError.FutharkBackwardFailed;

        const old_grad = self.grad_weight.arr;
        self.grad_weight.arr = new_grad;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_grad);

        var new_weight: ?*futhark.struct_futhark_f16_2d = null;
        const lr_bits: u16 = @bitCast(lr);
        const upd_rc = futhark.futhark_entry_embedding_update(
            self.ctx.ctx,
            &new_weight,
            self.weight.arr,
            self.grad_weight.arr,
            lr_bits,
        );
        if (upd_rc != 0 or new_weight == null) return AccelError.FutharkSFDUpdateFailed;

        const old_weight = self.weight.arr;
        self.weight.arr = new_weight;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_weight);

        const zeroed_grad = try FutharkArray2DF16.newZeros(self.ctx, self.vocab_size, self.dim, std.heap.page_allocator);
        const old_g = self.grad_weight.arr;
        self.grad_weight.arr = zeroed_grad.arr;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_g);
    }
};

