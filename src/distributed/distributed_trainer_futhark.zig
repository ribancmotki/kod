const std = @import("std");
const GPUCoordinator = @import("gpu_coordinator.zig").GPUCoordinator;
const MGT = @import("../tokenizer/mgt.zig").MGT;
const accel = @import("../hw/accel/accel_interface.zig");
const RSFAccelerator = accel.RSFAccelerator;
const FutharkArray2DF16 = accel.FutharkArray2DF16;
const FutharkArray1DF16 = accel.FutharkArray1DF16;
const FutharkArray3DF16 = accel.FutharkArray3DF16;
const PinnedMemory = accel.PinnedMemory;
const LearnedEmbedding = @import("../core/learned_embedding.zig").LearnedEmbedding;
const EmbeddingAccelerator = accel.EmbeddingAccelerator;
const futhark = @import("../hw/accel/futhark_bindings.zig");
const core_relational = @import("../core_relational/mod.zig");
const CREVPipeline = core_relational.CREVPipeline;
const ChaosCoreKernel = core_relational.ChaosCoreKernel;
const nsir = core_relational.nsir_core;
const SelfSimilarRelationalGraph = core_relational.SelfSimilarRelationalGraph;
const EntangledStochasticSymmetryOptimizer = core_relational.EntangledStochasticSymmetryOptimizer;
const SurpriseMemoryManager = core_relational.SurpriseMemoryManager;
const TemporalGraph = core_relational.TemporalGraph;
const QuantumState = core_relational.QuantumState;
const ReasoningOrchestrator = core_relational.ReasoningOrchestrator;
const SignalPropagationEngine = core_relational.SignalPropagationEngine;
const ZRuntime = core_relational.ZRuntime;
const RelationalGraphProcessingUnit = core_relational.RelationalGraphProcessingUnit;
const FNDSManager = core_relational.FNDSManager;
const VPU = core_relational.VPU;
const sfd = @import("../optimizer/sfd.zig");

pub const TrainerConfig = struct {
    learning_rate: f32 = 0.001,
    momentum: f32 = 0.0,
    max_line_size: usize = 10 * 1024 * 1024,
    checkpoint_version: u32 = 6,
};

pub const DistributedTrainerFuthark = struct {
    allocator: std.mem.Allocator,
    coordinator: *GPUCoordinator,
    tokenizer: MGT,
    accelerator: RSFAccelerator,
    model_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    local_batch_size: usize,
    global_step: u64,
    learning_rate: f32,
    momentum: f32,
    config: TrainerConfig,
    embedding: ?LearnedEmbedding,
    embedding_accel: ?EmbeddingAccelerator,
    crev_pipeline: ?CREVPipeline,
    crev_kernel: ?*ChaosCoreKernel,
    nsir_graph: SelfSimilarRelationalGraph,
    esso: EntangledStochasticSymmetryOptimizer,
    surprise_memory: SurpriseMemoryManager,
    temporal_graph: TemporalGraph,
    signal_engine: ?SignalPropagationEngine,
    z_runtime: *ZRuntime,
    r_gpu: ?RelationalGraphProcessingUnit,
    fnds_manager: FNDSManager,
    vpu: VPU,

    pub fn init(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        local_batch_size: usize,
    ) !DistributedTrainerFuthark {
        return initWithConfig(allocator, coordinator, model_dim, 1, local_batch_size, .{});
    }

    pub fn initWithConfig(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        num_layers: usize,
        local_batch_size: usize,
        config: TrainerConfig,
    ) !DistributedTrainerFuthark {
        if (model_dim == 0) return error.InvalidModelDim;
        if (model_dim % 2 != 0) return error.InvalidModelDim;
        if (num_layers == 0) return error.InvalidNumLayers;
        if (local_batch_size == 0) return error.InvalidBatchSize;
        if (coordinator.world_size == 0) return error.InvalidWorldSize;
        if (coordinator.rank >= coordinator.world_size) return error.InvalidRank;
        if (config.max_line_size == 0) return error.InvalidMaxLineSize;
        if (config.checkpoint_version == 0) return error.InvalidCheckpointVersion;
        try validateHyperparameters(config.learning_rate, config.momentum);

        const vocab = &[_][]const u8{
            "a",     "about",   "all",   "also",  "and",   "as",    "at",
            "be",    "because", "but",   "by",    "can",   "come",  "could",
            "day",   "do",      "even",  "find",  "first", "for",   "from",
            "get",   "give",    "go",    "have",  "he",    "her",   "here",
            "him",   "his",     "how",   "i",     "if",    "in",    "into",
            "it",    "its",     "just",  "know",  "like",  "look",  "make",
            "man",   "many",    "me",    "more",  "my",    "new",   "no",
            "not",   "now",     "of",    "on",    "one",   "only",  "or",
            "other", "our",     "out",   "people", "say",  "see",   "she",
            "so",    "some",    "take",  "tell",  "than",  "that",  "the",
            "their", "them",    "then",  "there", "these", "they",  "thing",
            "think", "this",    "those", "time",  "to",    "two",   "up",
            "use",   "very",    "want",  "way",   "we",    "well",  "what",
            "when",  "which",   "who",   "will",  "with",  "would", "year",
            "you",   "your",
        };
        const empty_anchors: []const []const u8 = &.{};

        var tokenizer = try MGT.init(allocator, vocab, empty_anchors, 50000, .english);
        errdefer tokenizer.deinit();

        std.debug.print("tokenizer.next_token_id = {d}\n", .{tokenizer.next_token_id});
        var actual_model_dim = model_dim;
        if (actual_model_dim < tokenizer.next_token_id) {
            actual_model_dim = tokenizer.next_token_id;
            if (actual_model_dim % 2 != 0) actual_model_dim += 1;
        }

        var accelerator = try RSFAccelerator.initMultiLayer(actual_model_dim, num_layers, allocator);
        errdefer accelerator.deinit();

        var embedding = try LearnedEmbedding.init(allocator, 50000, actual_model_dim, 42);
        errdefer embedding.deinit();

        const embedding_accel: ?EmbeddingAccelerator = null;

        var crev_kernel = try allocator.create(ChaosCoreKernel);
        crev_kernel.* = ChaosCoreKernel.init(allocator);
        errdefer {
            crev_kernel.deinit();
            allocator.destroy(crev_kernel);
        }

        var crev_pipeline = try CREVPipeline.init(allocator, crev_kernel);
        errdefer crev_pipeline.deinit();

        var nsir_graph = try SelfSimilarRelationalGraph.init(allocator);
        errdefer nsir_graph.deinit();

        const esso = EntangledStochasticSymmetryOptimizer.init(allocator, 1.0, 0.995, 1000);
        errdefer esso.deinit();

        const surprise_memory = SurpriseMemoryManager.init(
            allocator,
            &crev_kernel.storage,
            &crev_kernel.flow_analyzer,
        );
        errdefer surprise_memory.deinit();

        const temporal_graph_inst = TemporalGraph.init(allocator);
        errdefer temporal_graph_inst.deinit();

        const signal_engine_null: ?SignalPropagationEngine = null;

        var z_runtime = try ZRuntime.init(allocator);
        errdefer z_runtime.deinit();

        const r_gpu_inst = RelationalGraphProcessingUnit.init(allocator, 4, 4) catch null;
        errdefer {
            if (r_gpu_inst) |*rg| rg.deinit();
        }

        var fnds_manager_inst = try FNDSManager.init(allocator);
        errdefer fnds_manager_inst.deinit();

        var vpu_inst = try VPU.init(allocator);
        errdefer vpu_inst.deinit();

        return DistributedTrainerFuthark{
            .allocator = allocator,
            .coordinator = coordinator,
            .tokenizer = tokenizer,
            .accelerator = accelerator,
            .model_dim = actual_model_dim,
            .num_layers = num_layers,
            .vocab_size = tokenizer.next_token_id,
            .local_batch_size = local_batch_size,
            .global_step = 0,
            .learning_rate = config.learning_rate,
            .momentum = config.momentum,
            .config = config,
            .embedding = embedding,
            .embedding_accel = embedding_accel,
            .crev_pipeline = crev_pipeline,
            .crev_kernel = crev_kernel,
            .nsir_graph = nsir_graph,
            .esso = esso,
            .surprise_memory = surprise_memory,
            .temporal_graph = temporal_graph_inst,
            .signal_engine = signal_engine_null,
            .z_runtime = z_runtime,
            .r_gpu = r_gpu_inst,
            .fnds_manager = fnds_manager_inst,
            .vpu = vpu_inst,
        };
    }

    pub fn postInit(self: *DistributedTrainerFuthark) void {
        self.signal_engine = SignalPropagationEngine.init(
            self.allocator,
            &self.nsir_graph,
            &self.crev_kernel.?.flow_analyzer,
        );
    }

    pub fn deinit(self: *DistributedTrainerFuthark) void {
        self.accelerator.sync() catch {};
        self.vpu.deinit();
        self.fnds_manager.deinit();
        if (self.r_gpu) |*rg| rg.deinit();
        self.z_runtime.deinit();
        if (self.signal_engine) |*se| se.deinit();
        self.temporal_graph.deinit();
        self.surprise_memory.deinit();
        self.esso.deinit();
        self.nsir_graph.deinit();
        if (self.crev_pipeline) |*cp| cp.deinit();
        if (self.crev_kernel) |ck| {
            ck.deinit();
            self.allocator.destroy(ck);
        }
        if (self.embedding_accel) |*ea| ea.deinit();
        if (self.embedding) |*emb| emb.deinit();
        self.accelerator.deinit();
        self.tokenizer.deinit();
    }

    pub fn reinitEmbedding(self: *DistributedTrainerFuthark) !void {
        if (self.embedding) |*emb| emb.deinit();
        self.embedding = try LearnedEmbedding.init(
            self.allocator,
            self.tokenizer.next_token_id,
            self.model_dim,
            42,
        );
        self.vocab_size = self.tokenizer.next_token_id;
    }

    fn validateHyperparameters(learning_rate: f32, momentum: f32) !void {
        if (!std.math.isFinite(learning_rate)) return error.InvalidLearningRate;
        if (!std.math.isFinite(momentum)) return error.InvalidMomentum;
        if (learning_rate < 0.0 or learning_rate > 65504.0) return error.InvalidLearningRate;
        if (momentum < 0.0 or momentum >= 1.0) return error.InvalidMomentum;
    }

    fn openReadFile(path: []const u8) !std.fs.File {
        if (std.fs.path.isAbsolute(path)) {
            return std.fs.openFileAbsolute(path, .{ .mode = .read_only });
        }
        return std.fs.cwd().openFile(path, .{ .mode = .read_only });
    }

    fn createWriteFile(path: []const u8) !std.fs.File {
        if (std.fs.path.isAbsolute(path)) {
            return std.fs.createFileAbsolute(path, .{ .mode = 0o600 });
        }
        return std.fs.cwd().createFile(path, .{ .mode = 0o600 });
    }

    fn writeF32(writer: anytype, value: f32) !void {
        try writer.writeInt(u32, @as(u32, @bitCast(value)), .little);
    }

    fn readF32(reader: anytype) !f32 {
        const bits = try reader.readInt(u32, .little);
        return @as(f32, @bitCast(bits));
    }

    fn isTokenizableText(self: *DistributedTrainerFuthark, text: []const u8) !bool {
        var token_list = std.ArrayList(u32).init(self.allocator);
        defer token_list.deinit();
        try self.tokenizer.encode(text, &token_list);
        return token_list.items.len > 0;
    }

    fn extractDatasetText(self: *DistributedTrainerFuthark, line: []const u8) !?[]const u8 {
        const parsed = std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            line,
            .{ .allocate = .alloc_always },
        ) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => return null,
        };
        defer parsed.deinit();

        return switch (parsed.value) {
            .object => |obj| blk: {
                const text_value = obj.get("text") orelse break :blk null;
                break :blk switch (text_value) {
                    .string => |text| if (text.len > 0)
                        try self.allocator.dupe(u8, text)
                    else
                        null,
                    else => null,
                };
            },
            else => null,
        };
    }

    fn isUsableDatasetLine(self: *DistributedTrainerFuthark, line: []const u8) !bool {
        const maybe_text = try self.extractDatasetText(line);
        if (maybe_text) |text| {
            defer self.allocator.free(text);
            return try self.isTokenizableText(text);
        }
        return false;
    }

    fn appendDatasetRange(
        self: *DistributedTrainerFuthark,
        dataset_path: []const u8,
        start_valid_index: usize,
        count: usize,
        samples: *std.ArrayList([]const u8),
    ) !void {
        if (count == 0) {
            return;
        }

        const end_valid_index = try std.math.add(usize, start_valid_index, count);
        var appended: usize = 0;
        var line_index: usize = 0;

        const load_file = openReadFile(dataset_path) catch |err| {
            std.debug.print("[Rank {d}] ERROR: Cannot open dataset: {}\n", .{ self.coordinator.rank, err });
            return err;
        };
        defer load_file.close();

        var load_buf_reader = std.io.bufferedReader(load_file.reader());
        var load_stream = load_buf_reader.reader();

        while (try load_stream.readUntilDelimiterOrEofAlloc(self.allocator, '\n', self.config.max_line_size)) |line| {
            defer self.allocator.free(line);

            if (line_index >= end_valid_index) {
                return;
            }

            if (line_index >= start_valid_index) {
                const maybe_text = try self.extractDatasetText(line);
                if (maybe_text) |text_copy| {
                    try samples.append(text_copy);
                    appended += 1;
                }
            }
            line_index += 1;

            if (appended == count) {
                return;
            }
        }

        if (appended < count) {
            std.debug.print("[Rank {d}] WARN: read {d}/{d} samples before EOF (line_index={d})\n", .{ self.coordinator.rank, appended, count, line_index });
        }
    }

    fn readLayerMatrix(self: *DistributedTrainerFuthark, layer_idx: usize, kind: accel.WeightKind) ![]f16 {
        return self.accelerator.readLayerWeightsFlat(layer_idx, kind, self.allocator) catch |err| {
            std.debug.print("[Rank {d}] readLayerMatrix layer={d} err={}\n", .{ self.coordinator.rank, layer_idx, err });
            return err;
        };
    }

    fn allReduceFloat32Values(self: *DistributedTrainerFuthark, values: []f32) !void {
        if (values.len == 0 or self.coordinator.world_size <= 1) {
            return;
        }

        const byte_count = try std.math.mul(usize, values.len, @sizeOf(f32));
        const values_dev = try self.coordinator.allocDeviceMemory(byte_count);
        defer self.coordinator.freeDeviceMemory(values_dev);

        try self.coordinator.copyHostToDevice(values_dev, std.mem.sliceAsBytes(values), byte_count);
        try self.coordinator.allReduceFloat32(values_dev, values_dev, values.len);
        try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(values), values_dev, byte_count);
        try self.coordinator.synchronize();
    }

    fn allReduceScalarF32(self: *DistributedTrainerFuthark, value: f32) !f32 {
        var values = [1]f32{value};
        try self.allReduceFloat32Values(values[0..]);
        return values[0];
    }

    fn averageDeltaInPlace(self: *DistributedTrainerFuthark, delta: []f16) !void {
        if (delta.len == 0 or self.coordinator.world_size <= 1) {
            return;
        }

        const byte_count = try std.math.mul(usize, delta.len, @sizeOf(f16));
        const delta_dev = try self.coordinator.allocDeviceMemory(byte_count);
        defer self.coordinator.freeDeviceMemory(delta_dev);

        try self.coordinator.copyHostToDevice(delta_dev, std.mem.sliceAsBytes(delta), byte_count);
        try self.coordinator.allReduceFloat16(delta_dev, delta_dev, delta.len);
        try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(delta), delta_dev, byte_count);
        try self.coordinator.synchronize();

        const inv_world: f32 = 1.0 / @as(f32, @floatFromInt(self.coordinator.world_size));
        for (delta) |*value| {
            const scaled = @as(f32, @floatCast(value.*)) * inv_world;
            value.* = @floatCast(scaled);
        }
    }

    fn gpuAllReduceLayers(self: *DistributedTrainerFuthark) !void {
        if (self.coordinator.world_size <= 1) return;

        const kinds = [_]accel.WeightKind{
            .weights_s,
            .weights_t,
            .s_bias,
            .t_bias,
            .velocity_s,
            .velocity_t,
            .velocity_sb,
            .velocity_tb,
        };

        var li: usize = 0;
        while (li < self.num_layers) : (li += 1) {
            for (kinds) |k| {
                const info = try self.accelerator.getLayerDevicePtr(li, k);
                if (info.count == 0) continue;
                try self.coordinator.allReduceFloat16Avg(info.ptr, info.ptr, info.count);
            }
        }
    }

    fn applyDeltaToLayer(self: *DistributedTrainerFuthark, layer_idx: usize, base: []const f16, delta: []const f16, which: enum { s, t }) !void {
        if (base.len != delta.len) {
            return error.InvalidWeightsShape;
        }
        const half: usize = self.model_dim / 2;
        const expected_len = try std.math.mul(usize, half, half);
        if (base.len != expected_len) {
            std.debug.print("[Rank {d}] applyDeltaToLayer: base.len={d}, expected half*half={d}, model_dim={d}\n", .{ self.coordinator.rank, base.len, expected_len, self.model_dim });
            return error.InvalidWeightsShape;
        }

        var merged = try self.allocator.alloc(f16, base.len);
        defer self.allocator.free(merged);

        for (base, delta, 0..) |base_value, delta_value, idx| {
            const merged_value = @as(f32, @floatCast(base_value)) + @as(f32, @floatCast(delta_value));
            merged[idx] = @floatCast(merged_value);
        }

        switch (which) {
            .s => try self.accelerator.setLayerWeightsS(layer_idx, merged, half, half),
            .t => try self.accelerator.setLayerWeightsT(layer_idx, merged, half, half),
        }
    }

    fn applyBiasDeltaToLayer(self: *DistributedTrainerFuthark, layer_idx: usize, base: []const f16, delta: []const f16, which: enum { s, t }) !void {
        if (base.len != delta.len) {
            return error.InvalidWeightsShape;
        }
        const half: usize = self.model_dim / 2;
        if (base.len != half) {
            std.debug.print("[Rank {d}] applyBiasDeltaToLayer: base.len={d}, expected half={d}, model_dim={d}\n", .{ self.coordinator.rank, base.len, half, self.model_dim });
            return error.InvalidWeightsShape;
        }

        var merged = try self.allocator.alloc(f16, base.len);
        defer self.allocator.free(merged);

        for (base, delta, 0..) |base_value, delta_value, idx| {
            const merged_value = @as(f32, @floatCast(base_value)) + @as(f32, @floatCast(delta_value));
            merged[idx] = @floatCast(merged_value);
        }

        switch (which) {
            .s => try self.accelerator.setLayerSBias(layer_idx, merged, half),
            .t => try self.accelerator.setLayerTBias(layer_idx, merged, half),
        }
    }

    pub fn loadDataset(self: *DistributedTrainerFuthark, dataset_path: []const u8) ![][]const u8 {
        if (self.coordinator.world_size == 0) return error.InvalidWorldSize;
        if (self.coordinator.rank >= self.coordinator.world_size) return error.InvalidRank;

        const env_total_owned: ?[]u8 = std.process.getEnvVarOwned(self.allocator, "JAIDE_TOTAL_SAMPLES") catch null;
        defer if (env_total_owned) |o| self.allocator.free(o);
        const env_max_owned: ?[]u8 = std.process.getEnvVarOwned(self.allocator, "JAIDE_MAX_SAMPLES") catch null;
        defer if (env_max_owned) |o| self.allocator.free(o);

        var valid_sample_count: usize = 0;
        if (env_total_owned) |s| {
            valid_sample_count = std.fmt.parseInt(usize, s, 10) catch 0;
        }
        if (env_max_owned) |s| {
            const cap = std.fmt.parseInt(usize, s, 10) catch 0;
            if (cap > 0 and cap < valid_sample_count) valid_sample_count = cap;
        }

        if (valid_sample_count == 0) {
            std.debug.print("[Rank {d}] WARN: JAIDE_TOTAL_SAMPLES not provided, falling back to scan\n", .{self.coordinator.rank});
            const count_file = openReadFile(dataset_path) catch |err| return err;
            defer count_file.close();
            var count_buf_reader = std.io.bufferedReader(count_file.reader());
            var count_stream = count_buf_reader.reader();
            while (try count_stream.readUntilDelimiterOrEofAlloc(self.allocator, '\n', self.config.max_line_size)) |line| {
                defer self.allocator.free(line);
                valid_sample_count = try std.math.add(usize, valid_sample_count, 1);
            }
        }

        if (valid_sample_count == 0) {
            std.debug.print("[Rank {d}] ERROR: Dataset is empty\n", .{self.coordinator.rank});
            return error.EmptyDataset;
        }

        const base_per_rank = valid_sample_count / self.coordinator.world_size;
        const remainder = valid_sample_count % self.coordinator.world_size;
        const samples_per_rank = if (self.coordinator.rank < remainder) base_per_rank + 1 else base_per_rank;
        const start_valid_index = if (self.coordinator.rank < remainder)
            self.coordinator.rank * (base_per_rank + 1)
        else
            remainder * (base_per_rank + 1) + (self.coordinator.rank - remainder) * base_per_rank;

        var samples = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (samples.items) |sample| {
                self.allocator.free(sample);
            }
            samples.deinit();
        }

        if (samples_per_rank > 0) {
            try self.appendDatasetRange(dataset_path, start_valid_index, samples_per_rank, &samples);
        }

        if (samples.items.len != samples_per_rank) {
            std.debug.print("[Rank {d}] ERROR: partition got {d} samples, expected {d}\n", .{ self.coordinator.rank, samples.items.len, samples_per_rank });
            return error.InvalidDatasetPartition;
        }

        if (self.coordinator.isRoot()) {
            std.debug.print("[Rank {d}] Loaded {d} samples from total {d} (rank slice)\n", .{
                self.coordinator.rank,
                samples.items.len,
                valid_sample_count,
            });
        }

        return samples.toOwnedSlice();
    }

    pub fn trainEpoch(self: *DistributedTrainerFuthark, samples: [][]const u8) !f32 {
        if (self.local_batch_size == 0) return error.InvalidBatchSize;

        var total_loss: f32 = 0.0;
        var num_batches: usize = 0;

        const local_total: f32 = @floatFromInt(samples.len);
        const global_total = try self.allReduceScalarF32(local_total);
        _ = global_total;

        const local_batches_count: usize = (samples.len + self.local_batch_size - 1) / self.local_batch_size;
        var max_batches_local: f32 = @floatFromInt(local_batches_count);
        if (self.coordinator.world_size > 1) {
            var arr = [1]f32{max_batches_local};
            const byte_count = arr.len * @sizeOf(f32);
            const dev = try self.coordinator.allocDeviceMemory(byte_count);
            defer self.coordinator.freeDeviceMemory(dev);
            try self.coordinator.copyHostToDevice(dev, std.mem.sliceAsBytes(arr[0..]), byte_count);
            try self.coordinator.allReduceFloat32Max(dev, dev, arr.len);
            try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(arr[0..]), dev, byte_count);
            try self.coordinator.synchronize();
            max_batches_local = arr[0];
        }

        const target_batches: usize = @intFromFloat(max_batches_local);
        var batch_idx: usize = 0;
        var batch_start: usize = 0;
        while (batch_idx < target_batches) : (batch_idx += 1) {
            var batch: [][]const u8 = &.{};
            if (batch_start < samples.len) {
                const remaining = samples.len - batch_start;
                const batch_len = @min(self.local_batch_size, remaining);
                const batch_end = batch_start + batch_len;
                batch = samples[batch_start..batch_end];
                batch_start = batch_end;
            }

            const loss = self.trainStepFuthark(batch) catch |err| {
                std.debug.print("[Rank {d}] trainStepFuthark ERROR at step {d}: {}\n", .{ self.coordinator.rank, self.global_step, err });
                return err;
            };
            if (!std.math.isFinite(loss)) return error.InvalidLoss;
            if (batch.len > 0) {
                total_loss += loss;
                num_batches = try std.math.add(usize, num_batches, 1);
            }

            if (self.coordinator.isRoot() and self.global_step % 10 == 0) {
                std.debug.print("[Step {d}] Loss: {d:.4}\n", .{ self.global_step, loss });
            }

            self.global_step = try std.math.add(u64, self.global_step, 1);
        }

        var loss_and_count = [2]f32{ total_loss, @floatFromInt(num_batches) };
        try self.allReduceFloat32Values(loss_and_count[0..]);

        const global_loss_sum = loss_and_count[0];
        const global_batch_count = loss_and_count[1];

        if (global_batch_count < 1.0) {
            std.debug.print("[WARNING] No batches processed across all ranks\n", .{});
            return 0.0;
        }

        return global_loss_sum / global_batch_count;
    }

    fn runCoreRelationalPass(self: *DistributedTrainerFuthark, token_lists_items: []const std.ArrayList(u32)) void {
        if (token_lists_items.len == 0) return;
        const first_tokens = token_lists_items[0].items;
        const tensor_bytes = std.mem.sliceAsBytes(first_tokens);

        _ = self.nsir_graph.encodeInformation(tensor_bytes) catch {};

        if (self.r_gpu) |*rg| {
            rg.distributeGraph(&self.nsir_graph) catch {};
        }

        {
            var orchestrator = ReasoningOrchestrator.init(
                self.allocator,
                &self.nsir_graph,
                &self.esso,
                self.crev_kernel.?,
            );
            defer orchestrator.deinit();
            _ = orchestrator.runHierarchicalReasoning(50) catch {};
        }

        _ = self.surprise_memory.storeWithSurprise(tensor_bytes, null) catch {};

        {
            const now_ns: i64 = @truncate(std.time.nanoTimestamp());
            var node_iter = self.nsir_graph.nodes.iterator();
            while (node_iter.next()) |entry| {
                const node = entry.value_ptr;
                const qs = QuantumState.init(
                    node.qubit.a.re,
                    node.qubit.a.im,
                    node.qubit.b.re,
                    node.qubit.b.im,
                    node.phase,
                    0.0,
                );
                self.temporal_graph.addNodeAtTime(node.id, qs, now_ns) catch {};
            }
            const after_ns: i64 = @truncate(std.time.nanoTimestamp());
            self.temporal_graph.advanceTime(after_ns - now_ns);
        }

        if (self.signal_engine) |*se| {
            se.propagateStep() catch {};
        }

        {
            var name_buf: [64]u8 = undefined;
            const var_name = std.fmt.bufPrint(&name_buf, "train_{}", .{self.global_step}) catch "train";
            _ = self.z_runtime.createVariable(var_name, null) catch {};
        }
    }

    pub fn trainStepFuthark(self: *DistributedTrainerFuthark, batch: [][]const u8) !f32 {
        const local_active_f: f32 = if (batch.len > 0) 1.0 else 0.0;
        if (self.coordinator.world_size > 1) {
            const active_count = try self.allReduceScalarF32(local_active_f);
            if (active_count == 0.0) return 0.0;
        } else {
            if (batch.len == 0) return 0.0;
        }

        var token_lists = std.ArrayList(std.ArrayList(u32)).init(self.allocator);
        defer {
            for (token_lists.items) |*list| {
                list.deinit();
            }
            token_lists.deinit();
        }

        for (batch) |text| {
            var token_list = std.ArrayList(u32).init(self.allocator);
            errdefer token_list.deinit();
            try self.tokenizer.encode(text, &token_list);
            try token_lists.append(token_list);
        }

        const cap: usize = blk: {
            const env_val = std.process.getEnvVarOwned(self.allocator, "JAIDE_MAX_SEQ_LEN") catch break :blk 256;
            defer self.allocator.free(env_val);
            const parsed = std.fmt.parseInt(usize, env_val, 10) catch break :blk 256;
            if (parsed == 0) break :blk 256;
            break :blk parsed;
        };

        var max_seq_len: usize = 0;
        for (token_lists.items) |*list| {
            if (list.items.len > cap) {
                list.shrinkRetainingCapacity(cap);
            }
            max_seq_len = @max(max_seq_len, list.items.len);
        }
        if (max_seq_len > cap) max_seq_len = cap;

        if (self.coordinator.world_size > 1) {
            var msl_arr = [1]f32{@floatFromInt(max_seq_len)};
            const byte_count = msl_arr.len * @sizeOf(f32);
            const dev = try self.coordinator.allocDeviceMemory(byte_count);
            defer self.coordinator.freeDeviceMemory(dev);
            try self.coordinator.copyHostToDevice(dev, std.mem.sliceAsBytes(msl_arr[0..]), byte_count);
            try self.coordinator.allReduceFloat32Max(dev, dev, msl_arr.len);
            try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(msl_arr[0..]), dev, byte_count);
            try self.coordinator.synchronize();
            var global_msl: usize = @intFromFloat(msl_arr[0]);
            if (global_msl == 0) return 0.0;
            if (global_msl > cap) global_msl = cap;
            max_seq_len = global_msl;
        } else {
            if (max_seq_len == 0) return 0.0;
        }

        const effective_batch_size: usize = if (batch.len == 0) 1 else batch.len;
        const batch_rows = try std.math.mul(usize, effective_batch_size, max_seq_len);
        const data_elements = try std.math.mul(usize, batch_rows, self.model_dim);
        const data_size = try std.math.mul(usize, data_elements, @sizeOf(f16));

        var pinned_input = try PinnedMemory.alloc(data_size);
        defer pinned_input.free();
        var pinned_target = try PinnedMemory.alloc(data_size);
        defer pinned_target.free();

        const input_f16_data = pinned_input.asSlice(f16) orelse return error.AllocationFailed;
        const target_f16_data = pinned_target.asSlice(f16) orelse return error.AllocationFailed;
        if (input_f16_data.len != data_elements or target_f16_data.len != data_elements) {
            return error.InvalidPinnedMemorySize;
        }
        @memset(input_f16_data, @as(f16, 0.0));
        @memset(target_f16_data, @as(f16, 0.0));

        if (self.embedding) |*emb| {
            emb.zeroGrad();
            var b_idx: usize = 0;
            while (b_idx < token_lists.items.len) : (b_idx += 1) {
                const list = token_lists.items[b_idx].items;
                if (list.len == 0) continue;

                var emb_tensor = try emb.forward(self.allocator, list, max_seq_len);
                defer emb_tensor.deinit();

                const emb_rows = emb_tensor.shape.dims[0];
                var seq_idx: usize = 0;
                while (seq_idx < emb_rows) : (seq_idx += 1) {
                    const row_offset = try std.math.mul(usize, b_idx, max_seq_len);
                    const row_index = try std.math.add(usize, row_offset, seq_idx);
                    const base_idx = try std.math.mul(usize, row_index, self.model_dim);
                    var c: usize = 0;
                    while (c < self.model_dim) : (c += 1) {
                        const src_idx = seq_idx * self.model_dim + c;
                        if (src_idx < emb_tensor.data.len and base_idx + c < input_f16_data.len) {
                            input_f16_data[base_idx + c] = @floatCast(emb_tensor.data[src_idx]);
                        }
                    }
                }

                seq_idx = 0;
                while (seq_idx < list.len) : (seq_idx += 1) {
                    if (seq_idx + 1 >= list.len) continue;
                    const next_token: usize = @min(@as(usize, list[seq_idx + 1]), emb.vocab_size - 1);
                    const row_offset = try std.math.mul(usize, b_idx, max_seq_len);
                    const row_index = try std.math.add(usize, row_offset, seq_idx);
                    const base_idx = try std.math.mul(usize, row_index, self.model_dim);
                    var c: usize = 0;
                    while (c < self.model_dim) : (c += 1) {
                        const w_idx = next_token * self.model_dim + c;
                        if (w_idx < emb.weight.data.len and base_idx + c < target_f16_data.len) {
                            target_f16_data[base_idx + c] = @floatCast(emb.weight.data[w_idx]);
                        }
                    }
                }
            }
        } else {
            var b_idx: usize = 0;
            while (b_idx < token_lists.items.len) : (b_idx += 1) {
                const list = token_lists.items[b_idx].items;
                if (list.len == 0) continue;
                var seq_idx: usize = 0;
                while (seq_idx < list.len) : (seq_idx += 1) {
                    const token_index_raw: usize = @intCast(list[seq_idx]);
                    const token_index: usize = token_index_raw;
                    if (token_index >= self.model_dim) {
                        std.debug.print("[Rank {d}] token id {d} >= model_dim {d} (vocab_size={d}); aborting step\n", .{ self.coordinator.rank, token_index, self.model_dim, self.vocab_size });
                        return error.TokenIndexOutOfRange;
                    }

                    const row_offset = try std.math.mul(usize, b_idx, max_seq_len);
                    const row_index = try std.math.add(usize, row_offset, seq_idx);
                    const base_idx = try std.math.mul(usize, row_index, self.model_dim);
                    const final_idx = try std.math.add(usize, base_idx, token_index);
                    if (final_idx >= input_f16_data.len) return error.IndexOutOfBounds;
                    input_f16_data[final_idx] = @as(f16, 1.0);

                    if (seq_idx + 1 < list.len) {
                        const next_token_raw: usize = @intCast(list[seq_idx + 1]);
                        const next_token: usize = next_token_raw;
                        if (next_token >= self.model_dim) {
                            std.debug.print("[Rank {d}] next-token id {d} >= model_dim {d}; aborting step\n", .{ self.coordinator.rank, next_token, self.model_dim });
                            return error.TokenIndexOutOfRange;
                        }
                        const tgt_final = try std.math.add(usize, base_idx, next_token);
                        if (tgt_final >= target_f16_data.len) return error.IndexOutOfBounds;
                        target_f16_data[tgt_final] = @as(f16, 1.0);
                    }
                }
            }
        }

        if (self.global_step == 0 and self.coordinator.isRoot() and token_lists.items.len > 0) {
            const first_list = token_lists.items[0].items;
            const dump_n: usize = @min(@as(usize, 12), first_list.len);
            std.debug.print("[Rank 0 step0] first sample tokens (len={d}): ", .{first_list.len});
            {
                var i: usize = 0;
                while (i < dump_n) : (i += 1) {
                    std.debug.print("{d} ", .{first_list[i]});
                }
                std.debug.print("\n", .{});
            }

            std.debug.print("[Rank 0 step0] input row->token: ", .{});
            {
                var row: usize = 0;
                while (row < dump_n) : (row += 1) {
                    const base = row * self.model_dim;
                    var found: isize = -1;
                    var c: usize = 0;
                    while (c < self.model_dim) : (c += 1) {
                        if (input_f16_data[base + c] != @as(f16, 0.0)) {
                            found = @intCast(c);
                            break;
                        }
                    }
                    std.debug.print("[{d}]={d} ", .{ row, found });
                }
                std.debug.print("\n", .{});
            }
            std.debug.print("[Rank 0 step0] target row->token: ", .{});
            {
                var row: usize = 0;
                while (row < dump_n) : (row += 1) {
                    const base = row * self.model_dim;
                    var found: isize = -1;
                    var c: usize = 0;
                    while (c < self.model_dim) : (c += 1) {
                        if (target_f16_data[base + c] != @as(f16, 0.0)) {
                            found = @intCast(c);
                            break;
                        }
                    }
                    std.debug.print("[{d}]={d} ", .{ row, found });
                }
                std.debug.print("\n", .{});
            }
            std.debug.print("[Rank 0 step0] (target[i] must equal input[i+1] => 1-token shift)\n", .{});
            std.debug.print("[Rank 0 step0] effective_batch_size={d} max_seq_len={d} model_dim={d}\n", .{ effective_batch_size, max_seq_len, self.model_dim });
        }

        var inputs = try FutharkArray3DF16.newFromFlat(&self.accelerator.ctx, input_f16_data, effective_batch_size, max_seq_len, self.model_dim);
        defer inputs.free(&self.accelerator.ctx);

        var targets = try FutharkArray3DF16.newFromFlat(&self.accelerator.ctx, target_f16_data, effective_batch_size, max_seq_len, self.model_dim);
        defer targets.free(&self.accelerator.ctx);

        const lr_f16: f16 = @floatCast(self.learning_rate);
        const mom_f16: f16 = @floatCast(self.momentum);

        if (self.coordinator.world_size <= 1) {
            const loss_f16 = try self.accelerator.trainingStep(&inputs, &targets, lr_f16, mom_f16);
            try self.accelerator.sync();

            self.propagateEmbeddingGradients(&inputs, &targets, token_lists.items, effective_batch_size, max_seq_len) catch {};

            self.runCoreRelationalPass(token_lists.items);

            const loss_f32: f32 = @floatCast(loss_f16);
            if (!std.math.isFinite(loss_f32)) return error.InvalidLoss;
            return loss_f32;
        }

        const Snapshots = struct {
            ws: [][]f16,
            wt: [][]f16,
            sb: [][]f16,
            tb: [][]f16,
        };

        var snap = Snapshots{
            .ws = try self.allocator.alloc([]f16, self.num_layers),
            .wt = try self.allocator.alloc([]f16, self.num_layers),
            .sb = try self.allocator.alloc([]f16, self.num_layers),
            .tb = try self.allocator.alloc([]f16, self.num_layers),
        };
        defer {
            for (snap.ws) |p| if (p.len > 0) self.allocator.free(p);
            for (snap.wt) |p| if (p.len > 0) self.allocator.free(p);
            for (snap.sb) |p| if (p.len > 0) self.allocator.free(p);
            for (snap.tb) |p| if (p.len > 0) self.allocator.free(p);
            self.allocator.free(snap.ws);
            self.allocator.free(snap.wt);
            self.allocator.free(snap.sb);
            self.allocator.free(snap.tb);
        }
        for (snap.ws) |*p| p.* = &.{};
        for (snap.wt) |*p| p.* = &.{};
        for (snap.sb) |*p| p.* = &.{};
        for (snap.tb) |*p| p.* = &.{};

        var li_before: usize = 0;
        while (li_before < self.num_layers) : (li_before += 1) {
            snap.ws[li_before] = try self.readLayerMatrix(li_before, .weights_s);
            snap.wt[li_before] = try self.readLayerMatrix(li_before, .weights_t);
            snap.sb[li_before] = try self.readLayerMatrix(li_before, .s_bias);
            snap.tb[li_before] = try self.readLayerMatrix(li_before, .t_bias);
        }

        const loss_f16 = try self.accelerator.trainingStep(&inputs, &targets, lr_f16, mom_f16);
        try self.accelerator.sync();

        var li_after: usize = 0;
        while (li_after < self.num_layers) : (li_after += 1) {
            const ws_after = try self.readLayerMatrix(li_after, .weights_s);
            defer self.allocator.free(ws_after);
            const wt_after = try self.readLayerMatrix(li_after, .weights_t);
            defer self.allocator.free(wt_after);
            const sb_after = try self.readLayerMatrix(li_after, .s_bias);
            defer self.allocator.free(sb_after);
            const tb_after = try self.readLayerMatrix(li_after, .t_bias);
            defer self.allocator.free(tb_after);

            if (ws_after.len != snap.ws[li_after].len or
                wt_after.len != snap.wt[li_after].len or
                sb_after.len != snap.sb[li_after].len or
                tb_after.len != snap.tb[li_after].len)
            {
                return error.InvalidWeightsShape;
            }

            for (ws_after, snap.ws[li_after]) |*v, b| {
                v.* = @floatCast(@as(f32, @floatCast(v.*)) - @as(f32, @floatCast(b)));
            }
            for (wt_after, snap.wt[li_after]) |*v, b| {
                v.* = @floatCast(@as(f32, @floatCast(v.*)) - @as(f32, @floatCast(b)));
            }
            for (sb_after, snap.sb[li_after]) |*v, b| {
                v.* = @floatCast(@as(f32, @floatCast(v.*)) - @as(f32, @floatCast(b)));
            }
            for (tb_after, snap.tb[li_after]) |*v, b| {
                v.* = @floatCast(@as(f32, @floatCast(v.*)) - @as(f32, @floatCast(b)));
            }

            try self.averageDeltaInPlace(ws_after);
            try self.averageDeltaInPlace(wt_after);
            try self.averageDeltaInPlace(sb_after);
            try self.averageDeltaInPlace(tb_after);

            try self.applyDeltaToLayer(li_after, snap.ws[li_after], ws_after, .s);
            try self.applyDeltaToLayer(li_after, snap.wt[li_after], wt_after, .t);
            try self.applyBiasDeltaToLayer(li_after, snap.sb[li_after], sb_after, .s);
            try self.applyBiasDeltaToLayer(li_after, snap.tb[li_after], tb_after, .t);
        }
        try self.accelerator.sync();

        self.propagateEmbeddingGradients(&inputs, &targets, token_lists.items, effective_batch_size, max_seq_len) catch {};

        self.runCoreRelationalPass(token_lists.items);

        var loss_arr = [1]f32{@as(f32, @floatCast(loss_f16))};
        try self.allReduceFloat32Values(loss_arr[0..]);
        const inv_w: f32 = 1.0 / @as(f32, @floatFromInt(self.coordinator.world_size));
        const final_loss = loss_arr[0] * inv_w;
        if (!std.math.isFinite(final_loss)) return error.InvalidLoss;
        return final_loss;
    }

    pub fn saveCheckpoint(self: *DistributedTrainerFuthark, path: []const u8) !void {
        if (!self.coordinator.isRoot()) {
            try self.coordinator.synchronize();
            return;
        }

        try self.coordinator.synchronize();
        try self.accelerator.sync();

        const tmp_path = try std.fmt.allocPrint(self.allocator, "{s}.tmp", .{path});
        defer self.allocator.free(tmp_path);

        const file = createWriteFile(tmp_path) catch |err| {
            std.debug.print("Failed to create checkpoint file: {}\n", .{err});
            return err;
        };
        var file_closed = false;
        defer if (!file_closed) file.close();

        var buffered_writer = std.io.bufferedWriter(file.writer());
        var writer = buffered_writer.writer();

        try writer.writeInt(u32, self.config.checkpoint_version, .little);
        try writer.writeInt(u64, self.global_step, .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.model_dim)), .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.num_layers)), .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.vocab_size)), .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.local_batch_size)), .little);
        try writeF32(writer, self.learning_rate);
        try writeF32(writer, self.momentum);

        var li_save: usize = 0;
        while (li_save < self.num_layers) : (li_save += 1) {
            const weights_s_vals = try self.readLayerMatrix(li_save, .weights_s);
            defer self.allocator.free(weights_s_vals);
            for (weights_s_vals) |w| try writeF32(writer, @floatCast(w));

            const weights_t_vals = try self.readLayerMatrix(li_save, .weights_t);
            defer self.allocator.free(weights_t_vals);
            for (weights_t_vals) |w| try writeF32(writer, @floatCast(w));

            const s_bias_vals = try self.readLayerMatrix(li_save, .s_bias);
            defer self.allocator.free(s_bias_vals);
            for (s_bias_vals) |b| try writeF32(writer, @floatCast(b));

            const t_bias_vals = try self.readLayerMatrix(li_save, .t_bias);
            defer self.allocator.free(t_bias_vals);
            for (t_bias_vals) |b| try writeF32(writer, @floatCast(b));

            const vel_s_vals = try self.readLayerMatrix(li_save, .velocity_s);
            defer self.allocator.free(vel_s_vals);
            for (vel_s_vals) |v| try writeF32(writer, @floatCast(v));

            const vel_t_vals = try self.readLayerMatrix(li_save, .velocity_t);
            defer self.allocator.free(vel_t_vals);
            for (vel_t_vals) |v| try writeF32(writer, @floatCast(v));

            const vel_sb_vals = try self.readLayerMatrix(li_save, .velocity_sb);
            defer self.allocator.free(vel_sb_vals);
            for (vel_sb_vals) |v| try writeF32(writer, @floatCast(v));

            const vel_tb_vals = try self.readLayerMatrix(li_save, .velocity_tb);
            defer self.allocator.free(vel_tb_vals);
            for (vel_tb_vals) |v| try writeF32(writer, @floatCast(v));
        }

        try writeF32(writer, @as(f32, @floatCast(self.accelerator.clip_min)));
        try writeF32(writer, @as(f32, @floatCast(self.accelerator.clip_max)));

        const node_count: u32 = @intCast(self.nsir_graph.nodes.count());
        try writer.writeInt(u32, node_count, .little);

        var node_it = self.nsir_graph.nodes.iterator();
        while (node_it.next()) |entry| {
            const node = entry.value_ptr.*;
            const id_len: u32 = @intCast(node.id.len);
            try writer.writeInt(u32, id_len, .little);
            try writer.writeAll(node.id);
            try writer.writeAll(std.mem.asBytes(&node.qubit.a.re));
            try writer.writeAll(std.mem.asBytes(&node.qubit.a.im));
            try writer.writeAll(std.mem.asBytes(&node.qubit.b.re));
            try writer.writeAll(std.mem.asBytes(&node.qubit.b.im));
        }

        const edge_key_count: u32 = @intCast(self.nsir_graph.edges.count());
        try writer.writeInt(u32, edge_key_count, .little);

        var edge_it = self.nsir_graph.edges.iterator();
        while (edge_it.next()) |entry| {
            const key = entry.key_ptr.*;
            const edge_list = entry.value_ptr.*;

            const src_len: u32 = @intCast(key.source.len);
            try writer.writeInt(u32, src_len, .little);
            try writer.writeAll(key.source);

            const tgt_len: u32 = @intCast(key.target.len);
            try writer.writeInt(u32, tgt_len, .little);
            try writer.writeAll(key.target);

            const count: u32 = @intCast(edge_list.items.len);
            try writer.writeInt(u32, count, .little);

            for (edge_list.items) |edge| {
                try writer.writeAll(std.mem.asBytes(&edge.weight));
                try writer.writeByte(@intFromEnum(edge.quality));
            }
        }

        try buffered_writer.flush();
        try file.sync();
        file.close();
        file_closed = true;

        if (std.fs.path.isAbsolute(path)) {
            try std.fs.renameAbsolute(tmp_path, path);
        } else {
            try std.fs.cwd().rename(tmp_path, path);
        }

        const tok_path = try std.fmt.allocPrint(self.allocator, "{s}.tokenizer", .{path});
        defer self.allocator.free(tok_path);
        try self.tokenizer.saveVocab(tok_path);

        std.debug.print("Checkpoint saved to {s} at step {d}\n", .{ path, self.global_step });
    }

    pub fn loadCheckpoint(self: *DistributedTrainerFuthark, path: []const u8) !void {
        const file = openReadFile(path) catch |err| {
            std.debug.print("Failed to open checkpoint file: {}\n", .{err});
            return err;
        };
        defer file.close();

        var buf_reader = std.io.bufferedReader(file.reader());
        var reader = buf_reader.reader();

        const version = try reader.readInt(u32, .little);
        if (version != self.config.checkpoint_version) {
            return error.CheckpointVersionMismatch;
        }

        const saved_global_step = try reader.readInt(u64, .little);
        const saved_model_dim_u64 = try reader.readInt(u64, .little);
        const saved_num_layers_u64 = try reader.readInt(u64, .little);
        const saved_vocab_size_u64 = try reader.readInt(u64, .little);
        const saved_local_batch_size_u64 = try reader.readInt(u64, .little);
        const saved_learning_rate = try readF32(reader);
        const saved_momentum = try readF32(reader);

        const saved_model_dim: usize = std.math.cast(usize, saved_model_dim_u64) orelse return error.ModelDimMismatch;
        const saved_num_layers: usize = std.math.cast(usize, saved_num_layers_u64) orelse return error.NumLayersMismatch;
        const saved_vocab_size: usize = std.math.cast(usize, saved_vocab_size_u64) orelse return error.VocabSizeMismatch;
        const saved_local_batch_size: usize = std.math.cast(usize, saved_local_batch_size_u64) orelse return error.InvalidBatchSize;

        if (saved_model_dim != self.model_dim) return error.ModelDimMismatch;
        if (saved_num_layers != self.num_layers) return error.NumLayersMismatch;
        _ = saved_vocab_size;
        _ = saved_local_batch_size;

        try validateHyperparameters(saved_learning_rate, saved_momentum);
        self.learning_rate = saved_learning_rate;
        self.momentum = saved_momentum;
        self.global_step = saved_global_step;

        const half: usize = self.model_dim / 2;
        const weight_count = try std.math.mul(usize, half, half);

        var li_load: usize = 0;
        while (li_load < self.num_layers) : (li_load += 1) {
            const s_weights = try self.allocator.alloc(f16, weight_count);
            defer self.allocator.free(s_weights);
            for (s_weights) |*w| {
                const v = try readF32(reader);
                if (!std.math.isFinite(v)) return error.InvalidWeightValue;
                w.* = @floatCast(v);
            }

            const t_weights = try self.allocator.alloc(f16, weight_count);
            defer self.allocator.free(t_weights);
            for (t_weights) |*w| {
                const v = try readF32(reader);
                if (!std.math.isFinite(v)) return error.InvalidWeightValue;
                w.* = @floatCast(v);
            }

            try self.accelerator.setLayerWeightsS(li_load, s_weights, half, half);
            try self.accelerator.setLayerWeightsT(li_load, t_weights, half, half);

            const s_bias_data = try self.allocator.alloc(f16, half);
            defer self.allocator.free(s_bias_data);
            for (s_bias_data) |*b| {
                const v = try readF32(reader);
                if (!std.math.isFinite(v)) return error.InvalidWeightValue;
                b.* = @floatCast(v);
            }

            const t_bias_data = try self.allocator.alloc(f16, half);
            defer self.allocator.free(t_bias_data);
            for (t_bias_data) |*b| {
                const v = try readF32(reader);
                if (!std.math.isFinite(v)) return error.InvalidWeightValue;
                b.* = @floatCast(v);
            }

            try self.accelerator.setLayerSBias(li_load, s_bias_data, half);
            try self.accelerator.setLayerTBias(li_load, t_bias_data, half);

            const vel_s = try self.allocator.alloc(f16, weight_count);
            defer self.allocator.free(vel_s);
            for (vel_s) |*w| {
                const v = try readF32(reader);
                if (!std.math.isFinite(v)) return error.InvalidWeightValue;
                w.* = @floatCast(v);
            }

            const vel_t = try self.allocator.alloc(f16, weight_count);
            defer self.allocator.free(vel_t);
            for (vel_t) |*w| {
                const v = try readF32(reader);
                if (!std.math.isFinite(v)) return error.InvalidWeightValue;
                w.* = @floatCast(v);
            }

            const vel_sb = try self.allocator.alloc(f16, half);
            defer self.allocator.free(vel_sb);
            for (vel_sb) |*w| {
                const v = try readF32(reader);
                if (!std.math.isFinite(v)) return error.InvalidWeightValue;
                w.* = @floatCast(v);
            }

            const vel_tb = try self.allocator.alloc(f16, half);
            defer self.allocator.free(vel_tb);
            for (vel_tb) |*w| {
                const v = try readF32(reader);
                if (!std.math.isFinite(v)) return error.InvalidWeightValue;
                w.* = @floatCast(v);
            }

            try self.accelerator.setLayerVelocityS(li_load, vel_s, half, half);
            try self.accelerator.setLayerVelocityT(li_load, vel_t, half, half);
            try self.accelerator.setLayerVelocitySB(li_load, vel_sb, half);
            try self.accelerator.setLayerVelocityTB(li_load, vel_tb, half);
        }

        const clip_min_f32 = try readF32(reader);
        const clip_max_f32 = try readF32(reader);
        if (!std.math.isFinite(clip_min_f32) or !std.math.isFinite(clip_max_f32) or !(clip_min_f32 < clip_max_f32)) {
            return error.InvalidClipRange;
        }
        try self.accelerator.setClipRange(@floatCast(clip_min_f32), @floatCast(clip_max_f32));

        try self.accelerator.sync();

        self.nsir_graph.deinit();
        self.nsir_graph = try SelfSimilarRelationalGraph.init(self.allocator);

        const node_count_r = try reader.readInt(u32, .little);
        var ni: u32 = 0;
        while (ni < node_count_r) : (ni += 1) {
            const id_len = try reader.readInt(u32, .little);
            const id = try self.allocator.alloc(u8, id_len);
            errdefer self.allocator.free(id);
            try reader.readNoEof(id);

            var a_re: f64 = undefined;
            var a_im: f64 = undefined;
            var b_re: f64 = undefined;
            var b_im: f64 = undefined;
            try reader.readNoEof(std.mem.asBytes(&a_re));
            try reader.readNoEof(std.mem.asBytes(&a_im));
            try reader.readNoEof(std.mem.asBytes(&b_re));
            try reader.readNoEof(std.mem.asBytes(&b_im));

            const qubit = nsir.Qubit{
                .a = std.math.Complex(f64).init(a_re, a_im),
                .b = std.math.Complex(f64).init(b_re, b_im),
            };
            const node = try nsir.Node.init(self.allocator, id, &.{}, qubit, 0.0);
            try self.nsir_graph.addNode(node);
        }

        const edge_key_count_r = try reader.readInt(u32, .little);
        var ei: u32 = 0;
        while (ei < edge_key_count_r) : (ei += 1) {
            const src_len = try reader.readInt(u32, .little);
            const source = try self.allocator.alloc(u8, src_len);
            try reader.readNoEof(source);

            const tgt_len = try reader.readInt(u32, .little);
            const target = try self.allocator.alloc(u8, tgt_len);
            try reader.readNoEof(target);

            const count = try reader.readInt(u32, .little);
            var k: u32 = 0;
            while (k < count) : (k += 1) {
                var weight: f64 = undefined;
                try reader.readNoEof(std.mem.asBytes(&weight));
                const quality_byte = try reader.readByte();
                const quality: nsir.EdgeQuality = @enumFromInt(quality_byte);
                const edge = try nsir.Edge.init(
                    self.allocator,
                    source,
                    target,
                    quality,
                    weight,
                    std.math.Complex(f64).init(0.0, 0.0),
                    0.0,
                );
                try self.nsir_graph.addEdge(source, target, edge);
            }
        }

        if (self.signal_engine) |*se| se.deinit();
        self.signal_engine = SignalPropagationEngine.init(
            self.allocator,
            &self.nsir_graph,
            &self.crev_kernel.?.flow_analyzer,
        );

        const tok_path = try std.fmt.allocPrint(self.allocator, "{s}.tokenizer", .{path});
        defer self.allocator.free(tok_path);
        try self.tokenizer.loadVocab(tok_path);
        self.vocab_size = self.tokenizer.next_token_id;
        try self.reinitEmbedding();

        std.debug.print("Checkpoint loaded from {s} at step {d}\n", .{ path, self.global_step });
    }

    fn propagateEmbeddingGradients(
        self: *DistributedTrainerFuthark,
        inputs: *FutharkArray3DF16,
        targets: *FutharkArray3DF16,
        token_items: []const std.ArrayList(u32),
        effective_batch_size: usize,
        max_seq_len: usize,
    ) !void {
        if (self.embedding == null) return;
        const emb = &self.embedding.?;
        const fctx = self.accelerator.ctx.ctx;
        if (fctx == null) return;
        const cmin: u16 = @bitCast(self.accelerator.clip_min);
        const cmax: u16 = @bitCast(self.accelerator.clip_max);

        var re_act = try std.heap.page_allocator.alloc(?*futhark.struct_futhark_f16_3d, self.num_layers + 1);
        errdefer std.heap.page_allocator.free(re_act);
        defer std.heap.page_allocator.free(re_act);
        var re_own = try std.heap.page_allocator.alloc(bool, self.num_layers + 1);
        errdefer std.heap.page_allocator.free(re_own);
        defer std.heap.page_allocator.free(re_own);
        @memset(re_act, @as(?*futhark.struct_futhark_f16_3d, null));
        @memset(re_own, false);
        re_act[0] = inputs.arr;
        re_own[0] = false;

        var fi: usize = 0;
        while (fi < self.num_layers) : (fi += 1) {
            var fwd_out: ?*futhark.struct_futhark_f16_3d = null;
            const fwd_rc = futhark.futhark_entry_batch_forward(
                fctx,
                &fwd_out,
                re_act[fi],
                self.accelerator.layers[fi].weights_s.arr,
                self.accelerator.layers[fi].weights_t.arr,
                self.accelerator.layers[fi].s_bias.arr,
                self.accelerator.layers[fi].t_bias.arr,
                cmin,
                cmax,
            );
            if (fwd_rc != 0 or fwd_out == null) {
                var ci: usize = 0;
                while (ci < re_act.len) : (ci += 1) {
                    if (re_own[ci] and re_act[ci] != null) _ = futhark.futhark_free_f16_3d(fctx, re_act[ci]);
                }
                return;
            }
            var oftb_out: ?*futhark.struct_futhark_f16_3d = null;
            const oftb_rc = futhark.futhark_entry_batch_oftb_forward(fctx, &oftb_out, fwd_out);
            _ = futhark.futhark_free_f16_3d(fctx, fwd_out);
            if (oftb_rc != 0 or oftb_out == null) {
                var ci: usize = 0;
                while (ci < re_act.len) : (ci += 1) {
                    if (re_own[ci] and re_act[ci] != null) _ = futhark.futhark_free_f16_3d(fctx, re_act[ci]);
                }
                return;
            }
            re_act[fi + 1] = oftb_out;
            re_own[fi + 1] = true;
        }
        try self.accelerator.sync();

        var grad: ?*futhark.struct_futhark_f16_3d = null;
        const grad_rc = futhark.futhark_entry_compute_initial_grad_l2(fctx, &grad, re_act[self.num_layers], targets.arr);
        if (grad_rc != 0 or grad == null) {
            var ci: usize = 0;
            while (ci < re_act.len) : (ci += 1) {
                if (re_own[ci] and re_act[ci] != null) _ = futhark.futhark_free_f16_3d(fctx, re_act[ci]);
            }
            return;
        }

        var lb: usize = self.num_layers;
        while (lb > 0) : (lb -= 1) {
            var oftb_g: ?*futhark.struct_futhark_f16_3d = null;
            _ = futhark.futhark_entry_batch_oftb_backward(fctx, &oftb_g, grad);
            _ = futhark.futhark_free_f16_3d(fctx, grad);
            grad = oftb_g;
            if (grad == null) {
                var ci: usize = 0;
                while (ci < re_act.len) : (ci += 1) {
                    if (re_own[ci] and re_act[ci] != null) _ = futhark.futhark_free_f16_3d(fctx, re_act[ci]);
                }
                return;
            }

            var tup: ?*futhark.struct_futhark_opaque_tup5_grad_full = null;
            _ = futhark.futhark_entry_batch_gradients_full(
                fctx,
                &tup,
                re_act[lb - 1],
                grad,
                self.accelerator.layers[lb - 1].weights_s.arr,
                self.accelerator.layers[lb - 1].weights_t.arr,
                self.accelerator.layers[lb - 1].s_bias.arr,
                self.accelerator.layers[lb - 1].t_bias.arr,
                cmin,
                cmax,
            );
            _ = futhark.futhark_free_f16_3d(fctx, grad);
            grad = null;

            if (tup) |t| {
                var gws: ?*futhark.struct_futhark_f16_2d = null;
                var gwt: ?*futhark.struct_futhark_f16_2d = null;
                var gsb: ?*futhark.struct_futhark_f16_1d = null;
                var gtb: ?*futhark.struct_futhark_f16_1d = null;
                var gin: ?*futhark.struct_futhark_f16_3d = null;
                _ = futhark.futhark_project_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16_0(fctx, &gws, t);
                _ = futhark.futhark_project_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16_1(fctx, &gwt, t);
                _ = futhark.futhark_project_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16_2(fctx, &gsb, t);
                _ = futhark.futhark_project_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16_3(fctx, &gtb, t);
                _ = futhark.futhark_project_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16_4(fctx, &gin, t);
                _ = futhark.futhark_free_opaque_tup5_arr2d_f16_arr2d_f16_arr1d_f16_arr1d_f16_arr3d_f16(fctx, t);
                if (gws != null) _ = futhark.futhark_free_f16_2d(fctx, gws);
                if (gwt != null) _ = futhark.futhark_free_f16_2d(fctx, gwt);
                if (gsb != null) _ = futhark.futhark_free_f16_1d(fctx, gsb);
                if (gtb != null) _ = futhark.futhark_free_f16_1d(fctx, gtb);
                grad = gin;
            }
        }
        try self.accelerator.sync();

        var ai: usize = 0;
        while (ai < re_act.len) : (ai += 1) {
            if (re_own[ai] and re_act[ai] != null) {
                _ = futhark.futhark_free_f16_3d(fctx, re_act[ai]);
            }
        }

        if (grad) |gi| {
            const total = try std.math.mul(usize, try std.math.mul(usize, effective_batch_size, max_seq_len), self.model_dim);
            var host_f16 = std.heap.page_allocator.alloc(f16, total) catch {
                _ = futhark.futhark_free_f16_3d(fctx, gi);
                return;
            };
            defer std.heap.page_allocator.free(host_f16);

            _ = futhark.futhark_values_f16_3d(fctx, gi, @ptrCast(host_f16.ptr));
            _ = futhark.futhark_free_f16_3d(fctx, gi);
            try self.accelerator.sync();

            var b_idx: usize = 0;
            while (b_idx < token_items.len) : (b_idx += 1) {
                const list = token_items[b_idx].items;
                if (list.len == 0) continue;
                const seq_len = @min(list.len, max_seq_len);
                const grad_len = try std.math.mul(usize, seq_len, self.model_dim);
                var grad_data = self.allocator.alloc(f32, grad_len) catch continue;
                defer self.allocator.free(grad_data);

                var s: usize = 0;
                while (s < seq_len) : (s += 1) {
                    var c: usize = 0;
                    while (c < self.model_dim) : (c += 1) {
                        const src = b_idx * max_seq_len * self.model_dim + s * self.model_dim + c;
                        const dst = s * self.model_dim + c;
                        if (src < host_f16.len and dst < grad_data.len) {
                            grad_data[dst] = @floatCast(host_f16[src]);
                        }
                    }
                }

                var gfc = sfd.GradientFlowController.init(self.allocator, grad_data.len, 1.0, 0.1) catch {
                    emb.backward(list, grad_data, max_seq_len);
                    emb.applyGradients(self.learning_rate, self.momentum);
                    continue;
                };
                defer gfc.deinit();
                gfc.clipAndNormalize(grad_data) catch {};

                emb.backward(list, grad_data, max_seq_len);
                emb.applyGradients(self.learning_rate, self.momentum);
            }
        }
    }

    pub fn enableEmbeddingAccelerator(self: *DistributedTrainerFuthark) !void {
        if (self.embedding_accel != null) return;
        self.embedding_accel = try EmbeddingAccelerator.init(
            &self.accelerator.ctx,
            self.vocab_size,
            self.model_dim,
            42,
        );
    }

    pub fn buildKnowledgeGraph(self: *DistributedTrainerFuthark, text: []const u8) !void {
        if (self.crev_pipeline) |*cp| {
            _ = try cp.processTextStream(text);
        }
        const text_bytes = std.mem.sliceAsBytes(text);
        _ = self.nsir_graph.encodeInformation(text_bytes) catch {};
    }
};
