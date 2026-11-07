const std = @import("std");

const engine_sources = [_][]const u8{
    "src/dict.c",
    "src/board.c",
    "src/movegen.c",
    "src/eval.c",
    "src/nnue.c",
    "src/zobrist.c",
    "src/palimpsest.c",
    "src/lattice.c",
    "src/settle.c",
    "src/broker.c",
    "src/court.c",
    "src/simulation.c",
    "src/protocol.c",
    "src/endgame.c",
    "src/bayes_rack.c",
    "src/bot_archetypes.c",
};

const c_flags = [_][]const u8{
    "-std=c11",
    "-Wall",
    "-Wextra",
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // 1. Core Engine Executable (tessera)
    const engine_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    engine_mod.addIncludePath(b.path("include"));
    engine_mod.addCSourceFiles(.{
        .files = &engine_sources,
        .flags = &c_flags,
    });
    engine_mod.addCSourceFile(.{
        .file = b.path("src/main.c"),
        .flags = &c_flags,
    });

    const exe = b.addExecutable(.{
        .name = "tessera",
        .root_module = engine_mod,
    });

    b.installArtifact(exe);

    // 2. 'run' step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    const run_step = b.step("run", "Run the Tessera engine");
    run_step.dependOn(&run_cmd.step);

    // 3. 'bench' step
    const bench_cmd = b.addRunArtifact(exe);
    bench_cmd.addArg("bench");
    bench_cmd.step.dependOn(b.getInstallStep());
    const bench_step = b.step("bench", "Run the engine performance benchmark");
    bench_step.dependOn(&bench_cmd.step);

    // 4. Test Suite steps
    const test_step = b.step("test", "Build and run the complete test suite");

    const tests = [_]struct { name: []const u8, file: []const u8 }{
        .{ .name = "test_dict", .file = "tests/test_dict.c" },
        .{ .name = "test_movegen", .file = "tests/test_movegen.c" },
        .{ .name = "test_eval", .file = "tests/test_eval.c" },
        .{ .name = "test_lattice_proof", .file = "tests/test_lattice_proof.c" },
        .{ .name = "test_endgame", .file = "tests/test_endgame.c" },
    };

    for (tests) |t| {
        const test_mod = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        test_mod.addIncludePath(b.path("include"));
        test_mod.addCSourceFiles(.{
            .files = &engine_sources,
            .flags = &c_flags,
        });
        test_mod.addCSourceFile(.{
            .file = b.path(t.file),
            .flags = &c_flags,
        });

        const test_exe = b.addExecutable(.{
            .name = t.name,
            .root_module = test_mod,
        });

        const run_test = b.addRunArtifact(test_exe);
        test_step.dependOn(&run_test.step);
    }
}
