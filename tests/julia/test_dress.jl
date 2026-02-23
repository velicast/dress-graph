"""
Tests for the Julia DRESS module.

Run from the repo root:
    julia tests/julia/test_dress.jl

Or from the tests/julia/ directory:
    julia test_dress.jl
"""

# ── load module ────────────────────────────────────────────────────────

const THIS_DIR = @__DIR__
include(joinpath(THIS_DIR, "..", "..", "julia", "src", "DRESS.jl"))
using .DRESS
using Test

# ── helpers ──────────────────────────────────────────────────────────

"""Build a triangle graph: 0-1, 1-2, 0-2"""
triangle() = dress_fit(3, Int32[0, 1, 0], Int32[1, 2, 2])

"""Build a path graph: 0-1-2-3"""
path4() = dress_fit(4, Int32[0, 1, 2], Int32[1, 2, 3])

# ── tests ────────────────────────────────────────────────────────────

@testset "DRESS.jl" begin

    # ── construction ─────────────────────────────────────────────────
    @testset "construction" begin
        @testset "unweighted triangle" begin
            r = triangle()
            @test length(r.sources) == 3
            @test length(r.targets) == 3
            @test length(r.edge_dress) == 3
            @test length(r.node_dress) == 3
            @test r.iterations > 0
        end

        @testset "weighted triangle" begin
            r = dress_fit(3, Int32[0, 1, 0], Int32[1, 2, 2];
                          weights=[1.0, 2.0, 3.0])
            @test length(r.edge_dress) == 3
            # Undirected weighted edges get doubled
            @test r.edge_weight[1] ≈ 2.0 atol=1e-12
            @test r.edge_weight[2] ≈ 4.0 atol=1e-12
            @test r.edge_weight[3] ≈ 6.0 atol=1e-12
        end

        @testset "all variants" begin
            for v in (UNDIRECTED, DIRECTED, FORWARD, BACKWARD)
                r = dress_fit(3, Int32[0, 1, 0], Int32[1, 2, 2]; variant=v)
                @test length(r.edge_dress) == 3
                @test r.iterations >= 0
            end
        end

        @testset "precompute intercepts" begin
            r1 = dress_fit(3, Int32[0, 1, 0], Int32[1, 2, 2];
                           precompute_intercepts=true)
            r2 = dress_fit(3, Int32[0, 1, 0], Int32[1, 2, 2];
                           precompute_intercepts=false)
            @test length(r1.edge_dress) == 3
            @test length(r2.edge_dress) == 3
        end

        @testset "argument validation" begin
            @test_throws ArgumentError dress_fit(3, Int32[0, 1], Int32[1, 2, 2])
            @test_throws ArgumentError dress_fit(3, Int32[0, 1, 0], Int32[1, 2, 2];
                                                  weights=[1.0, 2.0])
        end
    end

    # ── fitting ──────────────────────────────────────────────────────
    @testset "fitting" begin
        @testset "triangle convergence" begin
            r = triangle()
            @test r.iterations > 0
            @test r.delta >= 0.0
        end

        @testset "triangle equal dress" begin
            r = triangle()
            d0 = r.edge_dress[1]
            for e in 2:3
                @test r.edge_dress[e] ≈ d0 atol=1e-6
            end
        end

        @testset "path positive dress" begin
            r = path4()
            for e in 1:3
                @test r.edge_dress[e] > 0.0
                @test r.edge_dress[e] < 2.0
            end
        end

        @testset "path symmetry" begin
            r = path4()
            # Endpoint edges (0-1 and 2-3) should be symmetric
            @test r.edge_dress[1] ≈ r.edge_dress[3] atol=1e-10
        end

        @testset "intercepts match no-intercepts" begin
            r1 = dress_fit(3, Int32[0, 1, 0], Int32[1, 2, 2];
                           max_iterations=100, epsilon=1e-10,
                           precompute_intercepts=true)
            r2 = dress_fit(3, Int32[0, 1, 0], Int32[1, 2, 2];
                           max_iterations=100, epsilon=1e-10,
                           precompute_intercepts=false)
            for e in 1:3
                @test r1.edge_dress[e] ≈ r2.edge_dress[e] atol=1e-8
            end
        end

        @testset "node dress" begin
            r = triangle()
            for u in 1:3
                @test r.node_dress[u] > 0.0
            end
            # K3: all nodes equal
            @test r.node_dress[1] ≈ r.node_dress[2] atol=1e-6
            @test r.node_dress[1] ≈ r.node_dress[3] atol=1e-6
        end

        @testset "weighted fit" begin
            r = dress_fit(3, Int32[0, 1, 0], Int32[1, 2, 2];
                          weights=[1.0, 2.0, 3.0])
            @test r.iterations > 0
            # Asymmetric weights → different dress values
            d_min = minimum(r.edge_dress)
            d_max = maximum(r.edge_dress)
            @test d_max - d_min > 1e-6
        end
    end

    # ── edge cases ───────────────────────────────────────────────────
    @testset "edge cases" begin
        @testset "single edge" begin
            r = dress_fit(2, Int32[0], Int32[1])
            @test length(r.edge_dress) == 1
            @test r.edge_dress[1] > 0.0
        end

        @testset "complete K4" begin
            r = dress_fit(4,
                Int32[0, 0, 0, 1, 1, 2],
                Int32[1, 2, 3, 2, 3, 3];
                max_iterations=200, epsilon=1e-10)
            d0 = r.edge_dress[1]
            for e in 2:6
                @test r.edge_dress[e] ≈ d0 atol=1e-6
            end
            # All nodes equal
            for u in 2:4
                @test r.node_dress[u] ≈ r.node_dress[1] atol=1e-6
            end
        end

        @testset "star graph" begin
            r = dress_fit(5,
                Int32[0, 0, 0, 0],
                Int32[1, 2, 3, 4])
            d0 = r.edge_dress[1]
            for e in 2:4
                @test r.edge_dress[e] ≈ d0 atol=1e-6
            end
        end
    end

    # ── DRESSResult display ──────────────────────────────────────────
    @testset "DRESSResult show" begin
        r = triangle()
        s = sprint(show, r)
        @test occursin("DRESSResult", s)
        @test occursin("E=3", s)
    end

end  # top-level testset
