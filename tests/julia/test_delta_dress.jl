"""
Tests for the Julia Δ^k-DRESS module.

Run from the repo root:
    julia tests/julia/test_delta_dress.jl

Or from the tests/julia/ directory:
    julia test_delta_dress.jl
"""

# ── load module ────────────────────────────────────────────────────────

const THIS_DIR = @__DIR__
include(joinpath(THIS_DIR, "..", "..", "julia", "src", "DRESS.jl"))
using .DRESS
using Test

# ── helpers ──────────────────────────────────────────────────────────

hist_total(r) = sum((entry.count for entry in r.histogram); init=Int64(0))

function hist_count_value(r, value)
    for entry in r.histogram
        if isapprox(entry.value, value; atol=1e-9)
            return entry.count
        end
    end
    return 0
end

K3_SRC = Int32[0, 1, 0]
K3_TGT = Int32[1, 2, 2]
K4_SRC = Int32[0, 0, 0, 1, 1, 2]
K4_TGT = Int32[1, 2, 3, 2, 3, 3]
P4_SRC = Int32[0, 1, 2]
P4_TGT = Int32[1, 2, 3]

const EPS = 1e-3

# ── tests ────────────────────────────────────────────────────────────

@testset "Δ^k-DRESS" begin

    @testset "histogram size" begin
        r = delta_fit(3, K3_SRC, K3_TGT; k=0, epsilon=1e-3)
        @test length(r.histogram) == 1

        r2 = delta_fit(3, K3_SRC, K3_TGT; k=0, epsilon=1e-6)
        @test length(r2.histogram) == 1

        @test r.histogram == r2.histogram
    end

    @testset "weighted histogram size" begin
        # Non-uniform weights should produce multiple exact values.
        r = delta_fit(3, K3_SRC, K3_TGT;
                            weights=Float64[1.0, 10.0, 1.0],
                            k=0, epsilon=1e-3)
        @test length(r.histogram) > 1
        @test hist_total(r) == 3
    end

    @testset "result type" begin
        r = delta_fit(3, K3_SRC, K3_TGT; k=0, epsilon=EPS)
        @test r isa DeltaDRESSResult
        # show method should contain "DeltaDRESSResult"
        io = IOBuffer()
        show(io, r)
        s = String(take!(io))
        @test occursin("DeltaDRESSResult", s)
    end

    @testset "Δ^0 on K3" begin
        r = delta_fit(3, K3_SRC, K3_TGT; k=0, epsilon=EPS)
        @test hist_total(r) == 3
        @test length(r.histogram) == 1
        @test hist_count_value(r, 2.0) == 3
    end

    @testset "Δ^1 on K3" begin
        r = delta_fit(3, K3_SRC, K3_TGT; k=1, epsilon=EPS)
        # C(3,1)=3 subgraphs * 1 edge = 3
        @test hist_total(r) == 3
    end

    @testset "Δ^2 on K3" begin
        r = delta_fit(3, K3_SRC, K3_TGT; k=2, epsilon=EPS)
        @test hist_total(r) == 0
    end

    @testset "Δ^0 on K4" begin
        r = delta_fit(4, K4_SRC, K4_TGT; k=0, epsilon=EPS)
        @test hist_total(r) == 6
        @test length(r.histogram) == 1
        @test hist_count_value(r, 2.0) == 6
    end

    @testset "Δ^1 on K4" begin
        r = delta_fit(4, K4_SRC, K4_TGT; k=1, epsilon=EPS)
        # C(4,1)=4 * 3 edges = 12
        @test hist_total(r) == 12
        @test length(r.histogram) == 1
        @test hist_count_value(r, 2.0) == 12
    end

    @testset "Δ^2 on K4" begin
        r = delta_fit(4, K4_SRC, K4_TGT; k=2, epsilon=EPS)
        # C(4,2)=6 * 1 edge = 6
        @test hist_total(r) == 6
    end

    @testset "k ≥ N" begin
        r1 = delta_fit(3, K3_SRC, K3_TGT; k=3, epsilon=EPS)
        @test hist_total(r1) == 0

        r2 = delta_fit(3, K3_SRC, K3_TGT; k=10, epsilon=EPS)
        @test hist_total(r2) == 0
    end

    @testset "precompute flag" begin
        r1 = delta_fit(4, K4_SRC, K4_TGT; k=1, epsilon=EPS,
                             precompute=false)
        r2 = delta_fit(4, K4_SRC, K4_TGT; k=1, epsilon=EPS,
                             precompute=true)
        @test r1.histogram == r2.histogram
    end

    @testset "path P4" begin
        r = delta_fit(4, P4_SRC, P4_TGT; k=0, epsilon=EPS)
        @test hist_total(r) == 3
        @test length(r.histogram) >= 2
    end

    @testset "Δ^1 on P4" begin
        r = delta_fit(4, P4_SRC, P4_TGT; k=1, epsilon=EPS)
        # Remove 0 → 2 edges, remove 1 → 1 edge,
        # remove 2 → 1 edge, remove 3 → 2 edges = 6 total
        @test hist_total(r) == 6
    end

    @testset "argument validation" begin
        @test_throws ArgumentError delta_fit(3, Int32[0, 1], Int32[1, 2, 2])
    end

    # ── multisets ──────────────────────────────────────────────────

    @testset "multisets disabled" begin
        r = delta_fit(3, K3_SRC, K3_TGT; k=0, epsilon=EPS)
        @test r.multisets === nothing
    end

    @testset "multisets Δ^0 K3" begin
        r = delta_fit(3, K3_SRC, K3_TGT; k=0, epsilon=EPS,
                            keep_multisets=true)
        @test r.num_subgraphs == 1
        @test size(r.multisets) == (1, 3)
        for v in r.multisets
            @test abs(v - 2.0) < EPS
        end
    end

    @testset "multisets Δ^1 K3 NaN pattern" begin
        r = delta_fit(3, K3_SRC, K3_TGT; k=1, epsilon=EPS,
                            keep_multisets=true)
        @test r.num_subgraphs == 3
        @test size(r.multisets) == (3, 3)
        for row in 1:3
            nans = count(isnan, r.multisets[row, :])
            @test nans == 2
            vals = filter(!isnan, r.multisets[row, :])
            @test length(collect(vals)) == 1
            @test abs(first(vals) - 2.0) < EPS
        end
    end

end
