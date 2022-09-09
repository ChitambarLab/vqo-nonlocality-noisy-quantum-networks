"""
This Julia script verifies the partially classical strategies that
are optimal for quantum non-n-locality in star and chain networks.

To run script:

    1. Install Julia at https://julialang.org/downloads/
    2. Launch the Julia REPL
    3. Install the QBase.jl packages: julia> using Pkg; Pkg.add("QBase")
    4. Run the script:

        julia> include("script/proof_numerics/verify_classical_non-n-local_strategies.jl")
"""

using QBase
using LinearAlgebra
using Test

σ_plus = Unitary((σz + σx)/sqrt(2))
σ_minus = Unitary((σz - σx)/sqrt(2))

ρ_00 = State([1 0 0 0;0 0 0 0;0 0 0 0;0 0 0 0])
ϕ_plus = State([1 0 0 1;0 0 0 0;0 0 0 0;1 0 0 1]/2)

A_0, A_1 = σ_plus, σ_minus
B_0, B_1 = σz, σx

A_cl_0, A_cl_1 = σz, σx
B_cl_0, B_cl_1 = σz, σz

@testset "optimal classical strategy on interior chain nodes" begin

    @testset "trilocal chain network" begin
        I22 = sum([
            tr(kron(A_0,B_0,B_cl_0,B_cl_0,B_0,A_0) * kron(ϕ_plus, ρ_00, ϕ_plus)),
            tr(kron(A_0,B_0,B_cl_0,B_cl_0,B_0,A_1) * kron(ϕ_plus, ρ_00, ϕ_plus)),
            tr(kron(A_1,B_0,B_cl_0,B_cl_0,B_0,A_0) * kron(ϕ_plus, ρ_00, ϕ_plus)),
            tr(kron(A_1,B_0,B_cl_0,B_cl_0,B_0,A_1) * kron(ϕ_plus, ρ_00, ϕ_plus)),
        ])/2^2

        J22 = sum([
            tr(kron(A_0,B_1,B_cl_1,B_cl_1,B_1,A_0) * kron(ϕ_plus, ρ_00, ϕ_plus)),
            -tr(kron(A_0,B_1,B_cl_1,B_cl_1,B_1,A_1) * kron(ϕ_plus, ρ_00, ϕ_plus)),
            -tr(kron(A_1,B_1,B_cl_1,B_cl_1,B_1,A_0) * kron(ϕ_plus, ρ_00, ϕ_plus)),
            tr(kron(A_1,B_1,B_cl_1,B_cl_1,B_1,A_1) * kron(ϕ_plus, ρ_00, ϕ_plus)),
        ])/2^2

        S_3chain = sqrt(I22) + sqrt(J22)
        @test S_3chain ≈ sqrt(2)
    end

    @testset "4-local chain network" begin
        I22 = sum([
            tr(kron(A_0,B_0,B_cl_0,B_cl_0,B_cl_0,B_cl_0,B_0,A_0) * kron(ϕ_plus, ρ_00, ρ_00, ϕ_plus)),
            tr(kron(A_0,B_0,B_cl_0,B_cl_0,B_cl_0,B_cl_0,B_0,A_1) * kron(ϕ_plus, ρ_00, ρ_00, ϕ_plus)),
            tr(kron(A_1,B_0,B_cl_0,B_cl_0,B_cl_0,B_cl_0,B_0,A_0) * kron(ϕ_plus, ρ_00, ρ_00, ϕ_plus)),
            tr(kron(A_1,B_0,B_cl_0,B_cl_0,B_cl_0,B_cl_0,B_0,A_1) * kron(ϕ_plus, ρ_00, ρ_00, ϕ_plus)),
        ])/2^2

        J22 = sum([
            tr(kron(A_0,B_1,B_cl_1,B_cl_1,B_cl_1,B_cl_1,B_1,A_0) * kron(ϕ_plus, ρ_00, ρ_00, ϕ_plus)),
            -tr(kron(A_0,B_1,B_cl_1,B_cl_1,B_cl_1,B_cl_1,B_1,A_1) * kron(ϕ_plus, ρ_00, ρ_00, ϕ_plus)),
            -tr(kron(A_1,B_1,B_cl_1,B_cl_1,B_cl_1,B_cl_1,B_1,A_0) * kron(ϕ_plus, ρ_00, ρ_00, ϕ_plus)),
            tr(kron(A_1,B_1,B_cl_1,B_cl_1,B_cl_1,B_cl_1,B_1,A_1) * kron(ϕ_plus, ρ_00, ρ_00, ϕ_plus)),
        ])/2^2

        S_4chain = sqrt(I22) + sqrt(J22)
        @test S_4chain ≈ sqrt(2)
    end
end

@testset "optimal classical strategy on single star point" begin

    @testset "bilocal network" begin
        I22 = sum([
            tr(kron(A_cl_0,B_cl_0,A_0,B_0) * kron(ρ_00, ϕ_plus)),
            tr(kron(A_cl_0,B_cl_0,A_1,B_0) * kron(ρ_00, ϕ_plus)),
            tr(kron(A_cl_1,B_cl_0,A_0,B_0) * kron(ρ_00, ϕ_plus)),
            tr(kron(A_cl_1,B_cl_0,A_1,B_0) * kron(ρ_00, ϕ_plus)),
        ])/2^2

        J22 = sum([
            tr(kron(A_cl_0,B_cl_1,A_0,B_1) * kron(ρ_00, ϕ_plus)),
            -tr(kron(A_cl_0,B_cl_1,A_1,B_1) * kron(ρ_00, ϕ_plus)),
            -tr(kron(A_cl_1,B_cl_1,A_0,B_1) * kron(ρ_00, ϕ_plus)),
            tr(kron(A_cl_1,B_cl_1,A_1,B_1) * kron(ρ_00, ϕ_plus)),
        ])/2^2

        S_2star = sqrt(I22) + sqrt(J22)
        @test S_2star ≈ sqrt(2)^(1/2)
    end

    @testset "trilocal star network" begin
        I22 = sum([
            tr(kron(A_cl_0,B_cl_0,A_0,B_0,A_0,B_0) * kron(ρ_00, ϕ_plus, ϕ_plus)),
            tr(kron(A_cl_0,B_cl_0,A_0,B_0,A_1,B_0) * kron(ρ_00, ϕ_plus, ϕ_plus)),
            tr(kron(A_cl_0,B_cl_0,A_1,B_0,A_0,B_0) * kron(ρ_00, ϕ_plus, ϕ_plus)),
            tr(kron(A_cl_0,B_cl_0,A_1,B_0,A_1,B_0) * kron(ρ_00, ϕ_plus, ϕ_plus)),
            tr(kron(A_cl_1,B_cl_0,A_0,B_0,A_0,B_0) * kron(ρ_00, ϕ_plus, ϕ_plus)),
            tr(kron(A_cl_1,B_cl_0,A_0,B_0,A_1,B_0) * kron(ρ_00, ϕ_plus, ϕ_plus)),
            tr(kron(A_cl_1,B_cl_0,A_1,B_0,A_0,B_0) * kron(ρ_00, ϕ_plus, ϕ_plus)),
            tr(kron(A_cl_1,B_cl_0,A_1,B_0,A_1,B_0) * kron(ρ_00, ϕ_plus, ϕ_plus)),
        ])/2^3

        J22 = sum([
            tr(kron(A_cl_0,B_cl_1,A_0,B_1,A_0,B_1) * kron(ρ_00, ϕ_plus, ϕ_plus)),
            -tr(kron(A_cl_0,B_cl_1,A_0,B_1,A_1,B_1) * kron(ρ_00, ϕ_plus, ϕ_plus)),
            -tr(kron(A_cl_0,B_cl_1,A_1,B_1,A_0,B_1) * kron(ρ_00, ϕ_plus, ϕ_plus)),
            tr(kron(A_cl_0,B_cl_1,A_1,B_1,A_1,B_1) * kron(ρ_00, ϕ_plus, ϕ_plus)),
            -tr(kron(A_cl_1,B_cl_1,A_0,B_1,A_0,B_1) * kron(ρ_00, ϕ_plus, ϕ_plus)),
            tr(kron(A_cl_1,B_cl_1,A_0,B_1,A_1,B_1) * kron(ρ_00, ϕ_plus, ϕ_plus)),
            tr(kron(A_cl_1,B_cl_1,A_1,B_1,A_0,B_1) * kron(ρ_00, ϕ_plus, ϕ_plus)),
            -tr(kron(A_cl_1,B_cl_1,A_1,B_1,A_1,B_1) * kron(ρ_00, ϕ_plus, ϕ_plus)),
        ])/2^3

        S_3star = I22^(1/3) + J22^(1/3)
        @test S_3star ≈ sqrt(2)^(2/3)
    end

    @testset "4-local star network" begin
        I22 = sum([
            tr(kron(A_cl_0,B_cl_0,A_0,B_0,A_0,B_0,A_0,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_0,B_cl_0,A_0,B_0,A_0,B_0,A_1,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_0,B_cl_0,A_0,B_0,A_1,B_0,A_0,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_0,B_cl_0,A_0,B_0,A_1,B_0,A_1,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_0,B_cl_0,A_1,B_0,A_0,B_0,A_0,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_0,B_cl_0,A_1,B_0,A_0,B_0,A_1,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_0,B_cl_0,A_1,B_0,A_1,B_0,A_0,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_0,B_cl_0,A_1,B_0,A_1,B_0,A_1,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_1,B_cl_0,A_0,B_0,A_0,B_0,A_0,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_1,B_cl_0,A_0,B_0,A_0,B_0,A_1,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_1,B_cl_0,A_0,B_0,A_1,B_0,A_0,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_1,B_cl_0,A_0,B_0,A_1,B_0,A_1,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_1,B_cl_0,A_1,B_0,A_0,B_0,A_0,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_1,B_cl_0,A_1,B_0,A_0,B_0,A_1,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_1,B_cl_0,A_1,B_0,A_1,B_0,A_0,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_1,B_cl_0,A_1,B_0,A_1,B_0,A_1,B_0) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
        ])/2^4

        J22 = sum([
            tr(kron(A_cl_0,B_cl_1,A_0,B_1,A_0,B_1,A_0,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            -tr(kron(A_cl_0,B_cl_1,A_0,B_1,A_0,B_1,A_1,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            -tr(kron(A_cl_0,B_cl_1,A_0,B_1,A_1,B_1,A_0,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_0,B_cl_1,A_0,B_1,A_1,B_1,A_1,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            -tr(kron(A_cl_0,B_cl_1,A_1,B_1,A_0,B_1,A_0,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_0,B_cl_1,A_1,B_1,A_0,B_1,A_1,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_0,B_cl_1,A_1,B_1,A_1,B_1,A_0,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            -tr(kron(A_cl_0,B_cl_1,A_1,B_1,A_1,B_1,A_1,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            -tr(kron(A_cl_1,B_cl_1,A_0,B_1,A_0,B_1,A_0,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_1,B_cl_1,A_0,B_1,A_0,B_1,A_1,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_1,B_cl_1,A_0,B_1,A_1,B_1,A_0,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            -tr(kron(A_cl_1,B_cl_1,A_0,B_1,A_1,B_1,A_1,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_1,B_cl_1,A_1,B_1,A_0,B_1,A_0,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            -tr(kron(A_cl_1,B_cl_1,A_1,B_1,A_0,B_1,A_1,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            -tr(kron(A_cl_1,B_cl_1,A_1,B_1,A_1,B_1,A_0,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
            tr(kron(A_cl_1,B_cl_1,A_1,B_1,A_1,B_1,A_1,B_1) * kron(ρ_00,ϕ_plus,ϕ_plus,ϕ_plus)),
        ])/2^4

        S_4star = I22^(1/4) + J22^(1/4)
        @test S_4star ≈ sqrt(2)^(3/4)
    end
end

@testset "Qubit Dephasing Factors as CHSH Product" for γ in 0:0.1:1

    ρ = State([1 0 0 sqrt(1-γ);0 0 0 0;0 0 0 0;sqrt(1-γ) 0 0 1]/2)

    A_deph_0 = σz
    A_deph_1 = σx

    B_deph_0 = Unitary((σz + sqrt(1-γ) * σx) / sqrt(1 + 1 - γ))
    B_deph_1 = Unitary((σz - sqrt(1-γ) * σx) / sqrt(1 + 1 - γ))

    @testset "bilocal network" for γ2 in 0:0.1:1
        ρ2 = State([1 0 0 sqrt(1-γ2);0 0 0 0;0 0 0 0;sqrt(1-γ2) 0 0 1]/2)
        B2_deph_0 = Unitary((σz + sqrt(1-γ2) * σx) / sqrt(1 + 1 - γ2))
        B2_deph_1 = Unitary((σz - sqrt(1-γ2) * σx) / sqrt(1 + 1 - γ2))

        I22 = sum([
            tr(kron(A_deph_0,B_deph_0,A_deph_0,B2_deph_0) * kron(ρ, ρ2)),
            tr(kron(A_deph_0,B_deph_0,A_deph_1,B2_deph_0) * kron(ρ, ρ2)),
            tr(kron(A_deph_1,B_deph_0,A_deph_0,B2_deph_0) * kron(ρ, ρ2)),
            tr(kron(A_deph_1,B_deph_0,A_deph_1,B2_deph_0) * kron(ρ, ρ2)),
        ])/2^2

        J22 = sum([
            tr(kron(A_deph_0,B_deph_1,A_deph_0,B2_deph_1) * kron(ρ, ρ2)),
            -tr(kron(A_deph_0,B_deph_1,A_deph_1,B2_deph_1) * kron(ρ, ρ2)),
            -tr(kron(A_deph_1,B_deph_1,A_deph_0,B2_deph_1) * kron(ρ, ρ2)),
            tr(kron(A_deph_1,B_deph_1,A_deph_1,B2_deph_1) * kron(ρ, ρ2)),
        ])/2^2

        S_2star = sqrt(I22) + sqrt(J22)
        lower_bound = sqrt(1 + ((1-γ)*(1-γ2))^(1/2))
        @test S_2star ≥ lower_bound || S_2star ≈ lower_bound
        @test S_2star ≈ (sqrt(1 + 1 - γ) * sqrt(1 + 1 - γ2))^(1/2)
    end

    @testset "trilocal star network" for γ2 in 0:0.2:1, γ3 in 0:0.2:1
        ρ2 = State([1 0 0 sqrt(1-γ2);0 0 0 0;0 0 0 0;sqrt(1-γ2) 0 0 1]/2)
        B2_deph_0 = Unitary((σz + sqrt(1-γ2) * σx) / sqrt(1 + 1 - γ2))
        B2_deph_1 = Unitary((σz - sqrt(1-γ2) * σx) / sqrt(1 + 1 - γ2))

        ρ3 = State([1 0 0 sqrt(1-γ3);0 0 0 0;0 0 0 0;sqrt(1-γ3) 0 0 1]/2)
        B3_deph_0 = Unitary((σz + sqrt(1-γ3) * σx) / sqrt(1 + 1 - γ3))
        B3_deph_1 = Unitary((σz - sqrt(1-γ3) * σx) / sqrt(1 + 1 - γ3))

        I22 = sum([
            tr(kron(A_deph_0,B_deph_0,A_deph_0,B2_deph_0,A_deph_0,B3_deph_0) * kron(ρ, ρ2, ρ3)),
            tr(kron(A_deph_0,B_deph_0,A_deph_0,B2_deph_0,A_deph_1,B3_deph_0) * kron(ρ, ρ2, ρ3)),
            tr(kron(A_deph_0,B_deph_0,A_deph_1,B2_deph_0,A_deph_0,B3_deph_0) * kron(ρ, ρ2, ρ3)),
            tr(kron(A_deph_0,B_deph_0,A_deph_1,B2_deph_0,A_deph_1,B3_deph_0) * kron(ρ, ρ2, ρ3)),
            tr(kron(A_deph_1,B_deph_0,A_deph_0,B2_deph_0,A_deph_0,B3_deph_0) * kron(ρ, ρ2, ρ3)),
            tr(kron(A_deph_1,B_deph_0,A_deph_0,B2_deph_0,A_deph_1,B3_deph_0) * kron(ρ, ρ2, ρ3)),
            tr(kron(A_deph_1,B_deph_0,A_deph_1,B2_deph_0,A_deph_0,B3_deph_0) * kron(ρ, ρ2, ρ3)),
            tr(kron(A_deph_1,B_deph_0,A_deph_1,B2_deph_0,A_deph_1,B3_deph_0) * kron(ρ, ρ2, ρ3)),
        ])/2^3

        J22 = sum([
            tr(kron(A_deph_0,B_deph_1,A_deph_0,B2_deph_1,A_deph_0,B3_deph_1) * kron(ρ, ρ2, ρ3)),
            -tr(kron(A_deph_0,B_deph_1,A_deph_0,B2_deph_1,A_deph_1,B3_deph_1) * kron(ρ, ρ2, ρ3)),
            -tr(kron(A_deph_0,B_deph_1,A_deph_1,B2_deph_1,A_deph_0,B3_deph_1) * kron(ρ, ρ2, ρ3)),
            tr(kron(A_deph_0,B_deph_1,A_deph_1,B2_deph_1,A_deph_1,B3_deph_1) * kron(ρ, ρ2, ρ3)),
            -tr(kron(A_deph_1,B_deph_1,A_deph_0,B2_deph_1,A_deph_0,B3_deph_1) * kron(ρ, ρ2, ρ3)),
            tr(kron(A_deph_1,B_deph_1,A_deph_0,B2_deph_1,A_deph_1,B3_deph_1) * kron(ρ, ρ2, ρ3)),
            tr(kron(A_deph_1,B_deph_1,A_deph_1,B2_deph_1,A_deph_0,B3_deph_1) * kron(ρ, ρ2, ρ3)),
            -tr(kron(A_deph_1,B_deph_1,A_deph_1,B2_deph_1,A_deph_1,B3_deph_1) * kron(ρ, ρ2, ρ3)),
        ])/2^3

        S_3star = I22^(1/3) + J22^(1/3)
        lower_bound = sqrt(1 + ((1-γ)*(1-γ2)*(1-γ3))^(1/3))
        @test S_3star ≥ lower_bound || S_3star ≈ lower_bound
        @test S_3star ≈ (sqrt(1 + 1 - γ) * sqrt(1 + 1 - γ2) * sqrt(1 + 1 - γ3) )^(1/3)
    end

    @testset "4-local star network" for γ2 in 0:0.25:1, γ3 in 0:0.25:1, γ4 in 0:0.25:1
        ρ2 = State([1 0 0 sqrt(1-γ2);0 0 0 0;0 0 0 0;sqrt(1-γ2) 0 0 1]/2)
        B2_deph_0 = Unitary((σz + sqrt(1-γ2) * σx) / sqrt(1 + 1 - γ2))
        B2_deph_1 = Unitary((σz - sqrt(1-γ2) * σx) / sqrt(1 + 1 - γ2))

        ρ3 = State([1 0 0 sqrt(1-γ3);0 0 0 0;0 0 0 0;sqrt(1-γ3) 0 0 1]/2)
        B3_deph_0 = Unitary((σz + sqrt(1-γ3) * σx) / sqrt(1 + 1 - γ3))
        B3_deph_1 = Unitary((σz - sqrt(1-γ3) * σx) / sqrt(1 + 1 - γ3))

        ρ4 = State([1 0 0 sqrt(1-γ4);0 0 0 0;0 0 0 0;sqrt(1-γ4) 0 0 1]/2)
        B4_deph_0 = Unitary((σz + sqrt(1-γ4) * σx) / sqrt(1 + 1 - γ4))
        B4_deph_1 = Unitary((σz - sqrt(1-γ4) * σx) / sqrt(1 + 1 - γ4))

        I22 = sum([
            tr(kron(A_deph_0,B_deph_0,A_deph_0,B2_deph_0,A_deph_0,B3_deph_0,A_deph_0,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_0,B_deph_0,A_deph_0,B2_deph_0,A_deph_0,B3_deph_0,A_deph_1,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_0,B_deph_0,A_deph_0,B2_deph_0,A_deph_1,B3_deph_0,A_deph_0,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_0,B_deph_0,A_deph_0,B2_deph_0,A_deph_1,B3_deph_0,A_deph_1,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_0,B_deph_0,A_deph_1,B2_deph_0,A_deph_0,B3_deph_0,A_deph_0,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_0,B_deph_0,A_deph_1,B2_deph_0,A_deph_0,B3_deph_0,A_deph_1,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_0,B_deph_0,A_deph_1,B2_deph_0,A_deph_1,B3_deph_0,A_deph_0,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_0,B_deph_0,A_deph_1,B2_deph_0,A_deph_1,B3_deph_0,A_deph_1,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_1,B_deph_0,A_deph_0,B2_deph_0,A_deph_0,B3_deph_0,A_deph_0,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_1,B_deph_0,A_deph_0,B2_deph_0,A_deph_0,B3_deph_0,A_deph_1,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_1,B_deph_0,A_deph_0,B2_deph_0,A_deph_1,B3_deph_0,A_deph_0,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_1,B_deph_0,A_deph_0,B2_deph_0,A_deph_1,B3_deph_0,A_deph_1,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_1,B_deph_0,A_deph_1,B2_deph_0,A_deph_0,B3_deph_0,A_deph_0,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_1,B_deph_0,A_deph_1,B2_deph_0,A_deph_0,B3_deph_0,A_deph_1,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_1,B_deph_0,A_deph_1,B2_deph_0,A_deph_1,B3_deph_0,A_deph_0,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_1,B_deph_0,A_deph_1,B2_deph_0,A_deph_1,B3_deph_0,A_deph_1,B4_deph_0) * kron(ρ,ρ2,ρ3,ρ4)),
        ])/2^4

        J22 = sum([
            tr(kron(A_deph_0,B_deph_1,A_deph_0,B2_deph_1,A_deph_0,B3_deph_1,A_deph_0,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
            -tr(kron(A_deph_0,B_deph_1,A_deph_0,B2_deph_1,A_deph_0,B3_deph_1,A_deph_1,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
            -tr(kron(A_deph_0,B_deph_1,A_deph_0,B2_deph_1,A_deph_1,B3_deph_1,A_deph_0,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_0,B_deph_1,A_deph_0,B2_deph_1,A_deph_1,B3_deph_1,A_deph_1,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
            -tr(kron(A_deph_0,B_deph_1,A_deph_1,B2_deph_1,A_deph_0,B3_deph_1,A_deph_0,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_0,B_deph_1,A_deph_1,B2_deph_1,A_deph_0,B3_deph_1,A_deph_1,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_0,B_deph_1,A_deph_1,B2_deph_1,A_deph_1,B3_deph_1,A_deph_0,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
            -tr(kron(A_deph_0,B_deph_1,A_deph_1,B2_deph_1,A_deph_1,B3_deph_1,A_deph_1,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
            -tr(kron(A_deph_1,B_deph_1,A_deph_0,B2_deph_1,A_deph_0,B3_deph_1,A_deph_0,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_1,B_deph_1,A_deph_0,B2_deph_1,A_deph_0,B3_deph_1,A_deph_1,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_1,B_deph_1,A_deph_0,B2_deph_1,A_deph_1,B3_deph_1,A_deph_0,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
            -tr(kron(A_deph_1,B_deph_1,A_deph_0,B2_deph_1,A_deph_1,B3_deph_1,A_deph_1,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_1,B_deph_1,A_deph_1,B2_deph_1,A_deph_0,B3_deph_1,A_deph_0,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
            -tr(kron(A_deph_1,B_deph_1,A_deph_1,B2_deph_1,A_deph_0,B3_deph_1,A_deph_1,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
            -tr(kron(A_deph_1,B_deph_1,A_deph_1,B2_deph_1,A_deph_1,B3_deph_1,A_deph_0,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
            tr(kron(A_deph_1,B_deph_1,A_deph_1,B2_deph_1,A_deph_1,B3_deph_1,A_deph_1,B4_deph_1) * kron(ρ,ρ2,ρ3,ρ4)),
        ])/2^4

        S_4star = I22^(1/4) + J22^(1/4)

        lower_bound = sqrt(1 + ((1-γ)*(1-γ2)*(1-γ3)*(1-γ4))^(1/4))
        @test S_4star ≥ lower_bound || S_4star ≈ lower_bound
        @test S_4star ≈ (sqrt(1+1-γ)*sqrt(1+1-γ2)*sqrt(1+1-γ3)*sqrt(1+1-γ4))^(1/4)
    end
end