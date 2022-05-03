using QBase
using LinearAlgebra
using Test

@testset "optimal classical strategy on single arm point" begin
    σ_plus = Unitary((σz + σx)/sqrt(2))
    σ_minus = Unitary((σz - σx)/sqrt(2))

    @testset "bilocal network" begin
        


    end

    @testset "trilocal star network" begin
        A1_0 = σz
        A1_1 = σx

        A2_0 = σ_plus
        A2_1 = σ_minus

        A3_0 = σ_plus
        A3_1 = σ_minus

        B1_0 = σz
        B1_1 = σz

        B2_0 = σz
        B2_1 = σx

        B3_0 = σz
        B3_1 = σx

        ψ = State([1 0 0 0;0 0 0 0;0 0 0 0;0 0 0 0])
        ϕ = State([1 0 0 1;0 0 0 0;0 0 0 0;1 0 0 1]/2)



        I22 = sum([
            tr(kron(A1_0,B1_0,A2_0,B2_0,A3_0,B3_0) * kron(ψ, ϕ, ϕ)),
            tr(kron(A1_0,B1_0,A2_0,B2_0,A3_1,B3_0) * kron(ψ, ϕ, ϕ)),
            tr(kron(A1_0,B1_0,A2_1,B2_0,A3_0,B3_0) * kron(ψ, ϕ, ϕ)),
            tr(kron(A1_0,B1_0,A2_1,B2_0,A3_1,B3_0) * kron(ψ, ϕ, ϕ)),
            tr(kron(A1_1,B1_0,A2_0,B2_0,A3_0,B3_0) * kron(ψ, ϕ, ϕ)),
            tr(kron(A1_1,B1_0,A2_0,B2_0,A3_1,B3_0) * kron(ψ, ϕ, ϕ)),
            tr(kron(A1_1,B1_0,A2_1,B2_0,A3_0,B3_0) * kron(ψ, ϕ, ϕ)),
            tr(kron(A1_1,B1_0,A2_1,B2_0,A3_1,B3_0) * kron(ψ, ϕ, ϕ)),
        ])/2^3

        J22 = sum([
            tr(kron(A1_0,B1_1,A2_0,B2_1,A3_0,B3_1) * kron(ψ, ϕ, ϕ)),
            -tr(kron(A1_0,B1_1,A2_0,B2_1,A3_1,B3_1) * kron(ψ, ϕ, ϕ)),
            -tr(kron(A1_0,B1_1,A2_1,B2_1,A3_0,B3_1) * kron(ψ, ϕ, ϕ)),
            tr(kron(A1_0,B1_1,A2_1,B2_1,A3_1,B3_1) * kron(ψ, ϕ, ϕ)),
            -tr(kron(A1_1,B1_1,A2_0,B2_1,A3_0,B3_1) * kron(ψ, ϕ, ϕ)),
            tr(kron(A1_1,B1_1,A2_0,B2_1,A3_1,B3_1) * kron(ψ, ϕ, ϕ)),
            tr(kron(A1_1,B1_1,A2_1,B2_1,A3_0,B3_1) * kron(ψ, ϕ, ϕ)),
            -tr(kron(A1_1,B1_1,A2_1,B2_1,A3_1,B3_1) * kron(ψ, ϕ, ϕ)),
        ])/2^3

        S_3star = I22^(1/3) + J22^(1/3)
        @test S_3star ≈ sqrt(2)^(2/3)
    end


end
