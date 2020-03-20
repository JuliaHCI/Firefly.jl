using Firefly
using Test
using MCMCChains

@testset "Firefly.jl" begin
    @testset "Distributions" begin
        include("distributions.jl")
    end
    @testset "findpeak" begin
        X = ones(100)
        @test findpeak(X) ≈ 1 rtol = 1e-2
        @test findpeaks(X, 1) ≈ [1] rtol = 1e-2

        X = append!(ones(100), 2 .* ones(80))
        @test findpeak(X) ≈ 1 rtol = 1e-2
        @test findpeaks(X, 1) ≈ [1] rtol = 1e-2
        @test findpeaks(X, 2) ≈ [1, 2] rtol = 1e-2

        # interface
        c = Chains(reshape(X, (180, 1, 1)), ["x"])
        @test findpeak(c) == findpeak(X)
        @test findpeaks(c) == findpeaks(X)
        @test findpeak(c) == findpeak(c, :x)
        @test findpeaks(c) == findpeaks(c, :x)
        @test findpeaks(c, 1) == findpeaks(c, :x, 1)
        @test findpeaks(c, 1) == findpeaks(c, :x)[1:1]

        c2 = Chains(ones(100, 2, 1), ["x", "y"])
        @test_throws MethodError findpeak(c2)
        @test_throws MethodError findpeaks(c2)
        @test_throws MethodError findpeaks(ones(100, 100))
        @test_throws ArgumentError findpeaks(ones(100), 2.4)
    end
end
