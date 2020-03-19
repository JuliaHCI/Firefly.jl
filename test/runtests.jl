using Firefly
using Test

@testset "Firefly.jl" begin
    @testset "Distributions" begin
        include("distributions.jl")
    end
end
