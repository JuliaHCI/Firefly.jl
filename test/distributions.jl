using Distributions
using Statistics

@testset "Radial Uniform" begin
    dist = RadialUniform(0, 1)

    @testset "Interface" begin
        @test minimum(dist) == 0
        @test maximum(dist) == 1
    end

    @testset "Accuracy" begin
        @test pdf(dist, -1) ≈ 0
        @test pdf(dist, 0.5) ≈ 1
        @test pdf(dist, 2) ≈ 0


        @test logpdf(dist, -1) ≈ -Inf
        @test logpdf(dist, 0.5) ≈ 0
        @test logpdf(dist, 2) ≈ -Inf

        @test cdf(dist, -1) ≈ 0
        @test cdf(dist, 0.5) ≈ 1 / 4
        @test cdf(dist, 2) ≈ 1
        
        @test logcdf(dist, -1) ≈ -Inf
        @test logcdf(dist, 0.5) ≈ -log(4)
        @test logcdf(dist, 2) ≈ 0

        @test quantile(dist, 1) ≈ 1
        @test quantile(dist, 0) ≈ 0

        for i in range(0, 1, length = 10)
            @test cdf(dist, quantile(dist, i)) ≈ i
        end
    end
end

@testset "Poisson Invariant" begin
    dist = PoissonInvariant(0, 1)

    @testset "Interface" begin
        @test minimum(dist) == 0
        @test maximum(dist) == 1
    end

    @testset "Accuracy" begin
        @test pdf(dist, -1) ≈ 0
        @test pdf(dist, 0.5) ≈ 1 / 2 / √0.5
        @test pdf(dist, 2) ≈ 0


        @test logpdf(dist, -1) ≈ -Inf
        @test logpdf(dist, 0.5) ≈ -log(2 * √0.5)
        @test logpdf(dist, 2) ≈ -Inf

        @test cdf(dist, -1) ≈ 0
        @test cdf(dist, 0.5) ≈ √0.5
        @test cdf(dist, 2) ≈ 1
        
        @test logcdf(dist, -1) ≈ -Inf
        @test logcdf(dist, 0.5) ≈ -log(2) / 2
        @test logcdf(dist, 2) ≈ 0

        @test quantile(dist, 1) ≈ 1
        @test quantile(dist, 0) ≈ 0
        
        for i in range(0, 1, length = 10)
            @test cdf(dist, quantile(dist, i)) ≈ i
        end
    end
end
