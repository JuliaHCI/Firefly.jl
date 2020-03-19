import Distributions: ContinuousUnivariateDistribution,
                      cdf,
                      logcdf,
                      pdf,
                      logpdf,
                      insupport

import Statistics: quantile

export RadialUniform,
       PoissonInvariant


"""
    RadialUniform(r_in, r_out) <: ContinuousUnivariateDistribution

A radially uniform distribution from `r_in` to `r_out`.

This distribution is derived from the change of variables evaluation of 

``p(r, \\theta) = p(x, y)\\left|\\nabla_{r,\\theta}(x, y) \\right|``

which given ``p(x, y)\\propto 1``, leads to ``p(r, \\theta)\\propto r``

# Form

``p(r\\in (r_\\text{in}, r_\\text{out})) = \\frac{2r}{r_\\text{out}^2 - r_\\text{in}^2}``

# Supported Functions

These functions have been explicitly written for `RadialUniform` from [Distributions.jl](https://github.com/juliastats/Distributions.jl). There may be more functionality available from fallback methods, but the following are guaranteed to work.
* [`pdf`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.pdf-Tuple{Distribution{Univariate,S}%20where%20S%3C:ValueSupport,Real})
* [`logpdf`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.logpdf-Tuple{Distribution{Univariate,S}%20where%20S%3C:ValueSupport,Real})
* [`cdf`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.cdf-Tuple{Distribution{Univariate,S}%20where%20S%3C:ValueSupport,Real})
* [`logcdf`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.logcdf-Tuple{Distribution{Univariate,S}%20where%20S%3C:ValueSupport,Real})
* [`quantile`](https://juliastats.org/Distributions.jl/stable/univariate/#Statistics.quantile-Tuple{Distribution{Univariate,S}%20where%20S%3C:ValueSupport,Real})
* [`minimum`](https://juliastats.org/Distributions.jl/stable/univariate/#Base.minimum-Tuple{Distribution{Univariate,S}%20where%20S%3C:ValueSupport})
* [`maximum`](https://juliastats.org/Distributions.jl/stable/univariate/#Base.maximum-Tuple{Distribution{Univariate,S}%20where%20S%3C:ValueSupport})

# Examples
```jldoctest
julia> using Distributions

julia> dist = RadialUniform(0, 10)
RadialUniform{Int64}(r_in=0, r_out=10)

julia> pdf(dist, -1)
0

julia> pdf(dist, 3)
0.06

julia> cdf(dist, quantile(dist, 0.5))
0.5000000000000001
```
"""
struct RadialUniform{T <: AbstractFloat} <: ContinuousUnivariateDistribution
    r_in::T
    r_out::T
    function RadialUniform(r_in::S, r_out::S) where S 
        T = float(S)
        new{T}(T(r_in), T(r_out))
    end
end

RadialUniform(r_in, r_out) = RadialUniform(promote(r_in, r_out)...)

Base.minimum(d::RadialUniform) = d.r_in
Base.maximum(d::RadialUniform) = d.r_out

quantile(d::RadialUniform, q::Real) = sqrt((d.r_out^2 - d.r_in^2) * q + d.r_in^2)

function cdf(d::RadialUniform{T}, x::Real) where {T}
    x < d.r_in && return zero(T) 
    x > d.r_out && return one(T)
    return (x^2 - d.r_in^2) / (d.r_out^2 - d.r_in^2)    
end

function logcdf(d::RadialUniform{T}, x::Real) where {T}
    x < d.r_in && return -T(Inf)
    x > d.r_out && return zero(T)
    return log(x^2 - d.r_in^2) - log(d.r_out^2 - d.r_in^2)
end

pdf(d::RadialUniform{T}, x::Real) where T = insupport(d, x) ? 2 * x / (d.r_out^2 - d.r_in^2) : zero(T)
logpdf(d::RadialUniform{T}, x::Real) where T = insupport(d, x) ? log(2x) - log(d.r_out^2 - d.r_in^2) : -T(Inf)



"""
    PoissonInvariant(a, b) <: ContinuousUnivariateDistribution

A Poisson invariant distributions truncated from `a` to `b`.

This distribution is derived from the determinant of the Fisher information matrix

``p(x\\in(a, b)) \\propto \\sqrt{E\\left[\\left(\\frac{d\\ln{L}}{dx} \\right)^2 \\right]}``

which gives ``p(x\\in(a, b)) \\propto 1/\\sqrt{x}``

# Form

``p(x\\in(a, b)) = \\frac{1}{2(\\sqrt{b} - \\sqrt{a}\\sqrt{x}}``


# Supported Functions

These functions have been explicitly written for `RadialUniform` from [Distributions.jl](https://github.com/juliastats/Distributions.jl). There may be more functionality available from fallback methods, but the following are guaranteed to work.
* [`pdf`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.pdf-Tuple{Distribution{Univariate,S}%20where%20S%3C:ValueSupport,Real})
* [`logpdf`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.logpdf-Tuple{Distribution{Univariate,S}%20where%20S%3C:ValueSupport,Real})
* [`cdf`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.cdf-Tuple{Distribution{Univariate,S}%20where%20S%3C:ValueSupport,Real})
* [`logcdf`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.logcdf-Tuple{Distribution{Univariate,S}%20where%20S%3C:ValueSupport,Real})
* [`quantile`](https://juliastats.org/Distributions.jl/stable/univariate/#Statistics.quantile-Tuple{Distribution{Univariate,S}%20where%20S%3C:ValueSupport,Real})
* [`minimum`](https://juliastats.org/Distributions.jl/stable/univariate/#Base.minimum-Tuple{Distribution{Univariate,S}%20where%20S%3C:ValueSupport})
* [`maximum`](https://juliastats.org/Distributions.jl/stable/univariate/#Base.maximum-Tuple{Distribution{Univariate,S}%20where%20S%3C:ValueSupport})

# Examples
```jldoctest
julia> using Distributions

julia> dist = PoissonInvariant(0, 1)
PoissonInvariant{Float64}(a=0.0, b=1.0)

julia> pdf(dist, -1)
0.0

julia> pdf(dist, 3)
0.0

julia> cdf(dist, quantile(dist, 0.5))
0.5
```
"""
struct PoissonInvariant{T <: AbstractFloat} <: ContinuousUnivariateDistribution
    a::T
    b::T
    function PoissonInvariant(a::S, b::S) where S
        T = float(S)
        new{T}(T(a), T(b))
    end
end

PoissonInvariant(a, b) = PoissonInvariant(promote(a, b)...)

Base.minimum(d::PoissonInvariant) = d.a
Base.maximum(d::PoissonInvariant) = d.b

quantile(d::PoissonInvariant, q::Real) = (q * (sqrt(d.b) - sqrt(d.a)) + sqrt(d.a))^2

function cdf(d::PoissonInvariant{T}, x::Real) where T
    x < d.a && return zero(T)
    x > d.b && return one(T)
    return (sqrt(x) - sqrt(d.a)) / (sqrt(d.b) - sqrt(d.a))
end

function logcdf(d::PoissonInvariant{T}, x::Real) where T
    x < d.a && return zero(T)
    x > d.b && return one(T)
    return log(sqrt(x) - sqrt(d.a)) - log(sqrt(d.b) - sqrt(d.a))
end

pdf(d::PoissonInvariant{T}, x::Real) where T = insupport(d, x) ? 1 / (2sqrt(x) * (sqrt(d.b) - sqrt(d.a))) : zero(T)
logpdf(d::PoissonInvariant{T}, x::Real) where T = insupport(d, x) ? -log(2sqrt(x) * (sqrt(d.b) - sqrt(d.a))) : -T(Inf)
