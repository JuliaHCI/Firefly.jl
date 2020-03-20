module Firefly

export findpeak,
       findpeaks

using KernelDensity: kde
using MCMCChains: Chains
using StatsBase: rle


include("distributions.jl")

###############################################################################

"""
    findpeak(::AbstractVector)
    findpeak(::Chains, [::Symbol])

Return a naïve maximum a posteriori (MAP) estimate from a chain.

This uses kernel density estimation (KDE) to estimate the continuous posterior distribution from the input chain and simply returns its max argument. This is naïve because we should not consider KDE peaks to be true "modes" or MAP estimates. Estimating accurate modes of a posterior sample is still an active area of statistical research.

# Examples
```jldoctest
julia> samples = randn(1000);

julia> findpeak(samples)
-0.15763422104075167
```
"""
function findpeak(samples::AbstractVector)
    k = kde(samples)
    return k.x[argmax(k.density)]
end

findpeak(chain::Chains) = findpeak(Array(chain))
findpeak(chain::Chains, s::Symbol) = findpeak(chain[s])

"""
    findpeaks(::AbstractVector, [n])
    findpeaks(::Chains, [::Symbol], [n])

Return the first `n` naïve maximum a posteriori (MAP) estimates from a chain.

This uses kernel density estimation (KDE) to estimate the continuous posterior distribution from the input sample and finds maxima via the finite-difference estimation of the derivative. This is naïve because we should not consider KDE peaks to be true "modes" or MAP estimates. Estimating accurate modes of a posterior sample is still an active area of statistical research.

# Examples
```jldoctest
julia> samples = randn(1000) .+ (randn(1000) .+ 10);

julia> findpeaks(samples, 2)
2-element Array{Float64,1}:
  9.82386893916195 
 14.595394948392164
```
"""
function findpeaks(samples::AbstractVector)
    k = kde(samples)
    # the sign of the difference will tell use whether we are increasing or decreasing
    # using rle gives us the points at which the sign switches (local extreema)
    runs = rle(sign.(diff(k.density)))
    # if we start going up, first extreme will be maximum, else minimum
    start = runs[1][1] == 1 ? 1 : 2
    # find the peak indices at the local minima
    peak_idx = cumsum(runs[2])[start:2:end]
    sorted_idx = sortperm(k.density[peak_idx], rev = true)
    return k.x[peak_idx[sorted_idx]]
end

findpeaks(samples::AbstractVector, n::Integer) = findpeaks(samples)[1:n]
findpeaks(chain::Chains) = findpeaks(Array(chain))
findpeaks(chain::Chains, n::Integer) = findpeaks(Array(chain), n)
findpeaks(chain::Chains, s::Symbol) = findpeaks(chain[s])
findpeaks(chain::Chains, s::Symbol, n::Integer) = findpeaks(chain[s], n)

end
