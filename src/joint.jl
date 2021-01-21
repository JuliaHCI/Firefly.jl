using ADI
using CoordinateTransformations
using Statistics
using WoodburyMatrices
using LinearAlgebra
using HCIToolbox
using Interpolations
using PSFModels: PSFModel
using StaticArrays

using ImageTransformations: box_extrapolation, center
using Interpolations: AbstractExtrapolation

import Distributions
import Distributions: ContinuousMatrixDistribution

struct JointModel{T <: AbstractArray, M <: AbstractMatrix, V <: AbstractVector, W <: AbstractWoodbury,WT,G<:CubeGenerator}
    cube::T
    target::M
    angles::V
    Σ::W
    Σ⁻¹AΛAᵀ::M
    w̄A::WT
    gen::G
    tmpR::M
end

"""
    JointModel(A, w, cube, psf)

Construct a joint model using the given basis vectors `A` and weights `w`. They should have shape `(n, Npx)` and `(n, M)`, respectively. `cube` will be used as the target for the likelihood.

# Extended help

The mathematical formulation of this model is
```math
\\text{cube} \\sim \\mathbf{A} \\cdot \\mathbf{w} + \\text{PSF} + \\epsilon
```
"""
function JointModel(A, w, cube::AbstractArray{T,3}, angles, psf; kwargs...) where T
    w̄ = mean(w, dims=1)
    w̄A = w̄ * A
    X = flatten(cube)
    resid = X - w * A
    Λ = cov(w, dims=1)
    C = fake_covariance(resid)
    Σ = SymWoodbury(C, A', Λ)
    Σ⁻¹AΛAᵀ = Σ \ (A' * (Λ * A))
    target = X .- w̄A
    gen = CubeGenerator(cube, angles, psf; kwargs...)
    return JointModel(cube, target, angles, Σ, Σ⁻¹AΛAᵀ, w̄A, gen, similar(target))
end

function JointModel(A, w, cube::AnnulusView{T,3}, angles, psf; kwargs...) where T
    w̄ = mean(w, dims=1)
    w̄A = w̄ * A
    X = cube()
    resid = X - w * A
    Λ = cov(w, dims=1)
    C = fake_covariance(resid)
    Σ = SymWoodbury(C, A', Λ)
    Σ⁻¹AΛAᵀ = Σ \ (A' * (Λ * A))
    target = X .- w̄A
    gen = CubeGenerator(cube, angles, psf; kwargs...)
    return JointModel(cube, target, angles, Σ, Σ⁻¹AΛAᵀ, w̄A, gen, similar(target))
end


"""
    Distributions.loglikelihood(::JointModel, [data]; r, theta, A=1)
    Distributions.loglikelihood(::JointModel, [data]; x, y, A=1)

Return the likelihood of the model target given the input parameters. By default uses the model's cube as the input, but a matrix can be passed directly. Returns the logkernel of a matrix normal distribution without column-wise variance.
"""
function Distributions.loglikelihood(model::JointModel, pos; A=zero(eltype(model.cube)))
    T = float(typeof(A))
    base = T.(model.target)
    R = model.gen(base, pos; A=-A)
    return -tr(R * (model.Σ \ R')) / 2
end

function ADI.reconstruct(model::JointModel, pos; A=zero(eltype(model.cube)))
    T = float(typeof(A))
    base = T.(model.target)
    R = model.gen(base, pos; A=-A)
    return expand(model.w̄A .+ R * model.Σ⁻¹AΛAᵀ)
end

function ADI.reconstruct(model::JointModel{<:AnnulusView}, pos; A=zero(eltype(model.cube)))
    T = float(typeof(A))
    base = T.(model.target)
    R = model.gen(base, pos; A=-A)
    return inverse(model.cube, model.w̄A .+ R * model.Σ⁻¹AΛAᵀ)
end

function model(model::JointModel, pos; A=zero(eltype(model.cube)))
    T = float(typeof(A))
    return model.gen(T, pos; A)
end

function Statistics.mean(m::JointModel, args...; params...)
    plan = model(m, args...; params...)
    L = reconstruct(m, args...; params...)
    return expand(plan .+ L)
end

function Statistics.mean(m::JointModel{<:AnnulusView}, args...; params...)
    plan = model(m, args...; params...)
    L = reconstruct(m, args...; params...)
    return inverse(m.cube, plan .+ L)
end

# TODO
# probably use Distributions.logkernel and Distributions.loglikelihood to further
# allow this to become a "distribution" for use in other inference methods.
struct JointDistribution{T,M<:AbstractMatrix{T},W<:AbstractWoodbury} <: ContinuousMatrixDistribution
    μ::M
    Σ::W
end

function Distributions.loglikelihood(d::JointDistribution, X::AbstractMatrix)
    R = X - d.μ
    return -tr(R * (d.Σ \ R')) / 2
end

Base.size(d::JointDistribution) = size(d.μ)

# covariance is just a diagonal of the variance
function fake_covariance(mat::AbstractMatrix)
    D = var(mat, dims = 1) |> vec
    @. D[iszero(D)] = Inf
    return Diagonal(D)
end
