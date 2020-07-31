using ADI
using Statistics
using WoodburyMatrices
using LinearAlgebra
using HCIToolbox
using Interpolations

import Distributions: loglikelihood, ContinuousMultivariateDistribution

struct JointModel{T <: AbstractArray, M <: AbstractMatrix, V <: AbstractVector, W <: AbstractWoodbury,P<:Union{HCIToolbox.PSFKernel, M}}
    cube::T
    target::M
    angles::V
    Σ::W
    Σ⁻¹AΛAᵀ::M
    Aw̄::T
    psf::P
end

"""
    JointModel(::ADI.ADIDesign, cube, psf)

Creates a joint model using the given design targeting the given cube, with the given PSF model.

# Extended Help
"""
function JointModel(design::ADI.ADIDesign, cube, psf)
    # design matrix (transpose for shape considerations)
    A = ADI.design(design)
    ref_cov = fake_covariance(flatten(cube))
    w = ADI.weights(design)
    w̄ = mean(w, dims=2)
    Λ = cov(w, dims=2)
    Σ = SymWoodbury(ref_cov, A, Λ)
    Σ⁻¹AΛAᵀ = Σ \ (A * (Λ * A'))
    Aw̄ = repeat(w̄' * A' |> expand, length(design.angles), 1, 1)
    target = flatten(cube)
    return JointModel(cube, target, design.angles, Σ, Σ⁻¹AΛAᵀ, Aw̄, psf)
end

"""
    (model::JoinModel)(base=model.Aw̄; r, theta, A=1)
    (model::JoinModel)(base=model.Aw̄; x, y, A=1)

Return the signal, `μ(params)` injected into `base`.
"""
function (model::JointModel)(base::AbstractArray{T}=model.Aw̄; s=1, degree=Linear(), params...) where T
    S = map(typeof, values(params))
    V = promote_type(T, S...)
    μ = inject!(V.(base), model.psf, model.angles; params...)
    return JointDistribution(flatten(μ), model.Σ, V(s))
end

"""
    ADI.reconstruct(::JointModel, [data]; r, theta, A=1)
    ADI.reconstruct(::JointModel, [data]; x, y, A=1)

Reconstruct the systematics conditioned on the input `data`. If not provided, will use the internal `target`. When `data` is a matrix, the returned reconstruction will also be a matrix. When `data` is a cube, the reconstruction will be expanded into a cube.
"""
function ADI.reconstruct(model::JointModel, data::AbstractMatrix; s=1, params...)
    base = reshape(model.Aw̄[1, :, :], 1, :)
    μ = model(;params...).μ
    return base .+ (data .- μ) * model.Σ⁻¹AΛAᵀ ./ s
end
ADI.reconstruct(model::JointModel; params...) = 
    reconstruct(model, model.cube; params...)
ADI.reconstruct(model::JointModel, data::AbstractArray{T,3}; params...) where T = 
    reconstruct(model, flatten(data); params...) |> expand

"""
    Distributions.loglikelihood(::JointModel, [data]; r, theta, A=1)
    Distributions.loglikelihood(::JointModel, [data]; x, y, A=1)

Return the likelihood of the model target given the input parameters. By default uses the model's cube as the input, but a matrix can be passed directly. Returns the logkernel of a matrix normal distribution without column-wise variance. 
"""
function loglikelihood(model::JointModel, data::AbstractMatrix=model.target; s=1, params...)
    μ = model(;params...).μ# |> flatten
    R = data .- μ
    n = size(R, 1)
    return -(n * s + tr(R * (model.Σ \ R')) / s) / 2
    # return -tr(R * (model.Σ⁻¹ * R')) / 2
end

# TODO
# probably use Distributions.logkernel and Distributions.loglikelihood to further
# allow this to become a "distribution" for use in other inference methods.
struct JointDistribution{T,M<:AbstractMatrix{T},W<:AbstractWoodbury{T}} <: ContinuousMultivariateDistribution
    μ::M
    Σ::W
    s::T
end

JointDistribution(μ::M, Σ::W, s) where {T,M<:AbstractMatrix,W<:AbstractWoodbury{T}} =
    JointDistribution(T.(μ), Σ, T(s))

function loglikelihood(d::JointDistribution, X::AbstractMatrix)
    R = X .- d.μ
    n = size(R, 1)
    return -(n * d.s + tr(R * (d.Σ \ R')) / d.s) / 2
end

Base.length(d::JointDistribution) = length(d.μ)

# covariance is just a diagonal of the variance
function fake_covariance(mat::AbstractMatrix)
    D = var(mat, dims = 1)[1, :]
    @. D[iszero(D)] = Inf
    return Diagonal(D)
end

function prepare_covariance(mat, fwhm)
    bad_diags = diagind(mat)[diag(mat) .== 0]
    mat[bad_diags] .= Inf
    # for j in axes(mat, 2), i in j + 1:size(mat, 1)
    #     if i - j > 2fwhm
    #         mat[i, j] = 0
    #         mat[j, i] = 0
    #     end
    # end
    # return factorize(mat)
    return Tridiagonal(mat)
end

