using ADI
using Statistics
using WoodburyMatrices
using LinearAlgebra
using HCIToolbox
using Interpolations

import Distributions
import Distributions: ContinuousMatrixDistribution

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
    JointModel(A, w, cube, psf)

Construct a joint model using the given basis vectors `A` and weights `w`. They should have shape `(n, Npx)` and `(n, M)`, respectively. `cube` will be used as the target for the likelihood.

# Extended help

The mathematical formulation of this model is
```math
\\text{cube} \\sim \\mathbf{A} \\cdot \\mathbf{w} + \\text{PSF} + \\epsilon
```
"""
function JointModel(A, w, cube, angles, psf)
    ref_cov = fake_covariance(flatten(cube))
    w̄ = mean(w, dims=2)
    Λ = cov(w, dims=2)
    Σ = SymWoodbury(ref_cov, A', Λ)
    Σ⁻¹AΛAᵀ = Σ \ (A' * (Λ * A))
    Aw̄ = repeat(w̄' * A |> expand, length(angles), 1, 1)
    target = flatten(cube)
    return JointModel(cube, target, angles, Σ, Σ⁻¹AΛAᵀ, Aw̄, psf)
end

"""
    (model::JoinModel)(base=model.Aw̄; r, theta, A=1)
    (model::JoinModel)(base=model.Aw̄; x, y, A=1)

Return the signal, `μ(params)` injected into `base`.
"""
function (model::JointModel)(base::AbstractArray{T}=model.Aw̄; degree=Linear(), params...) where T
    S = mapreduce(typeof, promote_type, values(params))
    V = promote_type(T, S)
    μ = inject!(V.(base), model.psf, model.angles; params...)
    return JointDistribution(flatten(μ), model.Σ)
end

"""
    ADI.reconstruct(::JointModel, [data]; r, theta, A=1)
    ADI.reconstruct(::JointModel, [data]; x, y, A=1)

Reconstruct the systematics conditioned on the input `data`. If not provided, will use the internal `target`. When `data` is a matrix, the returned reconstruction will also be a matrix. When `data` is a cube, the reconstruction will be expanded into a cube.
"""
function ADI.reconstruct(model::JointModel, cube = model.cube; params...)
    base = reshape(model.Aw̄[1, :, :], 1, :)
    μ = model(;params...).μ
    return base .+ (flatten(cube) .- μ) * model.Σ⁻¹AΛAᵀ |> expand
end

"""
    Distributions.loglikelihood(::JointModel, [data]; r, theta, A=1)
    Distributions.loglikelihood(::JointModel, [data]; x, y, A=1)

Return the likelihood of the model target given the input parameters. By default uses the model's cube as the input, but a matrix can be passed directly. Returns the logkernel of a matrix normal distribution without column-wise variance. 
"""
function loglikelihood(model::JointModel, data::AbstractMatrix=model.target; params...)
    μ = model(;params...).μ# |> flatten
    R = data .- μ
    return -tr(R * (model.Σ \ R')) / 2
end

# TODO
# probably use Distributions.logkernel and Distributions.loglikelihood to further
# allow this to become a "distribution" for use in other inference methods.
struct JointDistribution{T,M<:AbstractMatrix{T},W<:AbstractWoodbury} <: ContinuousMatrixDistribution
    μ::M
    Σ::W
end

function Distributions._logpdf(d::JointDistribution, X::AbstractMatrix)
    R = X .- d.μ
    return -tr(R * (d.Σ \ R')) / 2
end

Base.size(d::JointDistribution) = size(d.μ)

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

