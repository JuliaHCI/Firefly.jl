using ADI
using Statistics
using WoodburyMatrices
using LinearAlgebra
using HCIToolbox

import Distributions: loglikelihood

struct JointModel{T <: AbstractArray,M <: AbstractMatrix,V <: AbstractVector,W <: AbstractWoodbury}
    cube::T
    cube_mean::T
    target::M
    angles::V
    Σ::W
    psf::M
end

function JointModel(design::ADI.ADIDesign, cube, psf)
    # design matrix (transpose for shape considerations)
    A = design.A'
    ref_cov = fake_covariance(flatten(design.S))
    w = design.w
    # correction_factor = expand(mean(w, dims=1) * design.A)
    Λ = cov(w, dims = 2)
    Σ = SymWoodbury(ref_cov, A, Λ)

    # get the "target" which is the mean subtracted cube
    cube_mean = mean(cube, dims = 1)
    flat_mean, psf = promote(flatten(cube_mean), psf)
    target = flatten(cube) .- flat_mean

    return JointModel(cube, cube_mean, target, design.angles, Σ, psf)
end

function (model::JointModel)(;params...)
    T = float(typeof(params[1]))
    base = zeros(T, size(model.cube))
    return inject_image!(base, T.(model.psf), model.angles; params...)
end

function reconstruct(model::JointModel; params...)
    μ = model(;params...) |> flatten
    AΛA = model.Σ.B * (model.Σ.D * model.Σ.B')
    recon = μ .+ (model.target .- μ) * (model.Σ \ AΛA') |> expand
    return recon .+ model.cube_mean
end

function loglikelihood(model::JointModel; params...)
    μ = model(;params...) |> flatten
    R = model.target .- μ
    return -tr(R * (model.Σ \ R')) / 2
end

function fake_covariance(mat::AbstractMatrix)
    D = var(mat, dims = 1)[1, :]
    @. D[iszero(D)] = Inf
    return Diagonal(D)
end
