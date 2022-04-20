    
using SparseArrays
using LinearAlgebra

function sixrd!(du, u, p, t=nothing)
    β = p[1]
    μ = p[2]
    α = p[3]
    κ = p[4]
    c = p[5]
    adj = p[6]
    adj -= Diagonal(adj) # remove self travels

    S = @view u[:, 1]
    I = @view u[:, 2]
    X = @view u[:, 3]
    R = @view u[:, 4]
    D = @view u[:, 5]

    N = S + I + X + R + D

    ΔN = N - transpose(sum(adj; dims=1)) + sum(adj; dims=2)

    Smat = adj * Diagonal(S .* inv.(N))
    Imat = adj * Diagonal(I .* inv.(N))

    ΔS⁻ = S - transpose(sum(Smat; dims=1))
    ΔI = I - transpose(sum(Imat; dims=1)) + sum(Imat; dims=2)
    ΔIΔN⁻¹ = ΔI ./ ΔN

    du[:, 5] .= α .* (I + X)
    du[:, 4] .= μ .* (I + X)
    du[:, 3] .= κ .* I - μ .* X - α .* X
    du[:, 2] .= (-κ - μ - α) .* I

    val1 = β .* c .* ΔS⁻ .* ΔIΔN⁻¹
    du[:, 2] .+= vec(val1)
    du[:, 1] .= -vec(val1)

    val2 = transpose(transpose(β .* c .* ΔIΔN⁻¹) * Smat)
    du[:, 2] .+= vec(val2)
    du[:, 1] .+= -vec(val2)

    return du
end

mutable struct SIXRDMetaPopParams
    params::Matrix{Float64}
    adj::SparseMatrixCSC{Float64}

    SIXRDMetaPopParams() = new()
end

function SIXRDMetaPopODE(state::Matrix{Float64}, params::Matrix{Float64},
    adj::SparseMatrixCSC{Float64})::Matrix{Float64}

    p = params; adj = adj 
    adj -= Diagonal(adj) # remove self travels
    num_nodes = convert(Int, size(state, 1))

    if size(p, 1) != size(adj, 1)
        throw(DimensionMismatch("p must have same number of rows as adj"))
    elseif size(p, 2) != 5
        throw(DimensionMismatch("p must have 5 columns"))
    elseif size(adj, 1) != size(adj, 2)
        throw(DimensionMismatch("adj must be square"))
    end

    Sidx = 1;Iidx = 2;Xidx = 3;Ridx = 4;Didx = 5
    βidx = 1;cidx = 2;μidx = 3;αidx = 4;κidx = 5
    
    β = @view p[:, 1]
    c = @view p[:, 2]
    μ = @view p[:, 3]
    α = @view p[:, 4]
    κ = @view p[:, 5]
    
    S = @view state[:, Sidx]
    I = @view state[:, Iidx]
    X = @view state[:, Xidx]
    R = @view state[:, Ridx]
    D = @view state[:, Didx]
    
    N = S + I + X + R + D
    
    ret_mat = deepcopy(state)
    
    newS = @view ret_mat[:, Sidx]
    newI = @view ret_mat[:, Iidx]
    newX = @view ret_mat[:, Xidx]
    newR = @view ret_mat[:, Ridx]
    newD = @view ret_mat[:, Didx]
    
    ΔN = N - transpose(sum(adj, dims=1)) + sum(adj, dims=2)
    
    Smat = adj * Diagonal(S .* inv.(N))
    Imat = adj * Diagonal(I .* inv.(N))
    
    ΔS⁻ = S - transpose(sum(Smat, dims=1))
    ΔI = I - transpose(sum(Imat, dims=1)) + sum(Imat, dims=2)
    ΔIΔN⁻¹ = ΔI ./ ΔN
    
    newD .+= α .* (I + X)
    newR .+= μ .* (I + X)
    newX .+= κ .* I - μ .* X - α .* X
    newI .+= (-κ - μ - α) .* I
    
    val1 = β .* c .* ΔS⁻ .* ΔIΔN⁻¹
    newI .+= vec(val1)
    newS .-= vec(val1)
    
    val2 = transpose(transpose(β .* c .* ΔIΔN⁻¹) * Smat)
    newI .+= vec(val2)
    newS .-= vec(val2)
    
    return ret_mat
end

function accumulate_groups(vec1::AbstractVector, vec2::AbstractVector)
    groups = unique(vec2)

    d = Dict{eltype(vec2),eltype(vec1)}()
    for i in groups
        d[i] = sum(vec1[vec2 .== i])
    end
    return d
end
