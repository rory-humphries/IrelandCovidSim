function sixrd_metapop!(du, u, p, t=nothing)
    β = p[1]
    μ = p[2]
    α = p[3]
    κ = p[4]
    c = p[5]

    W = p[6]
    W -= Diagonal(W) # remove self travels
    Nv = size(W, 1)

    S = @view u[:, 1]
    I = @view u[:, 2]
    X = @view u[:, 3]
    R = @view u[:, 4]
    D = @view u[:, 5]

    N = S + I + X + R + D

    Wi = W * ones(Nv)

    du[:, 1] .= -(β .* c .* S .* I ./ N) + (W' * S) - (S .* Wi)
    du[:, 2] .= (β .* c .* S .* I ./ N) - ((μ .+ α .+ κ) .* I) + (W' * I) - (I .* Wi)
    du[:, 3] .= (κ .* I) - ((μ + α) .* X) + (W' * X) - (X .* Wi)
    du[:, 4] .= (μ .* I) + (μ .* X) + (W' * R) - (R .* Wi)
    du[:, 5] .= (α .* I) + (α .* X) + (W' * D) - (D .* Wi)

    return du
end

function sixrd!(du, u, p, t=nothing)
    β = p[1]
    μ = p[2]
    α = p[3]
    κ = p[4]
    c = p[5]
    
    S = u[1]
    I = u[2]
    X = u[3]
    R = u[4]
    D = u[5]

    N = S + I + X + R + D

    du[1] = -(β * c * S .* I ./ N) 
    du[2] = (β * c * S .* I ./ N) - ((μ + α + κ) * I)
    du[3] = (κ * I) - ((μ + α) * X)
    du[4] = (μ * I) + (μ * X)
    du[5] = (α * I) + (α * X)

    return du
end

function accumulate_groups(vec1::AbstractVector, vec2::AbstractVector)
    groups = unique(vec2)

    d = Dict{eltype(vec2),eltype(vec1)}()
    for i in groups
        d[i] = sum(vec1[vec2 .== i])
    end
    return d
end
