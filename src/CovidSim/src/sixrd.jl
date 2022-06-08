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

    ΔN = N + sum(adj; dims=1)' - sum(adj; dims=2)

    Smat = Diagonal(S .* inv.(N)) * adj
    Imat = Diagonal(I .* inv.(N)) * adj

    ΔS⁻ = S - sum(Smat; dims=2)
    ΔI = I - sum(Imat; dims=2) + sum(Imat; dims=1)'
    ΔIΔN⁻¹ = ΔI ./ ΔN

    du[:, 5] .= α .* (I + X)
    du[:, 4] .= μ .* (I + X)
    du[:, 3] .= κ .* I - μ .* X - α .* X
    du[:, 2] .= (-κ - μ - α) .* I

    val1 = β .* c .* ΔS⁻ .* ΔIΔN⁻¹
    du[:, 2] .+= vec(val1)
    du[:, 1] .= -vec(val1)

    val2 = β .* c .* (Smat * ΔIΔN⁻¹)
    du[:, 2] .+= vec(val2)
    du[:, 1] .+= -vec(val2)

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
