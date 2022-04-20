struct GravityModel{Ut,Vt,Ft,Pt} <: DiscreteMultivariateDistribution
    u::Ut
    v::Vt

    f::Ft

    p::Pt

    c::Float64
end

function GravityModel(u, v, f, p)
    c = zeros(Threads.nthreads())
    N = size(u, 2)

    Threads.@threads for i in 1:N
        ui = u[:, i]
        for j in 1:N
            c[Threads.threadid()] += f(ui, u[:, j], v[i, j], p)
        end
    end
    return GravityModel(u, v, f, p, sum(c))
end

Distributions.length(d::GravityModel) = 2
Distributions.size(d::GravityModel) = (2,)

function Distributions._logpdf(d::GravityModel, x::AbstractVector)
    @unpack u, v, f, p, c = d

    return log(f(u[:, x[1]], u[:, x[2]], v[x[1], x[2]], p) / c)
end

struct GravityModelSampler2 <: Sampleable{Multivariate,Discrete}
    d::GravityModel
    rowsum::Vector{Float64} # probability of leaving from node i
    rowsumsum::Float64
end

function Distributions.sampler(d::GravityModel)
    N = size(d.u, 2)
    rowsum = zeros(N)
    for i in 1:N
        rowsum[i] += sum(pdf(d, [[i, x] for x in 1:N]))
    end
    return GravityModelSampler2(d, rowsum, sum(rowsum))
end

Base.length(s::GravityModelSampler2) = 2

function Distributions._rand!(
    rng::AbstractRNG, s::GravityModelSampler2, x::AbstractVector{T}
) where {T<:Integer}
    N = length(s.rowsum)
    row = sample(Weights(s.rowsum, s.rowsumsum))
    col = sample(Weights(pdf(s.d, [[row, x] for x in 1:N])))
    return x .= [row, col]
end

function Distributions._rand!(
    rng::AbstractRNG, s::GravityModelSampler2, A::DenseMatrix{T}
) where {T<:Integer}
    N = length(s.rowsum)
    Nsamps = size(A, 2)

    row_samps = sample(1:N, Weights(s.rowsum, s.rowsumsum), Nsamps)

    k = 1
    for (i, count) in countmap(row_samps)
        col_samps = sample(1:N, Weights(pdf(s.d, [[i, x] for x in 1:N])), count)

        A[1, k:(k + count - 1)] .= i
        A[2, k:(k + count - 1)] .= col_samps
        k += count
    end
    return A
end

function exp_gravity(ui, uj, v, p)
    (α, β, γ) = p
    attr_i = ui[1]
    attr_j = uj[1]
    dist = v[1]
    return attr_i^α * attr_j^β * exp(-γ * dist)
end
