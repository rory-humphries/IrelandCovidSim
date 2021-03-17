using DelimitedFiles
using StatsBase
using Random
using Distributions
using SparseArrays

function gen_sparse_array(x::Array{Categorical,1}, vals::Array{Int64,1})
    
    mts_ = MersenneTwister.(1:Threads.nthreads())

    sz = sum(vals) 
    I = zeros(Int, sz)
    J = zeros(Int, sz)
    V = ones(Int, sz)

    Threads.@threads for i = 1:size(x, 1)
        dst = rand(mts_[Threads.threadid()], x[i], vals[i])
        offset = sum(vals[1:i - 1])
        for j = 1:size(dst, 1)
            I[j + offset] = dst[j]
            J[j + offset] = i
        end
    end
    return sparse(I, J, V)
end


