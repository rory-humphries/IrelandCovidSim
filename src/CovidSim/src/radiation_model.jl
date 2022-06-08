function radiation_sij(d, n, j)
    return sum(n[0 .< d .< d[j]])
end

function radiation_si(d, n)
    N = length(d)
    svec = zeros(N)
    srtd_inds = sortperm(d)

    first_ind = 1
    for i in srtd_inds
        if d[i] == 0
            first_ind += 1
        else
            break
        end
    end

    s = 0
    for k in first_ind:N
        j = srtd_inds[k]
        svec[j] = s
        s += n[j]
    end

    return svec
end

function radiation_model(m, n, s)
    return (m .* n) ./ ((m .+ s) .* (m .+ n .+ s))
end


