## Copied from Distributions.jl
## macro for argument checking
macro check_args(D, cond)
    quote
        if !($(esc(cond)))
            throw(ArgumentError(string(
                $(string(D)), ": the condition ", $(string(cond)), " is not satisfied.")))
        end
    end
end

# rowlogsumexps
rowlogsumexp(x) = logsumexp.(x[row,:] for row in 1:size(x)[1])

# replace nan by a number
function nan2num(x, g)
    for i in eachindex(x)
        @inbounds x[i] = ifelse(isnan(x[i]), g, x[i])
    end
end

# matching functions for fast integration
function unique_bounds(l, u)
    return unique(hcat(vec(l), vec(u)), dims = 1)
end

matchrow(a, B) = findfirst( i -> all(j->a[j] == B[i,j], 1:size(B,2)), 1:size(B,1) )

function match_unique_bounds(all_bounds, unique_bounds)
    return [matchrow(all_bounds[i,:], unique_bounds) for i in 1:size(all_bounds)[1]]
end

# yl = vec([1 2 3 4 5 6 1 2 3 4 5 6])
# yu = vec([7 8 9 10 11 12 7 9 8 12 11 10])

# ab = hcat(vec(yl), vec(yu))
# ub = unique_bounds(yl, yu)
# match_unique_bounds(ab, ub)

# unique_bounds(yl, yu)
# match_unique_bounds(yl, yu)

# recursively solve for quantiles of discrete distributions
function _solve_discrete_quantile(d::DiscreteUnivariateDistribution, q::Real)
    l, u = 1, 2*1
    while cdf.(d, u) < q
        l = u+1
        u = 2*l
    end
    while u-l > 1
        tmp = ceil((u+l)/2)
        if cdf.(d, tmp) >= q
            l, u = l, tmp
        else
            l, u = tmp, u
        end
    end
    if cdf.(d, l) >= q
        return l
    else
        return u
    end
end