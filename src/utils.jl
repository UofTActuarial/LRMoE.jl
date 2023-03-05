## Copied from Distributions.jl
## macro for argument checking
macro check_args(D, cond)
    quote
        if !($(esc(cond)))
            throw(
                ArgumentError(
                    string(
                        $(string(D)), ": the condition ", $(string(cond)),
                        " is not satisfied."),
                ),
            )
        end
    end
end

# rowlogsumexps
rowlogsumexp(x) = logsumexp(x; dims=2)

# replace nan by a number
function nan2num(x, g)
    for i in eachindex(x)
        @inbounds x[i] = ifelse(isnan(x[i]), g, x[i])
    end
end

# replace inf by a number
function inf2num(x, g)
    for i in eachindex(x)
        @inbounds x[i] = ifelse(isinf(x[i]), g, x[i])
    end
end

# matching functions for fast integration
function unique_bounds(l, u)
    return unique(hcat(vec(l), vec(u)); dims=1)
end

matchrow(a, B) = findfirst(i -> all(j -> a[j] == B[i, j], 1:size(B, 2)), 1:size(B, 1))

function match_unique_bounds(all_bounds, unique_bounds)
    return [matchrow(all_bounds[i, :], unique_bounds) for i in 1:size(all_bounds)[1]]
end

function match_unique_bounds_threaded(all_bounds, unique_bounds)
    result = fill(1, size(all_bounds)[1])
    @threads for i in 1:size(all_bounds)[1]
        result[i] = matchrow(all_bounds[i, :], unique_bounds)
    end
    return result
end

# recursively solve for quantiles of discrete distributions
function _solve_discrete_quantile(d::DiscreteUnivariateDistribution, q::Real)
    l, u = 1, 2 * 1
    while cdf.(d, u) < q
        l = u + 1
        u = 2 * l
    end
    while u - l > 1
        tmp = ceil((u + l) / 2)
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

# Convert exact Y to full Y
function _exact_to_full(Y)
    result = fill(NaN, size(Y)[1], size(Y)[2] * 4)
    for j in 1:size(Y)[2]
        result[:, 4 * (j - 1) + 1] .= 0.0
        result[:, 4 * (j - 1) + 2] .= Y[:, j]
        result[:, 4 * (j - 1) + 3] .= Y[:, j]
        result[:, 4 * (j - 1) + 4] .= Inf
    end
    return result
end