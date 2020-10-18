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