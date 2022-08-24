# Abstract type: whether the model has random effects
abstract type RandomEffects end
struct hasRandomEffects <: RandomEffects end
struct noRandomEffects <: RandomEffects end

# Model
abstract type LRMoEModel{re<:RandomEffects} end

const LRMoEModelSTD = LRMoEModel{noRandomEffects}
const LRMoEModelRE = LRMoEModel{hasRandomEffects}

# Result
abstract type LRMoEFittingResult end

# Model contains: regression coefficients, component distributions
#   number of iteration, convergence, likelihood, AIC, BIC -> Move to another struct
struct LRMoESTD <: LRMoEModelSTD
    α::Array
    comp_dist::Array
    function LRMoESTD(α, comp_dist)
        return (
            if size(α)[1] == size(comp_dist)[2]
                new(α, comp_dist)
            else
                error("Invalid specification of model.")
            end
        )
    end
end

struct LRMoESTDFit <: LRMoEFittingResult
    model_fit::LRMoESTD
    converge::Bool
    iter::Integer
    loglik::Real
    loglik_np::Real
    AIC::Real
    BIC::Real
    function LRMoESTDFit(model_fit, converge, iter, loglik, loglik_np, AIC, BIC)
        return new(model_fit, converge, iter, loglik, loglik_np, AIC, BIC)
    end
end

"""
    summary(obj)

Summarizes a fitted LRMoE model.

# Arguments
- `obj`: An object returned by `fit_LRMoE` function.

# Return Values
Prints out a summary of the fitted LRMoE model on screen.
"""
function summary(m::LRMoESTDFit)
    println("Model: LRMoE")
    if m.converge
        println("Fitting converged after $(m.iter) iterations")
    else
        println("Fitting NOT converged after $(m.iter) iterations")
    end
    println("Dimension of response: $(size(m.model_fit.comp_dist)[1])")
    println("Number of components: $(size(m.model_fit.comp_dist)[2])")
    println("Loglik: $(m.loglik)")
    println("Loglik (no penalty): $(m.loglik_np)")
    println("AIC: $(m.AIC)")
    println("BIC: $(m.BIC)")
    println("Fitted α:")
    println("$(m.model_fit.α)")
    println("Fitted component distributions:")
    return println("$(m.model_fit.comp_dist)")
end
