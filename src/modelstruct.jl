# Abstract type: whether the model has random effects
abstract type RandomEffects end
struct hasRandomEffects <: RandomEffects end
struct noRandomEffects <: RandomEffects end

# Model
abstract type LRMoEModel{re <: RandomEffects} end

const LRMoEModelSTD = LRMoEModel{noRandomEffects}
const LRMoEModelRE = LRMoEModel{hasRandomEffects}

# Result
abstract type LRMoEFittingResult end

# Model contains: regression coefficients, component distributions
#   number of iteration, convergence, likelihood, AIC, BIC -> Move to another struct
struct LRMoESTD <: LRMoEModelSTD
    α::Array
    comp_dist::Array
    LRMoESTD(α, comp_dist) = ( size(α)[1] == size(comp_dist)[2] ? new(α, comp_dist) : error("Invalid specification of model.") )
end

struct LRMoESTDFit <: LRMoEFittingResult 
    model_fit::LRMoESTD
    converge::Bool
    iter::Integer
    loglik::Real
    loglik_np::Real
    AIC::Real
    BIC::Real
    LRMoESTDFit(model_fit, converge, iter, loglik, loglik_np, AIC, BIC) = new(model_fit, converge, iter, loglik, loglik_np, AIC, BIC)
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
    println("$(m.model_fit.comp_dist)")
end

# tmp = LRMoESTD([1 2 ; 3 4], ["a" "b"])
# tmp1 = LRMoESTDFit(tmp, true, 20, 1.1, 1.2, 1.3)
# summary(tmp1)