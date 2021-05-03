## Data Formatting

There are a few data formatting issues to note when using the **LRMoE** package,
which are summarized below. Throughout this page, we assume there are `N = 100` insurance policyholders, 
who have two coverages `y_1` and `y_2` (dimension of response `D = 2`). Each policyholder has
5 covariates (number of covariates `P = 5`).

- Covariate `X`: The dimension of `X` should be 100 by 5 (`= N * P`), which is relatively straightforward.
    If there are factor covariates (e.g. indicators for urban and rural region), the intercept term should 
    be manually added, and the user needs to manually augment a column of zero-one indicator.

- Response `Y` (Exact Case): If both `y_1` and `y_2` are observed exactly (i.e. the incurred losses are not
    censored or truncated), then we can set `exact_Y = true` in the initialization and fitting function.
    In this case, the dimension of `Y` should be 100 by 2 (`= N * D`): the first column is for `y_1` and
    the second for `y_2`.

- Response `Y` (Not Exact Case): If either `y_1` or `y_2` is not observed exactly, then `exact_Y = false` and
    the dimension of `Y` should be 100 by 8 (`= N * 4D`): the first four columns are for `y_1` and the
    remaining for `y_2`. For each block of 4 columns, they should be structured as (`tl`, `yl`, `yu`, `tu`),
    corresponding to the lower bound of truncation, lower bound of censoring, upper bound of censoring and
    upper bound of truncation, respectively. Some typical cases are listed below:
    - Both `y_1` and `y_2` are observed exactly: Assume `y_1 = 2.0` and `y_2 = 3.0`, then the first row of `Y`
        should be `[0.0 2.0 2.0 Inf 0.0 3.0 3.0 Inf]`. Alternatively, we can set `exact_Y = true` (see the 
        previous case), and also set the first row of `Y` as `[2.0 3.0]`.
    - `y_1` is observed exactly, `y_2` is left-truncated at `1.0` but observed exactly (e.g. an insurance deductible):
        Assume `y_1 = 2.0` and `y_2 = 3.0`, then the first row of `Y` should be `[0.0 2.0 2.0 Inf 1.0 3.0 3.0 Inf]`.
    - `y_1` is observed exactly, `y_2` is right-censored at `2.0` (e.g. a payment limit):
        Assume `y_1 = 2.0` and `y_2 = 3.0`, then the first row of `Y` should be `[0.0 2.0 2.0 Inf 0.0 2.0 Inf Inf]`.
    - `y_1` is observed exactly, `y_2` is only observed within a range:
        Assume `y_1 = 2.0` and `y_2 = 3.0`, but we only observe `2.5 < y_2 < 3.5`,
        then the first row of `Y` should be `[0.0 2.0 2.0 Inf 0.0 2.5 3.5 Inf]`.

- Logit Regression Coefficients `α`: Assume we would like to fit an LRMoE with three latent classes (`g = 3`), then
    the dimension of `α` should be 5 by 3 (` = N * g`). For example, a noninformative guess can be 
    initialized as `α = fill(0.0, 5, 3)`.

- Component Distribution `comp_dist`: Assume we would like to fit an LRMoE with three lateht classes (`g = 3`), then
    the dimension of `comp_dist` should be 2 by 3 (`= D * g`). For example, if we assume `y_1` is a mixture of lognormals
    and `y_2` is a mixture of gammas, then `comp_dist = [LogNormalExpert(1.0, 2.0) LogNormalExpert(1.5, 2.5) LogNormalExpert(2.0, 3.0); GammaExpert(1.0, 2.0) GammaExpert(1.5, 2.5) GammaExpert(2.0, 3.0)]`. Note that the columns of `comp_dist` should be
    distinct, otherwise the model is not identifiable.

- Penalty on Logit Regression Coefficients `pen_α`: It should be a single number (i.e. a uniform penalty imposed on 
    all coefficients in `α`). For example, when `pen_α = 2.0`, then the penalty `sum( (α ./ 2.0).^2 )` is subtracted
    from the loglikelihood as a penalty. In other words, we would not like the magnitude of `α` to be too large.

- Penalty on Parameters of Expert Functions `pen_params`: Using the `comp_dist` mentioned above, `pen_params` should be
    a matrix of size 2 by 3 (`= D * g`), where each entry is a vector of real numbers penalizing the parameters of
    expert functions. Usually, the user can leave this argument as default. For more details, the user is referred to the package source code.


