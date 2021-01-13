# Predictive Functions

After fitting an LRMoE model, the following predictive functions provide further insights into the dataset.
These functions start with `predict_` followed by a quantity of interest (e.g. `mean_`) listed below.
* `class`: latent class probabilities and the most likely latent class;
* `mean`: mean of response;
* `var`: variance of response;
* `limit`: limited expected value (LEV) of response, that is, ``E[{min}(Y, d)]``;
* `excess`: expected excess value of response, that is, ``E[{max}(Y-d, 0)]``; and
* `VaRCTE`: quantile (or Value-at-Risk/VaR) and conditional tail expectation (CTE, or tail-VaR/TVaR) of response.

These quantities can be calculated based on either the `prior` and `posterior` latent class probabilities, as
indicated by the suffix of these functions.
* `prior`: the latent class probabilities are based on the covariates `X` and logit regression coefficients `α`.
* `posterior`: the latent class probabilities are based on the covariates `X`, logit regression coefficients `α` and observed values `Y`.

The differences of these probabilities can be found in [Fung et al. (2019)](https://www.cambridge.org/core/journals/astin-bulletin-journal-of-the-iaa/article/class-of-mixture-of-experts-models-for-general-insurance-application-to-correlated-claim-frequencies/E9FCCAD03E68C3908008448B806BAF8E).

The following contains a detailed description of all predictive functions included in the package.
Throughout this page, `Y` is a matrix of response, `X` a matrix of covariates, `α` a matrix of logit regression coefficients and
`model` a matrix of expert functions.

```@docs
predict_class_prior
predict_class_posterior
predict_mean_prior
predict_mean_posterior
predict_var_prior
predict_var_posterior
predict_limit_prior
predict_limit_posterior
predict_excess_prior
predict_excess_posterior
predict_VaRCTE_prior
predict_VaRCTE_posterior
```




