# Modelling Framework

The LRMoE model is formulated as follows. Let ``(\mathbf{x}_i, \mathbf{y}_i), i = 1, 2, \dots, n`` denote a set of observations, where ``\mathbf{x}_i`` denotes the covariates and ``\mathbf{y}_i`` the response(s).

Given ``\mathbf{x}_i``, the ``i``-th observation is classified into one of ``g`` latent classes by the so-called **logit gating function**. The probability of belonging to the ``j``-th latent class is given by

```math
\pi_j(\mathbf{x}_i; \mathbf{\alpha}) = \frac{\exp (\mathbf{\alpha}_j^T \mathbf{x}_i)}{\sum_{j'=1}^{g} \exp (\mathbf{\alpha}_{j'}^T \mathbf{x}_i)}, \quad j = 1, 2, \dots, g-1
```
 For model identifiability reasons, we assume ``\mathbf{\alpha}_g = \mathbf{0}`` which corresponds to the reference class.

 Conditional on the latent class ``j``, the distribution of the response ``\mathbf{y}_i`` is given by an **expert function** with density

```math
f_j(\mathbf{y}_i; \mathbf{\psi}_j) = \prod_{d=1}^{D} f_{jd}(\mathbf{y}_{id}; \mathbf{\psi}_{jd})
```
where we assume conditional independence of dimensions ``1, 2, \dots, D`` of ``\mathbf{y}_i``, if it is a vector of responses.

The likelihood function is therefore
```math
L(\mathbf{\alpha}, \mathbf{\psi}; \mathbf{x}, \mathbf{y}) = \prod_{i=1}^{n} \left\{ \sum_{j=1}^{g} \pi_j(\mathbf{x}_i; \mathbf{\alpha}) f_j(\mathbf{y}_i; \mathbf{\psi}_j) \right\}
```

Notice that the parameters ``\mathbf{\psi}_j`` do not involve regression on the covariates ``\mathbf{x}_i``, hence the model is termed as **reduced**. For an introduction to the general mixture-of-experts models, see e.g. [here](https://en.wikipedia.org/wiki/Mixture_of_experts).

 [Fung et al. (2019)](https://www.sciencedirect.com/science/article/pii/S0167668719303956) have shown that such simplification of model structure will not reduce its flexibility, and will significantly reduce the computation efforts in model inference. The parameters to estimate are the regression coefficients ``\mathbf{\alpha}_j`` and parameters of the expert functions ``\mathbf{\psi}_j``, which is implemented by the standard Expectation-Conditional-Maximization algorithm (details omitted).




