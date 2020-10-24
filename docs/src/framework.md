# Modelling Framework

The LRMoE model is formulated as follows. Let ``(\mathbf{x}_i, \mathbf{y}_i), i = 1, 2, \dots, n`` denote a set of observations, where ``\mathbf{x}_i`` denotes the covariates and ``\mathbf{y}_i`` the response(s).

Given  ``\mathbf{x}_i``, the ``i``-th observation is classified into one of ``g`` latent classes by the so-called **logit gating function**. The probability of belonging to the ``j``-th latent class is given by

```math
\pi_j(\mathbf{x}_i; \mathbf{\alpha}) = \frac{\exp (\mathbf{\alpha}_j^T \mathbf{x}_i)}{\sum_{j'=1}^{g} \exp (\mathbf{\alpha}_{j'}^T \mathbf{x}_i)}, \quad j = 1, 2, \dots, g-1
```
 For model identifiability reasons, we assume ``\mathbf{\alpha}_g = \mathbf{0}`` which corresponds to the reference class.

 Conditional on the latent class ``j``, the distribution of the response ``\mathbf{y}_i`` is given by an **expert function**.
