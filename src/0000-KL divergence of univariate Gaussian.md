# KL divergence of univariate Gaussian

## Gaussian Distribution

$$
x\sim \mathcal{N}(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left\{-\frac{1}{2\sigma^2}(x-\mu)^2\right\}
$$

## KL divergence

$$
KL(P||Q) = \int p(x)\log\frac{q(x)}{p(x)}dx
$$

Given two Gaussian distribution $p(x)=\mathcal{N}(x|\mu_1, \sigma_1^2)$ and $q(x)=\mathcal{N}(x|\mu_2, \sigma_2^2)$, calculate the divergence $KL(P||Q)$.
$$
\begin{align}
p(x)\frac{q(x)}{p(x)} &= p(x)\left\{\log\frac{\sigma_2}{\sigma_1} -\frac{1}{2\sigma^2_1}(x-\mu_1)^2+ \frac{1}{2\sigma_2^2}(x-\mu_2)^2\right\} \\
&= p(x)\left\{\log\frac{\sigma_2}{\sigma_1} -\frac{1}{2\sigma^2_1}(x-\mu_1)^2+ \frac{1}{2\sigma_2^2}(x-\mu_1+\mu_1-\mu_2)^2 \right\}\\
&=p(x)\left\{\log\frac{\sigma_2}{\sigma_1}-\frac{1}{2\sigma^2_1}(x-\mu_1)^2 +\frac{1}{2\sigma_2^2}(x-\mu_1) +\frac{1}{2\sigma_2^2}(\mu_1-\mu_2)^2 \right\} \\
&=p(x)\left\{\log\frac{\sigma_2}{\sigma_1}+(\frac{1}{2\sigma^2_2}-\frac{1}{2\sigma^2_1})(x-\mu_1)^2 +\frac{1}{2\sigma_2^2}(\mu_1-\mu_2)^2 \right\} \\

\int p(x)\frac{q(x)}{p(x)} &= \log\frac{\sigma_2}{\sigma_1} + \frac{(\mu_1-\mu_2)^2}{2\sigma_2^2} + \frac{\sigma_1^2}{2\sigma^2_2}-\frac{1}{2}
\end{align}
$$
Use the property $E[(x-E[x])] = 0$ in the 3rd line.

## Python Implementation

```python
import numpy as np


def gaussian_kl(mu1, sigma1, mu2, sigma2):
    d = sigma2 / sigma1
    kl = np.log(d) + (mu1-mu2)**2 / (2*sigma2**2) + 0.5*(1./d)**2 - 0.5
    return kl
```

