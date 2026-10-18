---
title: "Policy Gradients from the Ground Up"
date: 2026-10-14
draft: false
math: true
tags:
  - reinforcement-learning
  - policy-gradient
summary: "From Monte Carlo estimates to REINFORCE."
---
Main problem: in supervised learning, you have a loss function to differentiate. In RL the reward is a black box you can only sample from. Everything else in this chapter is to handle that fact.

## Setting

Contextual bandit: see context $\to$ policy take action $\to$ get reward.
- context: $x\sim \mu$
- policy: $\pi_{\theta}(a\mid x)$
- reward: $r\sim R(x,a)$
- objective: $J(\theta)=\mathbb{E}[r]$

## Estimation

The objective $J(\theta)$ is an expectation over unknown distributions. We use samples to estimate the expectation. As long as the samples are independently sampled from the correct distribution this gives us an unbiased estimate, known as the **Monte Carlo** estimate.

The Monte Carlo estimate is unbiased and its variance scales as $\sigma^2_{\theta}/n$, for $n$ samples.

> [!derivation]- Unbiased: $\mathbb{E}[\hat{J}_{\theta}]=J(\theta)$
>
> - $\hat{J}_{\theta}=\frac{1}{n}\sum_{i}r_{i}$: random variable estimator, calculated from $n$ samples.
> - If the estimator is unbiased we would have $\mathbb{E}[\hat{J}_{\theta}]=J(\theta)$.
>
> $$
> \begin{aligned}
> \mathbb{E}[\hat{J}_{\theta}]
>   &= \mathbb{E}\left[ \frac{1}{n}\sum_{i} r_{i} \right]
>   && \text{definition of } \hat{J}_{\theta} \\[6pt]
>   &= \frac{1}{n}\sum_{i}\mathbb{E}[r_{i}]
>   && \text{linearity of } \mathbb{E} \\[6pt]
>   &= \frac{1}{n}\cdot n\cdot J(\theta)
>   && \text{each } r_{i} \text{ drawn from the process defining } J(\theta) \\[6pt]
>   &= J(\theta)
> \end{aligned}
> $$
> In expectation, the estimator is the true objective.

> [!derivation]- Variance: $\mathrm{Var}(\hat{J}_{\theta})=\sigma^2_{\theta}/n$
>
> Since our estimator is unbiased, $\mathbb{E}[\hat{J}_{\theta}] = J(\theta)$ and we can swap one for the other:
>
> $$
> \mathrm{Var}(\hat{J}_{\theta})
> = \mathbb{E}\big[(\hat{J}_{\theta}-\mathbb{E}[\hat{J}_{\theta}])^2\big]
> = \mathbb{E}\big[(\hat{J}_{\theta}-J(\theta))^2\big]
> $$
>
> $$
> \begin{aligned}
> \mathrm{Var}(\hat{J}_{\theta})
>   &= \mathbb{E}\left[ \left( \frac{1}{n}\sum_{i}r_{i} - J(\theta) \right)^{2} \right]
>   && \text{plug in } \hat{J}_{\theta} \\[6pt]
>   &= \mathbb{E}\left[ \left( \frac{1}{n}\sum_{i}r_{i} - \frac{1}{n}\sum_{i}J(\theta) \right)^{2} \right]
>   && J(\theta)=\tfrac{1}{n}\textstyle\sum_{i}J(\theta),\ \text{const.} \\[6pt]
>   &= \mathbb{E}\left[ \left( \frac{1}{n}\sum_{i}\big(r_{i}-J(\theta)\big) \right)^{2} \right]
>   && \text{merge under one sum} \\[6pt]
>   &\overset{\star}{=} \frac{1}{n^{2}}\sum_{i}\mathbb{E}\big[(r_{i}-J(\theta))^{2}\big]
>   && \text{cross terms vanish} \\[6pt]
>   &= \frac{1}{n^{2}}\cdot n\,\sigma^{2}_{\theta}
>   && \sigma^{2}_{\theta}=\mathbb{E}[(r_{i}-J(\theta))^{2}],\ n \text{ of them} \\[6pt]
>   &= \frac{\sigma^{2}_{\theta}}{n}
> \end{aligned}
> $$
>
> $\star$ Squaring the sum gives $n$ diagonal terms plus cross terms $(r_{i}-J)(r_{j}-J)$ for $i\neq j$. Independence splits the product, and unbiasedness makes each factor zero, so every cross term is $0\cdot 0=0$. The diagonal terms survive because squaring kills the sign cancellations.
>
> This verifies the intuition that more samples are better as $n$ samples reduces the variance by a factor of $n$. Since samples are expensive in RL, we can also reduce $\sigma^2_{\theta}$ itself which is what the next section is about.

### Variance Reduction

Variance in reward comes from three sources:
1. Noise inherent to the reward function
2. Randomness in the policy's actions
3. Variation due to different contexts

We can reduce variance by accounting for term 3, variation across contexts. Given a context, we want to know the relative increase in reward an action gives, not the raw reward for that context, because the nature of each context can lead to different scales of reward values.

Motivated by this observation, instead of raw reward, let's use a relative-to-average reward $r_{i}-b(x_{i})$. Intuitively this term represents whether that reward is better or worse than average for that context, when $b(x)\approx \mathbb{E}_{a\sim \pi_{\theta}(x),\, r\sim R(x,a)}[r]$.

Subtracting a context-dependent baseline shifts the objective by a $\theta$-independent constant, so $\nabla J_b = \nabla J$ and the argmax is unchanged.

> [!derivation]- Baseline is free: $J_b(\theta) = J(\theta) - \mathbb{E}_\mu[b(x)]$
>
> $$
> \begin{aligned}
> J_b(\theta) &= \mathbb{E}_{x\sim\mu}\Big[\mathbb{E}_{a\sim\pi_\theta(x)}\big[\mathbb{E}_{r\sim R(x,a)}[\,r-b(x)\,]\big]\Big]
>   && \text{definition of } J_b \\[6pt]
>   &= \mathbb{E}_{x}\Big[\mathbb{E}_{a}\big[\mathbb{E}_{r}[r]-\mathbb{E}_{r}[b(x)]\big]\Big]
>   && \text{linearity of } \mathbb{E} \\[6pt]
>   &= \mathbb{E}_{x}\Big[\mathbb{E}_{a}\big[\mathbb{E}_{r}[r]\big]-b(x)\Big]
>   && b(x) \text{ const. w.r.t. } r,a \\[6pt]
>   &= \underbrace{\mathbb{E}_{x}\Big[\mathbb{E}_{a}\big[\mathbb{E}_{r}[r]\big]\Big]}_{=\,J(\theta)}
>      -\;\mathbb{E}_{x\sim\mu}\big[b(x)\big]
>   && \text{linearity, outer layer} \\[6pt]
>   &= J(\theta)-\underbrace{\mathbb{E}_{x\sim\mu}[b(x)]}_{\text{const., no }\theta}
> \end{aligned}
> $$
>
> The objective is shifted by a constant value so the optimization problem isn't changed.

So how should we pick $b(x)$? A function of $x$ alone can only ever touch the between-context term, and it zeroes that term exactly when it matches the per-context mean $\mathbb{E}[r\mid x]$. This is why the target is $b(x)\approx\mathbb{E}_{a\sim\pi_{\theta}(x),\,r\sim R(x,a)}[r]$.

> [!derivation]- $b(x)$ only reaches the between-context term
>
> Law of total variance, conditioning on the context:
>
> $$
> \mathrm{Var}(r)=\underbrace{\mathbb{E}_{x}\big[\mathrm{Var}(r\mid x)\big]}_{\text{within context}}+\underbrace{\mathrm{Var}_{x}\big(\mathbb{E}[r\mid x]\big)}_{\text{between context}}
> $$
>
> The second term is the context driven variance coming from the fact that different contexts have different typical rewards.
>
> Now redo the decomposition on the baseline-corrected reward:
>
> $$
> \begin{aligned}
> \mathrm{Var}\big(r-b(x)\big)
>   &= \mathbb{E}_{x}\big[\mathrm{Var}(r-b(x)\mid x)\big]+\mathrm{Var}_{x}\big(\mathbb{E}[r-b(x)\mid x]\big)
>   && \text{law of total variance} \\[6pt]
>   &= \underbrace{\mathbb{E}_{x}\big[\mathrm{Var}(r\mid x)\big]}_{\text{unchanged}}+\mathrm{Var}_{x}\big(\mathbb{E}[r\mid x]-b(x)\big)
>   && b(x) \text{ const. given } x
> \end{aligned}
> $$
>
> when conditioning on $x$, $b(x)$ becomes a constant and a constant never changes the variance. We can still change the value of the second term by adjusting $b(x)$ so we have $b(x)$ track $\mathbb{E}[r\mid x]$ so that variance term goes to zero.

### Off-Policy Estimation

Everything so far assumes that the data was collected by the same $\pi_{\theta}$ that we are trying to train (on-policy). If instead the data was generated by a different policy $\pi_{\theta'}$, the average $\frac{1}{n}\sum_{i}r_{i}$ would estimate $J(\theta')$ not $J(\theta)$. The fix is to reweight each sample by how much more or less likely $\pi_{\theta}$ would have been to take that action.

#### The reweighting trick

> [!result] Fact 1.1 importance reweighting
> Let $\rho,\rho'$ be distributions over $u$, with coverage $\rho'(u)>0$ whenever $\rho(u)>0$. Then
> $$\mathbb{E}_{u\sim\rho'}\!\left[\frac{\rho(u)}{\rho'(u)}\,u\right] = \mathbb{E}_{u\sim\rho}[u]$$
> 

> [!derivation]- Proof of Fact 1.1
>
> $$
> \begin{aligned}
> \mathbb{E}_{u\sim\rho'}\!\left[\frac{\rho(u)}{\rho'(u)}u\right]
> &= \int_u \frac{\rho(u)}{\rho'(u)}\,u\;\rho'(u)\,du
> && \text{definition of }\mathbb{E}_{u\sim\rho'} \\[6pt]
> &= \int_u u\,\rho(u)\,du
> && \rho'(u)\text{ cancels} \\[6pt]
> &= \mathbb{E}_{u\sim\rho}[u]
> && \text{definition of }\mathbb{E}_{u\sim\rho}
> \end{aligned}
> $$

Applying this to $J(\theta)$ with data from $\pi_{\theta'}$, we get the importance-weighted estimator
$$
\hat{J}_{\theta}=\frac{1}{n}\sum_{i} \frac{\pi_{\theta}(a_{i}\mid x_{i})}{\pi_{\theta'}(a_{i}\mid x_{i})} r_{i}=\frac{1}{n}\sum_{i}w_{i}r_{i},\qquad \mathbb{E}_{\theta'}[\hat{J}_{\theta}]=J(\theta)
$$

#### Cost

Though we have an unbiased estimate of $J(\theta)$ using data from a different policy, the cost is an increased variance.

> [!result] Variance of a reweighted reward
> $$\mathrm{Var}_{\theta'}(w_i r_i)=\underbrace{\mathbb{E}_\theta\big[(w_i-1)r_i^2\big]}_{\text{off-policy penalty}}+\underbrace{\mathrm{Var}_\theta(r_i)}_{\text{on-policy variance}}$$
> The second term is what you would have had on-policy. The first is the price of reweighting, and in RL it is usually positive.

> [!derivation]- Variance decomposition
> $$
> \begin{aligned}
> \mathrm{Var}_{\theta'}(w_i r_i)
> &= \mathbb{E}_{\theta'}\big[w_i^2 r_i^2\big]-\big(\mathbb{E}_{\theta'}[w_ir_i]\big)^2 \\[4pt]
> &= \mathbb{E}_{\theta'}\big[w_i^2 r_i^2\big]-J(\theta)^2
> && \mathbb{E}_{\theta'}[w_ir_i]=J(\theta) \\[6pt]
> &= \mathbb{E}_{\theta}\big[w_i r_i^2\big]-J(\theta)^2
> && \text{Fact 1.1 on }f=w_ir_i^2 \\[6pt]
> &= \mathbb{E}_{\theta}\big[w_i r_i^2\big]-\mathbb{E}_\theta\big[r_i^2\big]+\mathbb{E}_\theta\big[r_i^2\big]-J(\theta)^2 \\[6pt]
> &= \mathbb{E}_{\theta}\big[(w_i-1)r_i^2\big]+\mathrm{Var}_\theta(r_i)
> \end{aligned}
> $$

## Policy Gradient

Estimation tells us how good a policy is, now we will focus on how to update and improve policies. Like with supervised learning, we will use gradient steps to improve our policy, except we cannot differentiate through the black-box reward, so we will estimate the gradient with samples.

### Improvement with Gradients

Gradients are useful because they give a local improvement direction: a small enough step along $\nabla J(\theta)$ increases $J$. This comes from Taylor's theorem, which says that for a small step $u$ the function is approximately linear $$ J(\theta+u)=J(\theta)+\nabla J(\theta)^{\top}u+\underbrace{\tfrac{1}{2}u^{\top}\nabla^{2}J(\theta_{0})u+\dots}_{\text{negligible when }|u|\le\varepsilon} $$so choosing the best small step is the constrained problem $\max_{|u|\le\varepsilon}\ J(\theta)+\nabla J(\theta)^{\top}u$.

The solution to this is $u^\star=\alpha \nabla J(\theta)$ with $\alpha=\frac{\varepsilon}{\mid \nabla J(\theta)\mid}$, this gives us the update
$$
\theta \leftarrow \theta + \alpha \nabla J(\theta)
$$

> [!derivation]- Small step solves to $u^{\star}=\alpha\nabla J(\theta)$
>
> - $J(\theta)$ is a constant with respect to $u$, so we only need to maximize the linear term.
> - Write $v=\nabla J(\theta)$.
>
> $$
> \begin{aligned}
> \max_{\Vert u\Vert\le\varepsilon}\; v^{\top}u
>   &= \max_{\Vert u\Vert\le\varepsilon}\; \Vert v\Vert\,\Vert u\Vert\cos\angle(u,v)
>   && \text{definition of dot product} \\[6pt]
>   &= \Vert v\Vert\cdot\varepsilon\cdot 1
>   && \text{take } \Vert u\Vert=\varepsilon,\ \cos=1 \\[6pt]
>   &\Rightarrow\; u^{\star}=\varepsilon\,\frac{v}{\Vert v\Vert}
>   && \text{parallel to } v,\ \text{length } \varepsilon
> \end{aligned}
> $$
>
> Any step of length $\varepsilon$ pointing along the gradient is the best first-order improvement. As long as $\alpha$ is small enough that the higher order terms are negligible, this is an improvement step.

In supervised learning the loss is known, so the gradient $\nabla_{\theta}\ell(f_{\theta}(x_{i}),y_{i})$ is computed directly per data point and averaged. For a probabilistic model under cross-entropy loss $\ell=-\log f_{\theta}(y\mid x)$ the update is $$ \theta\leftarrow\theta+\alpha\,\frac{1}{n}\sum_{i}\nabla\log f_{\theta}(y_{i}\mid x_{i}) $$ which pushes the model to make the observed labels more likely. Keep this in mind, REINFORCE will look almost identical.

### Gradient Estimation

We cannot compute $\nabla J(\theta)$ directly because $R(x,a)$ is unknown and we only see samples of it. We use two tricks to solve this, the reweighting trick from Fact 1.1, and the log derivative trick.

> [!result] Log-derivative trick
> 
> $$\nabla f(\theta)=f(\theta)\nabla\log f(\theta)$$ Follows from $\frac{d}{dx}\log x=1/x$ and the chain rule.

> [!result] Policy gradient
> 
> Let $\rho_{\theta}(x,a)=\mu(x)\pi_{\theta}(a\mid x)$ be the joint distribution over context and action. For any data collection policy $\pi_{\theta'}$ with coverage, $$\nabla J(\theta)=\mathbb{E}_{x,a\sim\rho_{\theta'}}\left[\frac{\pi_{\theta}(a\mid x)}{\pi_{\theta'}(a\mid x)}\nabla\log\pi_{\theta}(a\mid x)\mathbb{E}_{r\sim R(x,a)}[r]\right]$$ The gradient is an expectation over things we can sample, so a Monte Carlo estimate is immediate.

> [!derivation]- Policy gradient
> 
> - The only place $\theta$ appears in $J$ is in the sampling distribution $\rho_{\theta}$, which is exactly what makes it hard to differentiate.
> - Reweight so the expectation is over a fixed $\rho_{\theta'}$, then $\theta$ only appears inside the integrand.
> 
> $$ \begin{aligned} \nabla J(\theta) &= \nabla\mathbb{E}_{x,a\sim\rho_{\theta}}\big[\mathbb{E}_{r\sim R(x,a)}[r]\big] && \text{definition of } J \\[6pt] &= \nabla\mathbb{E}_{x,a\sim\rho_{\theta'}}\left[\frac{\rho_{\theta}(x,a)}{\rho_{\theta'}(x,a)}\mathbb{E}_{r}[r]\right] && \text{Fact 1.1, needs } \rho_{\theta'}>0 \text{ where } \rho_{\theta}>0 \\[6pt] &= \mathbb{E}_{x,a\sim\rho_{\theta'}}\left[\frac{\nabla\rho_{\theta}(x,a)}{\rho_{\theta'}(x,a)}\mathbb{E}_{r}[r]\right] && \text{expectation no longer depends on } \theta,\ \text{linearity} \\[6pt] &= \mathbb{E}_{x,a\sim\rho_{\theta'}}\left[\frac{\rho_{\theta}(x,a)}{\rho_{\theta'}(x,a)}\nabla\log\rho_{\theta}(x,a)\mathbb{E}_{r}[r]\right] && \text{log-derivative trick} \\[6pt] &= \mathbb{E}_{x,a\sim\rho_{\theta'}}\left[\frac{\pi_{\theta}(a\mid x)}{\pi_{\theta'}(a\mid x)}\nabla\log\pi_{\theta}(a\mid x)\mathbb{E}_{r}[r]\right] && \star \end{aligned} $$
> 
> $\star$ Since $\rho_{\theta}(x,a)=\mu(x)\pi_{\theta}(a\mid x)$, the $\mu(x)$ cancels in the ratio, and $\nabla\log\rho_{\theta}=\nabla\log\mu(x)+\nabla\log\pi_{\theta}(a\mid x)=\nabla\log\pi_{\theta}(a\mid x)$ because $\mu$ does not depend on $\theta$. This is the whole point of the log-derivative trick: the unknown context distribution disappears.

Setting $\theta'=\theta$ (on-policy data) makes every importance weight equal to one.

> [!result] REINFORCE
> 
> With samples $(x_{i},a_{i},r_{i})$ collected by $\pi_{\theta}$, $$\theta\leftarrow\theta+\alpha\frac{1}{n}\sum_{i}\nabla\log\pi_{\theta}(a_{i}\mid x_{i})r_{i}$$

Compare with the cross-entropy update from the previous section: it is the same form, except each data point is weighted by its reward. The policy is pushed to make high-reward actions more likely, and (for negative rewards) low-reward actions less likely.

In the general off-policy case the estimate keeps the weights, $$ \widehat{\nabla J}'_{\theta}=\frac{1}{n}\sum_{i}\underbrace{\frac{\pi_{\theta}(a_{i}\mid x_{i})}{\pi_{\theta'}(a_{i}\mid x_{i})}}_{w_{i}}\nabla\log\pi_{\theta}(a_{i}\mid x_{i})r_{i} $$ which increases the likelihood of actions that achieved high reward _and_ were comparatively unlikely under $\pi_{\theta'}$ (large $w_{i}$). Both contribute to high variance, especially over multiple updates: weights become positively correlated with reward, and the policy gets pushed into regions poorly covered by $\pi_{\theta'}$. This is why taking many off-policy gradient steps on one batch does not work well, and it motivates the next section.

Baselines carry over for free. In every estimate above we may replace $r_{i}$ with $r_{i}-b(x_{i})$ without introducing bias, since $\nabla J_{b}=\nabla J$.

## Trust Regions

The gradient ascent step was defined by $\Vert u\Vert\le\varepsilon$, a Euclidean ball in parameter space. This is an arbitrary choice: two parameter vectors that are close in $\ell_{2}$ can induce very different action distributions, and vice versa. A sigmoid policy near saturation is the canonical example, a huge step in $\theta$ barely moves the probability, so ordinary policy gradient crawls exactly when the policy has found the right answer and wants to commit. We want "small step" to mean small in _policy_ space.

> [!result] KL divergence
> 
> $$\mathrm{KL}(\rho'\Vert\rho)=\mathbb{E}_{u\sim\rho'}\left[\log\frac{\rho'(u)}{\rho(u)}\right]$$ The surprise at seeing data from $\rho'$ when expecting $\rho$. Not a true distance (asymmetric, no triangle inequality) but always non-negative and zero only when the distributions coincide.

We measure the distance between two policies by KL averaged over contexts: $$ d_{\mathrm{KL}}(\theta',\theta)=\mathbb{E}_{x\sim\mu}\big[\mathrm{KL}(\pi_{\theta'}(\cdot\mid x)\Vert\pi_{\theta}(\cdot\mid x))\big]=\mathbb{E}_{x,a\sim\rho_{\theta'}}\left[\log\frac{\pi_{\theta'}(a\mid x)}{\pi_{\theta}(a\mid x)}\right] $$ This is zero at $\theta'=\theta$, grows as the policies diverge, and does not care how the policies are parameterized.

### Natural Policy Gradient

Same recipe as before: first order approximation to $J$, but now the constraint is on $d_{\mathrm{KL}}$ instead of $\Vert u\Vert$. Since $d_{\mathrm{KL}}$ is not quadratic in $u$, we take its second order Taylor approximation. The zeroth and first order terms vanish, and the surviving quadratic term is defined by a matrix called the Fisher information.

> [!result] Fisher information matrix
> 
> $$F(\theta)=\mathbb{E}_{x,a\sim\rho_{\theta}}\big[\nabla\log\pi_{\theta}(a\mid x)\nabla\log\pi_{\theta}(a\mid x)^{\top}\big]$$ and for small $u$, $d_{\mathrm{KL}}(\theta,\theta+u)\approx u^{\top}F(\theta)u$. Read $F$ as a sensitivity meter: large in directions where wiggling $\theta$ changes the action distribution a lot, small where it barely matters.

> [!derivation]- Expected score is zero: $\mathbb{E}_{a\sim\pi_{\theta}(x)}[\nabla\log\pi_{\theta}(a\mid x)]=0$
> 
> This identity is used twice (first-order term of the KL expansion, and the PPO gradient).
> 
> $$ \begin{aligned} \mathbb{E}_{a\sim\pi_{\theta}(x)}\big[\nabla\log\pi_{\theta}(a\mid x)\big] &= \sum_{a}\pi_{\theta}(a\mid x)\nabla\log\pi_{\theta}(a\mid x) && \text{definition of } \mathbb{E} \\[6pt] &= \sum_{a}\nabla\pi_{\theta}(a\mid x) && \text{log-derivative trick in reverse} \\[6pt] &= \nabla\sum_{a}\pi_{\theta}(a\mid x) && \text{linearity} \\[6pt] &= \nabla 1 = 0 && \text{probabilities sum to one} \end{aligned} $$

> [!derivation]- $d_{\mathrm{KL}}(\theta,\theta+u)\approx u^{\top}F(\theta)u$
> 
> - Expand $d_{\mathrm{KL}}(\theta,\theta+u)=\mathbb{E}_{x,a\sim\rho_{\theta}}[\log\pi_{\theta}(a\mid x)-\log\pi_{\theta+u}(a\mid x)]$ in $u$ around $u=0$. Note the sampling distribution $\rho_{\theta}$ does not depend on $u$, so gradients in $u$ pass straight through the expectation.
> 
> **Zeroth order.** $d_{\mathrm{KL}}(\theta,\theta)=0$.
> 
> **First order.** $$ \begin{aligned} \nabla_{u}d_{\mathrm{KL}}(\theta,\theta+u) &= -\mathbb{E}_{x,a\sim\rho_{\theta}}\big[\nabla_{u}\log\pi_{\theta+u}(a\mid x)\big] && \text{only the second log depends on } u \\[6pt] &\overset{u=0}{=} -\mathbb{E}_{x,a\sim\rho_{\theta}}\big[\nabla_{\theta}\log\pi_{\theta}(a\mid x)\big] = 0 && \text{expected score is zero} \end{aligned} $$ Alternatively: $d_{\mathrm{KL}}\ge 0$ with equality at $u=0$, so $u=0$ is a minimum, hence a critical point.
> 
> **Second order.** Using $\nabla^{2}\log\pi=\dfrac{\nabla^{2}\pi}{\pi}-\nabla\log\pi\nabla\log\pi^{\top}$, $$ \begin{aligned} \nabla^{2}_{u}d_{\mathrm{KL}}\big|_{u=0} &= -\mathbb{E}_{x,a\sim\rho_{\theta}}\big[\nabla^{2}\log\pi_{\theta}(a\mid x)\big] \\[6pt] &= -\mathbb{E}_{x}\Big[\sum_{a}\nabla^{2}\pi_{\theta}(a\mid x)\Big]+\mathbb{E}_{x,a\sim\rho_{\theta}}\big[\nabla\log\pi_{\theta}\nabla\log\pi_{\theta}^{\top}\big] && \pi\cdot\tfrac{\nabla^{2}\pi}{\pi}=\nabla^{2}\pi \\[6pt] &= -\mathbb{E}_{x}\big[\nabla^{2}1\big]+F(\theta) = F(\theta) && \text{probabilities sum to one} \end{aligned} $$
> 
> So the expansion is $0+0+u^{\top}F(\theta)u+\dots$ (the notes absorb the $\tfrac{1}{2}$ into $\varepsilon$).

With the quadratic approximation the KL-constrained step is $$ \max_{u}\ \nabla J(\theta)^{\top}u\quad\text{s.t.}\quad u^{\top}F(\theta)u\le\varepsilon^{2} $$

> [!result] Natural policy gradient
> 
> The solution is $u^{\star}=\alpha F(\theta)^{-1}\nabla J(\theta)$ with $\alpha=\varepsilon\big(\nabla J(\theta)^{\top}F(\theta)^{-1}\nabla J(\theta)\big)^{-1/2}$, giving $$\theta\leftarrow\theta+\alpha F(\theta)^{-1}\nabla J(\theta)$$ $F^{-1}$ preconditions the gradient so equal steps in parameter space are equal steps in policy space. The update is (approximately) invariant to how $\pi_{\theta}$ is parameterized.

> [!derivation]- Solving the KL-constrained step
> 
> - The trick is a change of variables that turns the ellipsoid $u^{\top}Fu\le\varepsilon^{2}$ into a ball, after which it is the same problem as gradient ascent.
> - Let $v=F^{1/2}u$ and $c=F^{-1/2}\nabla J(\theta)$.
> 
> $$ \begin{aligned} \nabla J(\theta)^{\top}u &= \big(F^{1/2}c\big)^{\top}F^{-1/2}v = c^{\top}v && F^{1/2} \text{ symmetric, cancels} \\[6pt] u^{\top}Fu &= v^{\top}v=\Vert v\Vert^{2} && \text{substitute } u=F^{-1/2}v \\[6pt] \Rightarrow\quad v^{\star} &= \varepsilon\frac{c}{\Vert c\Vert} && \text{Euclidean problem from before} \\[6pt] \Rightarrow\quad u^{\star} &= F^{-1/2}v^{\star}=\varepsilon\frac{F^{-1}\nabla J(\theta)}{\Vert F^{-1/2}\nabla J(\theta)\Vert} && \text{undo change of variables} \\[6pt] &= \varepsilon\big(\nabla J^{\top}F^{-1}\nabla J\big)^{-1/2}F^{-1}\nabla J(\theta) && \Vert F^{-1/2}g\Vert^{2}=g^{\top}F^{-1}g \end{aligned} $$

> [!result] Why it fixes saturation (two-action sigmoid)
> 
> For $\pi(1)=\sigma(\theta)$ with rewards $r_{1},r_{0}$: $\nabla J=\sigma(1-\sigma)(r_{1}-r_{0})$ and $F=\sigma(1-\sigma)$. Plain PG steps shrink like $\sigma(1-\sigma)\to 0$ as the policy commits, so reaching $\pi(1)=1-\epsilon$ takes time exponential in $1/\epsilon$. NPG's step is $F^{-1}\nabla J=r_{1}-r_{0}$, a constant, so it takes time logarithmic in $1/\epsilon$.

In practice everything is estimated from on-policy samples $x_{i},a_{i},r_{i}\sim\rho_{\theta}$: $\nabla J$ by REINFORCE, and $$ \hat{F}_{\theta}=\frac{1}{n}\sum_{i}\nabla\log\pi_{\theta}(a_{i}\mid x_{i})\nabla\log\pi_{\theta}(a_{i}\mid x_{i})^{\top} $$

### Proximal Policy Optimization

NPG gives a principled direction but requires inverting an estimate of $F(\theta)$, a (number of parameters)$^{2}$ matrix. For deep networks with millions or billions of parameters this is out of the question. PPO keeps the idea that closeness should be measured in KL, but swaps the hard constraint for a soft penalty. Having collected data with $\theta'$, solve $$ \max_{\theta}\ J(\theta)-\lambda\,d_{\mathrm{KL}}(\theta',\theta)=\max_{\theta}\ J_{\lambda}(\theta;\theta') $$ Rather than a first or second order approximation, we take several ordinary gradient steps on this objective. Since the data came from $\pi_{\theta'}$, the $\nabla J$ term uses the off-policy REINFORCE estimate.

> [!derivation]- Gradient of the KL penalty
> 
> $$ \begin{aligned} \nabla_{\theta}d_{\mathrm{KL}}(\theta',\theta) &= \nabla_{\theta}\mathbb{E}_{x,a\sim\rho_{\theta'}}\big[\log\pi_{\theta'}(a\mid x)-\log\pi_{\theta}(a\mid x)\big] && \text{definition of } d_{\mathrm{KL}} \\[6pt] &= -\mathbb{E}_{x,a\sim\rho_{\theta'}}\big[\nabla_{\theta}\log\pi_{\theta}(a\mid x)\big] && \rho_{\theta'},\ \pi_{\theta'} \text{ do not depend on } \theta \end{aligned} $$
> 
> Note this is over $\rho_{\theta'}$, not $\rho_{\theta}$, so the expected-score identity does _not_ make it vanish. It is zero only at $\theta=\theta'$.

> [!result] PPO gradient
> 
> Combining the off-policy policy gradient with $-\lambda\nabla d_{\mathrm{KL}}$ and factoring out $\nabla\log\pi_{\theta}$, $$\nabla J_{\lambda}(\theta;\theta')=\mathbb{E}_{x,a\sim\rho_{\theta'}}\left[\nabla\log\pi_{\theta}(a\mid x)\left(\frac{\pi_{\theta}(a\mid x)}{\pi_{\theta'}(a\mid x)}\mathbb{E}_{r}[r]+\lambda\right)\right]$$ Estimated from data collected with $\pi_{\theta'}$, with the usual baseline, $$\widehat{\nabla J}'_{\lambda,\theta}=\frac{1}{n}\sum_{i}\nabla\log\pi_{\theta}(a_{i}\mid x_{i})\left(\frac{\pi_{\theta}(a_{i}\mid x_{i})}{\pi_{\theta'}(a_{i}\mid x_{i})}\big(r_{i}-b(x_{i})\big)+\lambda\right)$$

Compare with off-policy REINFORCE, which has only the importance-weighted reward term. The penalty adds a uniform bonus $\lambda$ to every (context, action) pair in the dataset: every observed action gets its likelihood nudged up regardless of reward. This encourages $\pi_{\theta}$ to imitate the data collection policy $\pi_{\theta'}$, keeps the KL small, and mitigates the variance blowup when $\theta$ wanders into regions not covered by $\theta'$. Because this is an off-policy estimate with the penalty holding the policy in place, we can now take multiple steps on one batch.

The most widely used variant, **PPO-clip**, replaces the penalty with switching logic on the importance weights: discard a data point from the gradient estimate as soon as (i) $w_{i}>1+\epsilon$ for a positive instance $r_{i}-b(x_{i})>0$, or (ii) $w_{i}<1-\epsilon$ for a negative instance. This caps the variance and stops the policy from continuing to update in low-coverage directions. Combining clipping with a _group_ baseline (for each context, sample $G$ actions and use their mean reward as $b(x_{i})$, optionally dividing by the group standard deviation) gives **GRPO**, which is what is widely used for post-training language models.
