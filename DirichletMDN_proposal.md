# Data-driven joint PDF for three-stream mixing via Dirichlet-Mixture Density Networks

A plan for replacing the pixel-wise softmax DNN of Yellapantula et al. (2019) with a parametric mixture model that lives natively on the 2-simplex, subsuming the analytical hierarchy of Perry & Mueller (2018).

## 1. Problem context

Three-component passive mixing in turbulent flow produces two independent mixture fractions $(Z_1, Z_2)$ subject to $Z_1 + Z_2 + Z_3 = 1$ with $Z_i \ge 0$. The joint subgrid PDF $P(Z_1, Z_2)$ is supported on the unit triangle (the 2-simplex):

$$
\mathcal{T} = \{(Z_1, Z_2) : Z_1 \ge 0,\ Z_2 \ge 0,\ Z_1 + Z_2 \le 1\}.
$$

For LES closure, we need a model that maps subgrid moments $(\tilde Z_1, \tilde Z_2, \widetilde{Z_1''^2}, \widetilde{Z_2''^2}, \widetilde{Z_1''Z_2''})$ to $P(Z_1, Z_2)$ that is **valid** (non-negative, normalized, supported on $\mathcal{T}$), **moment-consistent**, and **accurate** across all mixing regimes (equal, favored, layered, premixed).

### 1.1 Analytical baseline (Perry & Mueller 2018)

Perry & Mueller laid out a hierarchy of bivariate-beta distributions, with selection based on the physical mixing configuration:

| Distribution | Parameters | Constraint |
|---|---|---|
| Dirichlet | 3 | All component pairs negatively correlated; all permutations neutral |
| Beta-Delta | 3 | $Z_2, Z_3$ perfectly correlated |
| CM (1/2/3) | 4 | One permutation neutral; one favored pair |
| BVB5 (12/13/23) | 5 | Two permutations neutral; positive correlations allowed |
| BVB6 | 6 | Fully general but requires 3rd-order moment |
| SMLD | 5 | Maximum entropy; no beta marginals; no foreknowledge needed |

The decision tree (their Fig. 1) requires knowing the mixing configuration to pick the right distribution — exactly what an LES closure does not know a priori.

### 1.2 Limitations of the current ML approach (MLPDF.py)

The current model predicts all 4096 entries of a $64 \times 64$ histogram of $P(Z_1, Z_2)$ via a softmax output layer trained with BCE loss. Limitations:

1. ~50% of output cells are forced to be zero (upper triangle) — wasted capacity.
2. No physical priors: support, normalization, beta marginals, moment consistency are all *learned* rather than *enforced*.
3. RF model balloons to ~48 GB; DNN spuriously places mass at pure-component corners (visible in I3, I4 figures).
4. Heavy imbalance: most binned PDFs are near-$\delta$ at late times → easy regime dominates training.
5. Layered cases (L1, L2, PL1) — the hardest for the analytical models — also remain hard for the pixel-wise DNN.

## 2. What the model represents — background on Mixture Density Networks

This section explains the model class conceptually before §3 specifies the concrete architecture for the three-stream problem.

### 2.1 What is a Mixture Density Network?

A standard regression network learns a function $f_\theta : x \mapsto y$ that outputs a single prediction. A **mixture density network** (MDN; Bishop 1994) learns a function that outputs the parameters of a *probability distribution* over $y$:

$$
f_\theta : x \mapsto \{\text{parameters of } p(y \mid x)\}.
$$

Instead of predicting "the answer," it predicts "the *distribution of possible answers*, conditioned on $x$." The trained network plus the parametric family together define a **conditional density model** $p_\theta(y \mid x)$.

Training is by maximum likelihood: given dataset $\{(x^{(i)}, y^{(i)})\}$, minimize the negative log-likelihood

$$
\mathcal{L}(\theta) = -\sum_i \log p_\theta(y^{(i)} \mid x^{(i)}).
$$

The "mixture" part means $p_\theta(y \mid x)$ is a weighted sum of $K$ simpler base distributions:

$$
p_\theta(y \mid x) = \sum_{k=1}^K \pi_k(x)\,p_\text{base}(y; \boldsymbol{\phi}_k(x)).
$$

Both the mixing weights $\pi_k$ (on the simplex, via softmax) and the per-component parameters $\boldsymbol{\phi}_k$ (with activations matching each parameter's domain) are functions of $x$ — the network outputs all of them simultaneously.

The classical Bishop (1994) MDN uses Gaussian base components, $p_\text{base} = \mathcal{N}(\mu_k(x), \sigma_k^2(x))$. Replacing Gaussians with another base family is what distinguishes Dirichlet-MDNs and logistic-normal-MDNs.

### 2.2 What the network is modeling in this problem

For three-stream mixing:

- **Input** $x = (\tilde Z_1, \widetilde{Z_1''^2}, \tilde Z_2, \widetilde{Z_2''^2}, \widetilde{Z_1''Z_2''}) \in \mathbb{R}^5$ — the LES subgrid moments.
- **Output** $y = (Z_1, Z_2) \in \mathcal{T}$ — a single random draw of mixture fractions from a subgrid point inside the filter cell.

The network is modeling

$$
p_\theta(Z_1, Z_2 \mid \tilde Z_1, \widetilde{Z_1''^2}, \tilde Z_2, \widetilde{Z_2''^2}, \widetilde{Z_1''Z_2''})
$$

— exactly the same object as Perry & Mueller's analytical PDF families, but where the mapping from moments to PDF parameters is *learned* rather than hand-derived.

**Important: it is *not* modeling PDF values pixel-by-pixel.** This is the key shift from the current `MLPDF.py`:

|                              | Current pixel-wise softmax DNN          | Dirichlet MDN                              |
|------------------------------|------------------------------------------|---------------------------------------------|
| Network output               | 4096 numbers (PDF on a fixed 64×64 grid) | $\sim 30$ numbers (mixture params)          |
| What output represents       | $p(Z_1^i, Z_2^j)$ at predetermined grid points | The *parameters* of an analytical PDF family |
| PDF reconstructed by         | Reading off network outputs              | Evaluating the parametric formula           |
| Support, normalization       | Have to be learned from data             | Built into the parametric form              |
| Resolution                   | Fixed at training time                   | Continuous; query at any $(Z_1, Z_2)$       |

The MDN learns the *recipe for constructing the PDF*, not the PDF values themselves.

### 2.3 What a Dirichlet base component represents

A Dirichlet distribution lives natively on the 2-simplex $\mathcal{T}$. Parameterized by a concentration vector $\boldsymbol{\alpha} = (a_1, a_2, a_3)$ with each $a_i > 0$:

$$
\mathrm{Dir}(Z_1, Z_2, Z_3; \boldsymbol{\alpha}) = \frac{\Gamma(a_1+a_2+a_3)}{\Gamma(a_1)\Gamma(a_2)\Gamma(a_3)}\,Z_1^{a_1-1} Z_2^{a_2-1} Z_3^{a_3-1}.
$$

Three intuitions:

1. **Location.** The mean sits at $\bar Z_i = a_i / \alpha_0$ where $\alpha_0 = \sum_i a_i$. So $\boldsymbol{\alpha}$ encodes both *where* the mass is centered on the simplex and *how concentrated* it is.

2. **Spread.** The total $\alpha_0$ controls concentration:
   - $\alpha_0 \to \infty$ → mass collapses to a delta at the mean (fully-mixed limit)
   - $\alpha_0 \to 0$ → mass piles up at the corners (pure-component limit)
   - $\alpha_0 \sim O(1)$ → broad distribution covering much of the simplex (early-time mixing)

3. **What one Dirichlet can and cannot represent.** A single Dirichlet has unimodal density with negative correlations between all component pairs — exactly the analytical Dirichlet model in Perry & Mueller, same equations, same limitations. It cannot represent multimodality, positive correlations, or asymmetric mixing.

### 2.4 What the mixture of Dirichlets adds

A weighted sum of $K$ Dirichlets, each with its own $\boldsymbol{\alpha}_k$ and weight $\pi_k$, is a **universal density approximator** on the simplex. Concretely, in your problem:

| Mixing regime              | What the mixture does |
|----------------------------|-----------------------|
| Late-time equilibrated     | One dominant component with large $\alpha_0$, $\bar Z \approx (\tilde Z_1, \tilde Z_2, \tilde Z_3)$. Recovers single Dirichlet. |
| Early-time pure components | $K \approx 3$ components, each with small $\alpha_0$ centered near a corner. Weights $\pi_k$ track prescribed feed fractions. |
| Layered (L1, L2)           | 2–3 components on a line through the simplex, reproducing the bimodal/trimodal structure that single-Dirichlet models fail at. |
| Premixed (PI1, PI2)        | One narrow high-$\alpha_0$ component along the premixing line, plus a broader one for residual mixing. |
| Favored mixing (CM-like)   | Asymmetric weights on components positioned to break the all-pairs-negative-correlation constraint of a single Dirichlet. |

Mixture parameter count is small ($K + 3K \approx 32$ for $K=8$), but the resulting family is rich enough to interpolate continuously across all Perry & Mueller regimes — and it does so as a function of the input moments, *learned* from DNS.

### 2.5 The full picture

```
input moments                      forward pass
(5 numbers)         ─────────────► neural network ─────► (π₁,…,π_K, α₁,…,α_K)
                                                                │
                                                                ▼
                                                  analytical mixture-of-Dirichlets
                                                                │
                                                                ▼
   evaluate at any (Z₁, Z₂)    ◄────────────  p_θ(Z₁, Z₂ | moments)
   or sample from it directly
```

The network is a learned **moment → mixture-parameter map**. The mixture *family* (Dirichlet) acts as a hard prior — every output is guaranteed to be a valid PDF on the simplex. Training adjusts the network so that for each moment input, the produced mixture assigns high likelihood to the DNS particle samples observed in the corresponding filter cell.

This is why the model needs so few parameters compared to the current pixel-wise DNN: the *analytical structure* of bivariate-beta distributions does most of the work, and the network only has to handle the moments-to-parameters mapping — a 5-to-30 numerical regression, not a 5-to-4096 image regression.

### 2.6 Logistic-normal as an alternative base family

A logistic-normal random variable on the simplex is constructed by drawing a Gaussian $\mathbf{u} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ in $\mathbb{R}^{K-1}$ and mapping it through the additive log-ratio inverse:

$$
Z_i = \frac{\exp(u_i)}{1 + \sum_{j=1}^{K-1} \exp(u_j)}, \quad Z_K = \frac{1}{1 + \sum_{j=1}^{K-1} \exp(u_j)}.
$$

The density on the simplex is closed-form via change of variables, so NLL training works the same way.

**What it gives you that Dirichlet doesn't:** an arbitrary covariance structure $\boldsymbol{\Sigma}$ between transformed coordinates. Dirichlet forces all $\mathrm{Cov}(Z_i, Z_j) < 0$ — the limitation that drove Perry & Mueller to invent CM and BVB5 in the first place. Logistic-normal can represent positive correlations within a single component.

**What it costs you:**

- No closed-form moments — $\mathbb{E}[Z_i]$ requires Monte Carlo or numerical integration. This breaks the analytical moment-matching loss term (§3.4), which is one of the cleanest features of the Dirichlet formulation.
- Density diverges at corners ($Z_i = 0$), where early-time mixing PDFs concentrate. Needs clipping or a zero-augmented variant (Bear & Billheimer 2016).
- More parameters per component: $(K-1) + K(K-1)/2$ vs. Dirichlet's 3 (for the 2-simplex).

**Recommendation:** start with the Dirichlet mixture. The Perry & Mueller hierarchy is bivariate-beta-based, the simplex corners are physically meaningful (pure feed streams), and the closed-form moment-matching loss is a significant advantage. If a Dirichlet mixture with reasonable $K$ persistently struggles with CM-3-like positive-correlation cases (BVB5-23), revisit logistic-normal as either a replacement or a hybrid component.

### 2.7 Why this framing matters for the paper

The cleanest description in the paper is **not** "we trained a bigger neural network." It is:

> The network is a learned moment-to-parameter map for a fixed parametric family (mixture of Dirichlets on the simplex). The family is chosen because (i) each component coincides with the Dirichlet distribution that Perry & Mueller (2018) identified as the natural baseline for three-component mixing, (ii) the mixture is a universal density approximator on the simplex and recovers CM, BVB5, and Beta-Delta as limiting cases, and (iii) validity, support, normalization, and moment self-consistency are built into the analytical family rather than learned from data.

This is a much sharper claim than "DNN with 4096 outputs is more accurate than the analytical model," and gives the model a clear interpretation that maps onto the existing combustion literature.

## 3. Proposed model: Dirichlet Mixture Density Network

### 3.1 Architecture

A small MLP $f_\theta$ ingests the moment vector $\mathbf{m} = (\tilde Z_1, \tilde Z_2, \widetilde{Z_1''^2}, \widetilde{Z_2''^2}, \widetilde{Z_1''Z_2''}) \in \mathbb{R}^5$ and outputs the parameters of a $K$-component **mixture of Dirichlet distributions** on the 2-simplex:

$$
P(Z_1, Z_2 \mid \mathbf{m}) = \sum_{k=1}^{K} \pi_k(\mathbf{m})\,\mathrm{Dir}\bigl((Z_1, Z_2, 1{-}Z_1{-}Z_2);\ \boldsymbol{\alpha}_k(\mathbf{m})\bigr)
$$

with

- mixture weights: $\boldsymbol{\pi}(\mathbf{m}) \in \Delta^{K-1}$ via softmax
- concentration vectors: $\boldsymbol{\alpha}_k(\mathbf{m}) \in \mathbb{R}_{>0}^3$ via softplus

**Output dimensionality:** $K + 3K \approx 24\text{–}40$ for $K \in [6, 10]$, vs. 4096 today.

A reasonable starting architecture:

```
Input (5) → Linear(5→128) → SiLU → Linear(128→256) → SiLU
         → Linear(256→256) → SiLU
         → Split:
              head_π : Linear(256→K)   → softmax → π
              head_α : Linear(256→3K)  → softplus → α (reshape K×3)
```

### 3.2 Why mixtures of Dirichlets?

A single Dirichlet cannot represent favored mixing or multimodal early-time PDFs. A *mixture* of Dirichlets can:

- $K = 1$ recovers the Dirichlet limit of Perry & Mueller exactly.
- $K = 2$ with one near-$\delta$ component reproduces Beta-Delta limits.
- Several components with disjoint support recover CM- and BVB5-like asymmetric mixing.
- Many near-$\delta$ components recover early-time corner-concentrated PDFs (cases I1 at $t \to 0$).
- The mixture has beta-distribution marginals (a mixture of betas — actually richer than the analytical BVB5 marginals, which are infinite sums of betas anyway).

**Closed-form properties for Dirichlet mixtures** (useful for both training and validation):

For a single Dirichlet with $\alpha = (a_1, a_2, a_3)$, $\alpha_0 = \sum a_i$:

$$
\mathbb{E}[Z_i] = a_i/\alpha_0, \quad
\mathrm{Var}[Z_i] = \tfrac{a_i(\alpha_0 - a_i)}{\alpha_0^2(\alpha_0 + 1)}, \quad
\mathrm{Cov}[Z_i, Z_j] = -\tfrac{a_i a_j}{\alpha_0^2(\alpha_0 + 1)}.
$$

For the mixture, $\mathbb{E}[Z_i] = \sum_k \pi_k \mathbb{E}_k[Z_i]$ and second moments follow analogously by the law of total variance. This means the moment-recovery loss below is differentiable in closed form.

### 3.3 How to choose $K$

A common confusion: $K$ is **not** a parameter that sets where in the simplex the model can operate. The Perry & Mueller analytical models accept any $(\tilde Z_1, \tilde Z_2)$ in the simplex through their moment-to-parameter formulas; the MDN does the same thing through its continuous moment-to-parameter network, *for any $K$ including $K = 1$*. So there are two distinct senses of "general" that should not be conflated:

| Sense | What controls it |
|---|---|
| **A. Coverage in mean-value space** — can the model produce a PDF for any $(\tilde Z_1, \tilde Z_2)$ in $\mathcal{T}$? | Training data coverage in moment space + smoothness of the moment → param mapping + trunk capacity. **Not $K$.** |
| **B. Shape complexity at a fixed mean** — for given moments, can the model represent any subgrid PDF shape consistent with those moments? | $K$ (and the base-family expressiveness). |

The architecture in §3.1 handles Sense A through the network trunk for any $K$. $K$ controls only Sense B — how rich a shape the mixture can take for one input.

#### What shape complexity does the Perry & Mueller test suite demand?

| Regime | Shape characteristic | Components needed |
|---|---|---|
| Late-time, fully mixed | Near-$\delta$ at the mean | 1 (high $\alpha_0$) |
| Smooth intermediate mixing | Unimodal, broad | 1–2 |
| Asymmetric / favored mixing (CM-like) | Unimodal with positive correlation | 2–3 (single Dirichlet cannot do this) |
| Early-time pure components (I1 at $t \to 0$) | Three $\delta$-like peaks at corners | 3–4 |
| Layered (L1, L2, PL1) | Bimodal / trimodal along a line | 3–5 |
| Premixed (PI1–PI3, PL1) | Sharp ridge + diffuse residual | 2–4 |
| Boundary transitions | Mass on an edge, one component near zero | 2–4 |

The pathological regime is early-time pure components: three near-delta components plus a partial-mixing component → **$K = 4$ minimum**, every component doing useful work.

#### Practical reasons the nominal $K$ should be larger than the per-shape minimum

1. **Mode-collapse slack.** In trained MDNs, 1–3 components typically end up with vanishingly small $\pi_k$ on any given input. Effective $K$ is ~60–80% of nominal $K$.
2. **Universal-approximation rate.** Dirichlet mixtures on the simplex achieve $L^1$ error roughly $\sim K^{-1/d}$ for smooth densities ($d = 2$ here). The compositional-data literature typically uses $K \approx 10\text{–}30$ for ~1% $L^1$ error.
3. **Layered cases** — the ones where Perry & Mueller's analytical models struggled most — need multi-modal mixtures. Do not under-budget for them.
4. **You now have ~156 DNS configurations** worth of training data. A capacity-limited model leaves accuracy on the table.

#### Recommendation

| $K$ | Use as |
|---|---|
| 1 | Sanity baseline. Reproduces the single-Dirichlet analytical model; should match Perry & Mueller's Dirichlet result. |
| 4 | Lower bound on practical setting. Tests whether the architecture handles hard cases at all. |
| **8** | **First serious run.** Reasonable default for the Perry & Mueller test suite. |
| **16** | **Recommended target.** Slack for mode-collapse; handles layered + early-time corner regimes comfortably. |
| 32 | Capacity ceiling for clean ablation. If $K = 32$ beats $K = 16$, you are capacity-limited and the trunk / regularization needs revisiting before adding more components. |
| 64+ | Diminishing returns, unstable training. At this point a normalizing flow is the better next step. |

Run a $K$-ablation as part of the paper: $K \in \{1, 2, 4, 8, 16, 32\}$. Plot test NLL (and joint-PDF $L^1$ from Eq. 1 of your 2019 paper) vs. $K$. The resulting curve is the **data-driven analog of Perry & Mueller's Fig. 1 decision tree**, but as a continuous capacity dial rather than a discrete choice among Dirichlet (3 params) / CM (4) / BVB5 (5).

#### Adaptive-$K$ alternatives (future work)

Fixed $K$ wastes capacity where it isn't needed and runs short where it is. Two principled fixes:

1. **Sparse mixtures.** Add a strong entropy penalty on $\boldsymbol{\pi}$ plus an $L^1$ penalty pushing weights to zero. Effective $K$ varies with input automatically. Simple modification of the loss in §3.4.
2. **Stick-breaking / DPM-style truncation.** Parameterize $\boldsymbol{\pi}$ via stick-breaking with a large $K_{\max}$ (say 32) and a concentration prior favoring few components. Components past the active set decay to zero. More principled, more involved.

For a first paper, fixed-$K$ ablation is sufficient; flag adaptive-$K$ as a natural follow-up.

### 3.4 Loss function

Train end-to-end with negative log-likelihood on DNS samples, plus an auxiliary moment-matching term:

$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^N \log P(Z_1^{(i)}, Z_2^{(i)} \mid \mathbf{m}^{(i)}; \theta) + \lambda_m \, \mathcal{L}_{\text{mom}} + \lambda_e \, \mathcal{L}_{\text{ent}}.
$$

where:

- $\mathcal{L}_{\text{mom}} = \| \mathbf{m}(\boldsymbol{\pi}, \boldsymbol{\alpha}) - \mathbf{m}_{\text{input}} \|_2^2$ — enforces that the predicted mixture has the moments it was conditioned on. Uses the closed-form expressions in §3.2.
- $\mathcal{L}_{\text{ent}} = -\beta \sum_k \pi_k \log \pi_k$ — entropy regularizer on mixture weights to prevent collapse to a single component (standard trick for MDNs; see Eigen et al. 2013, Pereyra et al. 2017).
- Hyperparameters: $\lambda_m \in [0.1, 1]$, $\lambda_e \in [10^{-3}, 10^{-2}]$.

The Dirichlet log-density is closed-form and differentiable, so no reparameterization tricks are needed for training (if you later want to sample-and-backprop, use the Kumaraswamy reparameterization of Wu et al. 2024).

### 3.5 Connection to the Perry & Mueller hierarchy

This mixture model **subsumes the analytical decision tree**:

| Mixing regime | Recovered by mixture |
|---|---|
| All components mix equally → Dirichlet | $K{=}1$, single concentration vector |
| Favored pair → CM | $K{=}2$, components separated in favored direction |
| Two favored pairs → BVB5 | $K{=}3$ with asymmetric concentrations |
| Full premixing → Beta-Delta | $K{=}2$ with one collapsed component |
| Unknown / layered | Many active components, learned from data |

The network *learns* which configuration is active from the input moments — no foreknowledge needed.

## 4. Training sample generation

### 4.1 Conceptual framing — what is a training example?

The $w^3$ cubic box acts as an LES box filter applied to the DNS field. The $\sim w^3$ DNS cell values $(Z_1, Z_2)$ inside that box *are* the empirical subgrid PDF that the model must approximate. So:

> **One box = one filter cell = one training PDF.** The $\sim w^3$ cell values inside it are *samples from that one PDF*, not independent training examples.

Translating the box around the DNS domain produces *different filter cells* — different conditioning moments and a different subgrid PDF — and each such filter cell is a new training example. Multiple box widths $w$ produce filter cells of different sizes, which span different parts of moment space (small $w$ → high variance, near-corner; large $w$ → low variance, near-mean).

This reframing has three consequences for data generation:

1. **Storage per box is bounded by the cost of representing one empirical PDF**, not by particle count. Either a binned histogram or a subsampled set of particles is fine.
2. **Stride controls how densely moment space is sampled**, not sample independence. Overlapping boxes give distinct (correlated) filter cells with smoothly varying moments — fine for training, but redundant.
3. **The right balance knob is moment-space stratification of filter cells**, applied after extraction.

### 4.2 What's wrong with the current pipeline (`EnsightPDFml.py:OuputPDF`)

1. **Pre-binning into $64 \times 64$ histograms** locks in the pixel representation we're trying to escape; can't reuse for MDN/flow NLL training.
2. **Single fixed bin grid** — no flexibility to change representation, resolution, or scalar convention later without rerunning extraction.
3. **No moment-space balancing** — late-time near-$\delta$ filter cells dominate; rare configurations (high variance, near-corner means, layered transitions) are under-represented. The model overfits the easy regime.
4. **No use of $S_3$ permutation symmetry** of the three streams — wastes a free 6× data multiplier.
5. **Stride choice is opaque** — stride 32 on width 64 produces ~75% redundancy in moment space that's never deduplicated. Inefficient.

### 4.3 Revised pipeline (five changes, in order of impact)

**(1) Save the empirical PDF as particle samples, not as a pre-computed histogram.**
For each filter cell (box), save:
- The 5 moments $(\tilde Z_1, \widetilde{Z_1''^2}, \tilde Z_2, \widetilde{Z_2''^2}, \widetilde{Z_1''Z_2''})$
- $N \in [2048, 8192]$ DNS cell values $(Z_1, Z_2)$ drawn without replacement from the $\sim w^3$ cells inside the box

These samples are the empirical-PDF representation of that one filter cell. At training time, the Dirichlet-mixture NLL is evaluated on them directly (no binning). At validation time, bin them at any resolution you like. Storage is comparable to the current 64×64 histogram (~32 KB) but resolution-free.

This is the same data that the current pipeline already accesses inside `OuputPDF` (the `np.ravel(ScalarA[block])` step) — we're just persisting samples instead of throwing them away after the histogram is built.

**(2) Stratified selection in 5-D moment space.**
After all filter cells are extracted, bin them in 5-D moment space (~$10^5$ coarse bins). Cap each bin at $K \approx 50$ filter cells, keeping a representative subset across each cell. Outcome: dataset reduced 5–20×, with rare regions (high-variance, near-corner, layered transitions) at full weight rather than drowned by late-time near-$\delta$ cells.

**(3) Treat stride as a coverage knob, not an IID knob.**
- For *training* extraction: use a small stride (e.g., $w/4$) to sample moment space densely. Rely on step (2) to prune the resulting near-duplicates. Net effect is better moment-space coverage at fixed budget.
- For *validation* extraction: use the same stride; the held-out *configurations* (not held-out boxes) keep validation honest. See §5.1.

**(4) $S_3$ permutation augmentation.**
For each filter cell, generate up to 6 training records by relabeling components: $(Z_1, Z_2, Z_3) \to (Z_2, Z_1, Z_3)$, $\to (Z_3, Z_1, Z_2)$, etc. Apply the relabeling to *both* the moments (analytically) and the particle samples. Valid because Perry & Mueller's setup uses physically identical passive scalars. Free 6× data multiplier; bakes the permutation symmetry of the Dirichlet base distribution into the empirical training distribution rather than asking the network to discover it.

**(5) Streaming / online dataset, with active-learning option.**
Replace `.gz` files of histograms with a PyTorch `Dataset` that reads DNS chunks lazily and computes (moments, samples) for one filter cell at a time. Lets you change box width, sample count, augmentation strategy without re-running extraction. Reorganize on-disk DNS to **zarr** or **HDF5** chunked storage so a $w^3$ block is one read.

Once a baseline model exists, add an active-learning pass: score every filter cell by NLL or JSD vs. its empirical samples, add the top-$K$ worst-fitting cells to the training set, retrain. Targets the genuinely hard PDFs (layered transitions, early-time multimodal, premixing rings) that uniform sampling under-represents.

### 4.4 Expected scale

For 156 configurations × ~100 snapshots × 3 box widths:

| Stage | Filter-cell count (training examples) |
|---|---|
| Raw cells extracted, stride $w/4$ | $\sim 10^9$ |
| Raw cells extracted, stride $w$ (no overlap) | $\sim 10^7$ |
| After moment-space stratification (cap 50/bin) | $\sim 10^5$–$10^6$ |
| After $S_3$ augmentation | $\sim 10^6$–$10^7$ |

A balanced $10^6$-cell dataset with subsampled particles per cell will train a better conditional density model than $10^9$ unbalanced raw cells, at a fraction of the I/O cost. Active learning closes the remaining tail.

### 4.5 Drop-in replacement for `OuputPDF`

```python
import numpy as np
import itertools

def ExtractFilterCells(scA, scB, nx, width, stride, n_samples=4096, rng=None):
    """For each translated box (= one LES filter cell), return its moments and
    a subsampled empirical representation of its subgrid PDF.

    One returned record = one training PDF. The `samples` field is a sparse
    representation of the empirical subgrid PDF inside the box; it is *not*
    a set of independent training examples.
    """
    rng = rng or np.random.default_rng()
    records = []
    for i, j, k in itertools.product(range(width, nx + width, stride),
                                     repeat=3):
        block = np.s_[i-width:i+width, j-width:j+width, k-width:k+width]
        a = scA[block].ravel()
        b = scB[block].ravel()
        m1, m2 = a.mean(), b.mean()
        v1, v2 = ((a - m1) ** 2).mean(), ((b - m2) ** 2).mean()
        cov    = ((a - m1) * (b - m2)).mean()
        idx = rng.choice(a.size, size=n_samples, replace=False)
        records.append({
            "moments": np.array([m1, v1, m2, v2, cov], dtype=np.float32),
            "samples": np.stack([a[idx], b[idx]], axis=1).astype(np.float32),
        })
    return records


def StratifyMomentSpace(records, bins_per_dim=10, cap_per_bin=50, rng=None):
    """Cap the number of filter cells per 5-D moment-space bin so that rare
    mixing regimes (high variance, near-corner means, layered transitions)
    are not drowned by abundant late-time near-delta cells.
    """
    rng = rng or np.random.default_rng()
    M = np.stack([r["moments"] for r in records])
    edges = [np.linspace(M[:, d].min(), M[:, d].max(), bins_per_dim + 1)
             for d in range(5)]
    keys = np.stack([np.digitize(M[:, d], edges[d][1:-1]) for d in range(5)],
                    axis=1)
    keys_t = [tuple(k) for k in keys]
    from collections import defaultdict
    buckets = defaultdict(list)
    for idx, key in enumerate(keys_t):
        buckets[key].append(idx)
    keep = []
    for idxs in buckets.values():
        chosen = rng.choice(idxs, size=min(cap_per_bin, len(idxs)),
                            replace=False)
        keep.extend(chosen.tolist())
    return [records[i] for i in keep]


def AugmentPermutations(records):
    """S_3 permutation augmentation. For each filter cell, generate up to 6
    records by relabeling components (Z1, Z2, Z3) -> sigma(Z1, Z2, Z3).
    Both samples and moments transform consistently. Valid because Perry &
    Mueller's three streams are physically identical passive scalars.
    """
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2),
             (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    aug = []
    for r in records:
        Z1, Z2 = r["samples"][:, 0], r["samples"][:, 1]
        Z3 = 1.0 - Z1 - Z2
        Z  = np.stack([Z1, Z2, Z3], axis=1)
        for p in perms:
            Zp = Z[:, list(p)]
            a, b = Zp[:, 0], Zp[:, 1]
            m1, m2 = a.mean(), b.mean()
            v1, v2 = ((a - m1) ** 2).mean(), ((b - m2) ** 2).mean()
            cov    = ((a - m1) * (b - m2)).mean()
            aug.append({
                "moments": np.array([m1, v1, m2, v2, cov], dtype=np.float32),
                "samples": np.stack([a, b], axis=1).astype(np.float32),
            })
    return aug
```

## 5. Training, validation, and test loop

### 5.1 Data split — by configuration, not by box

**Critical:** split train/val/test by *initial condition* (the 26 mean values × 6 case types), not by individual boxes. Otherwise validation leaks: many boxes from the same DNS run look almost identical, and an apparently strong validation score reflects memorization rather than generalization.

A reasonable split for your 26 means:
- 18 training configs (per case type)
- 4 validation configs (per case type, used for hyperparameter tuning and early stopping)
- 4 test configs (per case type, untouched until final reporting)

Plus one held-out *case type* if you want to test cross-regime generalization (e.g., train on isotropic + premixed, test on layered).

### 5.2 PyTorch dataset and model

```python
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class PDFSampleDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        return (torch.from_numpy(r["moments"]),
                torch.from_numpy(r["samples"]))


class DirichletMDN(nn.Module):
    def __init__(self, n_in=5, hidden=256, K=8):
        super().__init__()
        self.K = K
        self.trunk = nn.Sequential(
            nn.Linear(n_in, 128), nn.SiLU(),
            nn.Linear(128, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head_pi    = nn.Linear(hidden, K)
        self.head_alpha = nn.Linear(hidden, 3 * K)

    def forward(self, m):
        h = self.trunk(m)
        pi    = torch.softmax(self.head_pi(h), dim=-1)         # (B, K)
        alpha = torch.nn.functional.softplus(self.head_alpha(h)) + 1e-3
        alpha = alpha.view(-1, self.K, 3)                       # (B, K, 3)
        return pi, alpha


def dirichlet_logpdf(z, alpha, eps=1e-6):
    """z: (..., 3), alpha: (..., 3). Returns log p(z)."""
    z = z.clamp(eps, 1.0 - eps)
    z = z / z.sum(-1, keepdim=True)
    log_B = torch.lgamma(alpha).sum(-1) - torch.lgamma(alpha.sum(-1))
    return ((alpha - 1.0) * torch.log(z)).sum(-1) - log_B


def mixture_nll(pi, alpha, samples):
    """pi: (B, K), alpha: (B, K, 3), samples: (B, N, 2). Returns scalar NLL."""
    B, N, _ = samples.shape
    Z3 = 1.0 - samples.sum(-1, keepdim=True)
    Z  = torch.cat([samples, Z3], dim=-1)                       # (B, N, 3)

    # Broadcast: (B, N, K, 3) for densities
    Z_e     = Z.unsqueeze(2).expand(-1, -1, alpha.shape[1], -1)
    alpha_e = alpha.unsqueeze(1).expand(-1, N, -1, -1)
    log_p_k = dirichlet_logpdf(Z_e, alpha_e)                    # (B, N, K)

    log_pi  = torch.log(pi + 1e-12).unsqueeze(1)                # (B, 1, K)
    log_mix = torch.logsumexp(log_pi + log_p_k, dim=-1)         # (B, N)
    return -log_mix.mean()


def moment_loss(pi, alpha, m_in):
    """Closed-form moment-matching penalty on the mixture."""
    a0 = alpha.sum(-1)                                          # (B, K)
    means = alpha / a0.unsqueeze(-1)                            # (B, K, 3)
    var   = means * (1 - means) / (a0.unsqueeze(-1) + 1.0)      # (B, K, 3)
    # mixture means
    E    = (pi.unsqueeze(-1) * means).sum(1)                    # (B, 3)
    # mixture second moments (raw): E[Z^2] per component
    EZ2k = var + means ** 2                                     # (B, K, 3)
    EZ2  = (pi.unsqueeze(-1) * EZ2k).sum(1)                     # (B, 3)
    Var  = EZ2 - E ** 2                                         # (B, 3)
    # covariance E[Z1 Z2]
    EZ12k = (-alpha[..., 0] * alpha[..., 1]
             / (a0 ** 2 * (a0 + 1))) + means[..., 0] * means[..., 1]
    EZ12  = (pi * EZ12k).sum(1)                                 # (B,)
    Cov12 = EZ12 - E[..., 0] * E[..., 1]                        # (B,)

    pred = torch.stack([E[..., 0], Var[..., 0],
                        E[..., 1], Var[..., 1], Cov12], dim=-1)
    return ((pred - m_in) ** 2).mean()


def entropy_reg(pi):
    return -(pi * torch.log(pi + 1e-12)).sum(-1).mean()
```

### 5.3 Training loop

```python
def train(model, train_loader, val_loader, epochs=200,
          lr=3e-4, lam_m=0.5, lam_e=1e-3, device="cuda"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state = float("inf"), None

    for ep in range(epochs):
        model.train()
        for m, samples in train_loader:
            m, samples = m.to(device), samples.to(device)
            pi, alpha = model(m)
            loss_nll = mixture_nll(pi, alpha, samples)
            loss_mom = moment_loss(pi, alpha, m)
            loss_ent = -lam_e * entropy_reg(pi)   # maximize entropy
            loss = loss_nll + lam_m * loss_mom + loss_ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        val_nll = evaluate(model, val_loader, device)
        if val_nll < best_val:
            best_val, best_state = val_nll, {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()}
        print(f"epoch {ep:3d}  val_nll {val_nll:.4f}  best {best_val:.4f}")

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for m, samples in loader:
        m, samples = m.to(device), samples.to(device)
        pi, alpha = model(m)
        nll = mixture_nll(pi, alpha, samples).item()
        tot += nll * m.size(0); n += m.size(0)
    return tot / n
```

### 5.4 Validation metrics

Track all of these — each catches a different failure mode:

| Metric | What it tells you | Computed on |
|---|---|---|
| NLL on held-out samples | Overall density fit | val/test samples |
| Moment recovery error | Self-consistency of mixture w.r.t. inputs | analytical, no sampling |
| Marginal-PDF JSD | Marginal $P(Z_1)$ vs DNS marginal | binned for diagnostic |
| Joint PDF L1 (eq. 1 of your paper) | Apples-to-apples vs analytical models | binned |
| Joint PDF JSD | Symmetric divergence | binned |
| Conditional reaction rate $\overline{\dot \omega}$ vs DNS | The thing LES actually needs | binned, integrated |

The last one is what your reviewers will care about — the JSD/L1 metrics are easier to game by overfitting to the binning.

### 5.5 Test-time evaluation protocol

1. Held-out configurations: report all six metrics, broken down by case type (isotropic / layered / premixed) and by mixing time $t/\tau_{\text{eddy}}$.
2. Compare against:
   - Analytical baselines (Dirichlet, CM, BVB5, SMLD) from Perry & Mueller
   - Your prior pixel-wise DNN (DNN-5)
   - The new Dirichlet MDN
3. **Ablation table** — varying $K$, with/without moment loss, with/without permutation augmentation, with/without stratified sampling.
4. **Out-of-distribution stress test** — train on isotropic + premixed, test on layered. The data-driven model should not be merely interpolating between known cases.
5. **A posteriori** check: drop the model into an LES of one of the Perry & Mueller configurations, compare statistics vs DNS.

### 5.6 Hyperparameter notes

| Parameter | Suggested range | Notes |
|---|---|---|
| $K$ (mixture components) | 6–10 | Start at 8; ablate to 1, 4, 16 |
| Hidden width | 128–256 | Network is shallow; trunk depth 3 is enough |
| Sample count per box | 2048–8192 | More is better for NLL variance, but I/O dominates |
| $\lambda_m$ (moment loss) | 0.1–1.0 | Tune so the two losses are within 10× at convergence |
| $\lambda_e$ (entropy reg) | $10^{-3}$–$10^{-2}$ | Prevents component collapse |
| Batch size | 64–256 | NLL is well-behaved; can go large |
| Learning rate | 1–3 × $10^{-4}$ | AdamW + cosine schedule |
| Epochs | 100–300 | Early-stop on validation NLL |

## 6. Novelty

Three layers, framed honestly:

1. **MDN methodology** — Bishop (1994) established the framework; Sadowski & Baldi (2019), Tsuchida et al. (2019), and Wu et al. (2024) extended it to Dirichlet outputs and Dirichlet mixtures. Not novel as an ML technique.

2. **Application to multi-scalar subgrid PDF closure** — no published Dirichlet-mixture MDN for joint scalar PDFs in turbulent combustion. The closest precedent (Bode et al. 2023) uses a *Gaussian* MDN for reaction rates, which is the wrong base distribution for compositions. **This is novel in the combustion-ML literature.**

3. **Scientific framing** — replacing the analytical decision tree of Perry & Mueller (2018) with a single moment-conditioned mixture that subsumes Dirichlet/CM/BVB5/Beta-Delta as limits, validated on DNS that explicitly covers all the configurations the analytical models were designed for. **This is the actual publishable contribution.**

Suggested framing sentence for the paper:

> "Mixture density networks (Bishop 1994) with Dirichlet components on the simplex (Sadowski & Baldi 2019; Tsuchida et al. 2019) have not previously been applied to joint subgrid PDF closure for multiscalar turbulent mixing. We adopt this architecture as a unified data-driven alternative to the analytical bivariate-beta hierarchy of Perry and Mueller (2018), in which the Dirichlet, CM, and BVB5 distributions appear as limiting cases."

## 7. References

### Foundational ML

- Bishop, C. M. (1994). *Mixture Density Networks*. NCRG Technical Report 94/004, Aston University. [PDF](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) · [Semantic Scholar](https://www.semanticscholar.org/paper/Mixture-density-networks-Bishop/4cf3569e045993dfe090749f26a55a768684ab86)
- Sensoy, M., Kaplan, L., Kandemir, M. (2018). *Evidential Deep Learning to Quantify Classification Uncertainty*. NeurIPS. [arXiv:1806.01768](https://arxiv.org/abs/1806.01768)
- Sadowski, P., Baldi, P. (2019). *Neural Network Regression with Beta, Dirichlet, and Dirichlet-Multinomial Outputs*. OpenReview. [BJeRg205Fm](https://openreview.net/forum?id=BJeRg205Fm)
- Tsuchida, R., Mok, S., et al. (2019). *Quantifying Intrinsic Uncertainty in Classification via Deep Dirichlet Mixture Networks*. [arXiv:1906.04450](https://arxiv.org/abs/1906.04450)
- Wu, M., Zhou, B., Zhang, J., et al. (2024). *Improved Evidential Deep Learning via a Mixture of Dirichlet Distributions*. [arXiv:2402.06160](https://arxiv.org/abs/2402.06160)

### Compositional data / logistic-normal alternative

- Bear, J., Billheimer, D. (2016). *A Logistic Normal Mixture Model for Compositional Data Allowing Essential Zeros*. Austrian J. Statistics. [link](https://www.ajs.or.at/index.php/ajs/article/view/vol45-4-1)
- Fang, Z., Subedi, S. (2023). *Clustering microbiome data using mixtures of logistic normal multinomial models*. Scientific Reports. [link](https://www.nature.com/articles/s41598-023-41318-8)

### Combustion / presumed-PDF baselines

- Perry, B. A., Mueller, M. E. (2018). *Joint probability density function models for multiscalar turbulent mixing*. Combustion and Flame 193, 344–362. [link](https://www.sciencedirect.com/science/article/abs/pii/S0010218018301512)
- Henry de Frahan, M. T., Yellapantula, S., King, R., Day, M. S., Grout, R. W. (2019). *Deep learning for presumed probability density function models*. Combustion and Flame 208, 436–450. [link](https://www.sciencedirect.com/science/article/abs/pii/S0010218019303220)
- Yellapantula, S., Perry, B. A., Henry de Frahan, M. T., Mueller, M. E., Grout, R. (2019). *Machine Learning based models for joint PDF shapes for multi-scalar mixing in turbulent flows*. 11th U.S. National Combustion Meeting.
- Bode, M., et al. (2023). *Probabilistic deep learning of turbulent premixed combustion*. [ResearchGate](https://www.researchgate.net/publication/373001968_Probabilistic_deep_learning_of_turbulent_premixed_combustion)
- Yao, S., et al. (2022). *On the Use of Machine Learning for Subgrid Scale Filtered Density Function Modelling in Large Eddy Simulations of Combustion Systems*. Springer chapter. [link](https://link.springer.com/chapter/10.1007/978-3-031-16248-0_8)
- Chen, Z., et al. (2021). *Machine learning assisted modeling of mixing timescale for LES/PDF of high-Karlovitz turbulent premixed combustion*. Combustion and Flame. [OSTI](https://www.osti.gov/pages/biblio/1841532)
- Wu, X., et al. (2025). *Novel Data-Driven PDF Modeling in FGM Method Based on Sparse Turbulent Flame Data*. MDPI Energies. [link](https://www.mdpi.com/1996-1073/18/13/3546)

### Regularization for mixture density networks

- Eigen, D., Ranzato, M., Sutskever, I. (2013). *Learning Factored Representations in a Deep Mixture of Experts*. [arXiv:1312.4314](https://arxiv.org/abs/1312.4314)
- Pereyra, G., et al. (2017). *Regularizing Neural Networks by Penalizing Confident Output Distributions*. [arXiv:1701.06548](https://arxiv.org/abs/1701.06548)
