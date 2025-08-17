# **A General Framework for “Free-Lunch” Utility in Differentially-Private SGD**

---

## **Abstract**
State the utility–privacy dilemma of user-level
DP-SGD, introduce the three-component framework
(curvature-aware noise, optimiser plug-ins, post-hoc influence
calibration), and preview accuracy / AUC gains on several datasets.

---

## **1 Introduction**
### 1.1 Motivation  
Explain clipping/noise trade-off, slice-specific degradation, and why
existing fixes are piecemeal.

### 1.2 Our Question  
Can we **recover lost utility “for free’’**—i.e.\ without increasing the
\((\varepsilon,\delta)\) budget—by coordinating curvature, optimisation,
and data influence?

### 1.3 Contributions  
1. **General Framework** unifying noise shaping, DP-aware optimisation,
   and influence re-weighting.  
2. **Curvature-Aware DP-SGD** based on a low-rank Fisher metric with
   variance \(\sigma^{2}\Delta_{2}^{2}\lambda_j^{-1}\) along each
   top-\(k\) eigen-direction.  
3. **Influence-Function Calibration**: iterative, line-search variant
   that respects post-processing DP.  
4. **Seamless Integrations**: optimisers such as DP-SAT plug in at Stage 2
   with no DP re-proof.  
5. **Comprehensive Ablations** across \(k\), clip rules, IF budgets, and
   optimiser choices.

---

## **2 Related Work**
*Noise allocation*, *DP-aware optimisers*, *post-hoc utility repair*;
show they address isolated stages, whereas we provide a unifying view.

---

## **3 Unified Framework for Utility-Preserving DP-SGD**
### 3.1 Three Stages, One Budget  
| Stage | DP property | What can vary | Plug-in examples |
|-------|-------------|--------------|------------------|
| **S1 Noise Shaping** | Adds randomness | Metric & covariance | Isotropic, SSD-Diag, **Fisher-LR (ours)** |
| **S2 DP Optimiser** | Post-processing | LR, momentum | DP-SGD, **DP-SAT**, DP-Adam |
| **S3 Post-processing** | Post-processing | Re-weight, fine-tune | **IF calibration (ours)** |

### 3.2 Workflow Diagram *(figure placeholder)*  
### 3.3 Drop-in API *(code snippet placeholder)*  

---

## **4 Curvature-Aware DP-SGD (Stage 1)**
### 4.1 Low-Rank Fisher Metric  
Compute damped Fisher  
\(F\simeq U_k\Lambda_kU_k^\top\) and keep the top \(k\) eigen-pairs.

### 4.2 Mahalanobis Clipping  
Clip per-sample/user gradients to
\(\|g\|_{F^{-1}}\le\Delta_{2}\); calibration ensures the same effective
Euclidean sensitivity as vanilla DP-SGD.

### 4.3 Anisotropic Gaussian Noise  
Inject noise  
\[
\Delta\theta_{\text{priv}}
  = \sigma\,\Delta_{2}\;
    U_k\Lambda_k^{-\tfrac12}z,
  \qquad z\sim\mathcal N(0,I_k).
\]
Variance along eigen-vector \(u_j\) is
\(\sigma^{2}\Delta_{2}^{2}\lambda_j^{-1}\):
*less* noise in sharp directions, *more* in flat ones, while the overall
covariance equals \(\sigma^{2}\Delta_{2}^{2}F^{-1}\).

### 4.4 Implementation Notes  
GPU-friendly eigensolver, adaptive clip via Euclidean quantile, optional
complement noise for full-rank guarantees.

### 4.5 Discussion  
How re-allocating the *same* noise budget can act as adaptive
regularisation and improve generalisation.

---

## **5 Influence-Function Calibration (Stage 3)**
*(unchanged outline: formulation, enhancements, guarantee)*

---

## **6 Experiments**
*(setup, baselines, ablations, results, discussion)*

---

## **7 Conclusion**
Reiterate that smarter noise + optimiser + IF repair yields “free-lunch’’
utility under the same DP budget; outline future directions.

---

## **Appendix**
Proofs, pseudocode, hyper-parameter grids, extended tables.