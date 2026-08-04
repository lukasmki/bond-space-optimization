# Z-vector (adjoint) bond-order gradient

Equations behind `bo_flux_gradient()` in [bond.py](bond.py), used by
`BondFluxCalculator(..., zvector=True)`.

## Notation

| symbol | meaning |
| --- | --- |
| $S$ | AO overlap, $S_{\mu\nu}=\langle\mu\mid\nu\rangle$ |
| $P^\alpha,P^\beta$ | spin density matrices |
| $D=P^\alpha+P^\beta$, $R=P^\alpha-P^\beta$ | total / spin density |
| $C^\sigma$, $C^\sigma_{\rm occ}$ | MO coefficients, occupied block |
| $i,j$ / $a,b$ / $p,q$ | occupied / virtual / any MO |
| $\varepsilon^\sigma_p$ | orbital energies |
| $A,B$ | atom indices; $\mu\in A$ means AO $\mu$ sits on atom $A$ |
| $X^{1,k}$ | $\partial X/\partial R_k$, a perturbation by nuclear coordinate $k$ |

The Mayer bond order, with $\widetilde D = DS$ and $\widetilde R = RS$:

$$
B_{AB}=\sum_{\mu\in A}\sum_{\nu\in B}
\left[\widetilde D_{\mu\nu}\widetilde D_{\nu\mu}
     +\widetilde R_{\mu\nu}\widetilde R_{\nu\mu}\right]
$$

## The problem

The bond-flux objective is the scalar

$$
E=\tfrac12\sum_{AB} m_{AB}\left(B_{AB}-B^{\rm target}_{AB}\right)^2 ,
\qquad
G_{AB}\equiv\frac{\partial E}{\partial B_{AB}}=m_{AB}\left(B_{AB}-B^{\rm target}_{AB}\right)
$$

with mask $m_{AB}$ selecting the constrained pairs. The direct route builds
every $\partial B_{AB}/\partial R_k$ and then contracts,

$$
\frac{\partial E}{\partial R_k}=\sum_{AB}G_{AB}\frac{\partial B_{AB}}{\partial R_k},
$$

which needs the density response $\partial P^\sigma/\partial R_k$ for **every**
$k$ — one CPHF solve per atom. Because $E$ is a single scalar, contracting
*before* solving reduces this to one solve in total.

## 1. Differentiate in the AO basis

Broadcast $G$ to AO pairs, $g_{\mu\nu}=G_{A(\mu)B(\nu)}$ (symmetric), and set

$$
Y_D = 2\,g\circ\widetilde D^{\mathsf T},\qquad
Y_R = 2\,g\circ\widetilde R^{\mathsf T}
$$

($\circ$ is elementwise). The factor 2 collects the two equal terms from
differentiating the quadratic $\widetilde D_{\mu\nu}\widetilde D_{\nu\mu}$.
Applying the chain rule through $\widetilde D=DS$, $\widetilde R=RS$ gives the
total differential

$$
dE=\sum_\sigma \operatorname{Tr}\!\left[(W^\sigma)^{\mathsf T}dP^\sigma\right]
   +\operatorname{Tr}\!\left[(W^S)^{\mathsf T}dS\right]
$$

$$
W_D=Y_DS,\quad W_R=Y_RS,\qquad
W^\alpha=W_D+W_R,\quad W^\beta=W_D-W_R,\qquad
W^S=DY_D+RY_R
$$

So $W^\sigma=\partial E/\partial P^\sigma$ at fixed $S$, and
$W^S=\partial E/\partial S$ at fixed $P$.

## 2. The explicit overlap term

$W^S$ contracts with derivative integrals only — no response, no CPHF:

$$
\frac{\partial S_{\mu\nu}}{\partial R_A}
=-\langle\nabla\mu|\nu\rangle\,[\mu\in A]-\langle\nabla\nu|\mu\rangle\,[\nu\in A]
\;\Longrightarrow\;
\sum_{\mu\nu}W^S_{\mu\nu}\frac{\partial S_{\mu\nu}}{\partial R_A}
=-\!\!\sum_{\mu\in A,\;\nu}\left(W^S+W^{S\mathsf T}\right)_{\mu\nu}\langle\nabla\mu|\nu\rangle
$$

Both terms read the *same* `int1e_ipovlp` slice; only the index written is
transposed. Getting this wrong makes $\partial S/\partial R$ asymmetric and
breaks translational invariance.

## 3. Project the density response onto orbital rotations

With $\partial P^\sigma_{\mu\nu}/\partial R_k=\sum_{pi}\left[C_{\mu p}U^{k}_{pi}C_{\nu i}+C_{\nu p}U^{k}_{pi}C_{\mu i}\right]$,

$$
\sum_{\mu\nu}W^\sigma_{\mu\nu}\frac{\partial P^\sigma_{\mu\nu}}{\partial R_k}
=\sum_{pi}L^\sigma_{pi}U^{k,\sigma}_{pi},
\qquad
L^\sigma=(C^\sigma)^{\mathsf T}\left(W^\sigma+W^{\sigma\mathsf T}\right)C^\sigma_{\rm occ}
$$

$L$ is the whole objective compressed into one $n_{\rm mo}\times n_{\rm occ}$
matrix — the reason a single solve suffices.

## 4. Split $U$, then swap with the adjoint

Occupied-occupied rotations are fixed by orthonormality, not by the CPHF:

$$
U^{k}_{ji}=-\tfrac12 S^{1,k}_{ji}
$$

The virtual-occupied block obeys the CPHF equation (PySCF sign convention,
$V$ = the response/kernel operator)

$$
(\varepsilon_a-\varepsilon_i)U^{k}_{ai}+V_{ai}(U^{k})
=-\left(h^{1,k}_{ai}-\varepsilon_i S^{1,k}_{ai}\right)
$$

Separating $V(U)=V(U_{vo})+V(U_{oo})$ defines the operator $\mathbf A$ and a
per-atom right-hand side:

$$
(\mathbf A\,U^{k}_{vo})_{ai}\equiv(\varepsilon_a-\varepsilon_i)U^{k}_{ai}+V_{ai}(U^{k}_{vo})
=\underbrace{-\left(h^{1,k}_{ai}-\varepsilon_i S^{1,k}_{ai}\right)-V_{ai}(U^{k}_{oo})}_{\textstyle \mathrm{RHS}^{k}}
$$

$\mathbf A$ is symmetric, so $\mathbf A^{-1}$ may be moved off $\mathrm{RHS}^k$
and onto $L$ — the adjoint step:

$$
\sum_{ai}L_{ai}U^{k}_{ai}
=\sum_{ai}L_{ai}\left(\mathbf A^{-1}\mathrm{RHS}^{k}\right)_{ai}
=\sum_{ai}Z_{ai}\,\mathrm{RHS}^{k}_{ai},
\qquad\boxed{\mathbf A\,Z=L_{vo}}
$$

$Z$ is independent of $k$: **one** solve for all $3N$ components. In the form
passed to `lib.krylov`, which solves $(1+a)x=b$,

$$
Z_{ai}+\frac{V_{ai}(Z)}{\varepsilon_a-\varepsilon_i}
=\frac{L_{ai}}{\varepsilon_a-\varepsilon_i}
$$

Symmetry of $V$ also removes the per-atom cost of the $U_{oo}$ coupling, since
$V(Z)$ can be formed once:

$$
\sum_{ai}Z_{ai}V_{ai}(U^{k}_{oo})=\sum_{ji}U^{k}_{oo,ji}V_{ji}(Z)
=-\tfrac12\sum_{ji}S^{1,k}_{ji}V_{ji}(Z)
$$

## 5. Assemble

$$
\frac{\partial E}{\partial R_k}=
\sum_\sigma\left[
-\tfrac12\sum_{ji}S^{1,k,\sigma}_{ji}\left(L^\sigma_{ji}-V_{ji}(Z^\sigma)\right)
-\sum_{ai}Z^\sigma_{ai}\left(h^{1,k,\sigma}_{ai}-\varepsilon^\sigma_i S^{1,k,\sigma}_{ai}\right)
\right]
\;-\!\!\sum_{\mu\in k,\;\nu}\left(W^S+W^{S\mathsf T}\right)_{\mu\nu}\langle\nabla\mu|\nu\rangle
$$

The force is $-\partial E/\partial R_k$.

## Cost

| | CPHF solves | also needed |
| --- | --- | --- |
| direct (`bo_gradient`) | $3N$ | `make_h1` |
| adjoint (`bo_flux_gradient`) | $1$ | `make_h1`, one extra $V(Z)$ |

`make_h1` survives because $\mathrm{RHS}^k$ still needs the perturbed Fock
matrix of every atom; it sets the floor on the achievable speedup. Measured
2.1x at 4 atoms and 2.3x at 8, improving with size.

The adjoint is exact, so it does not trade accuracy for speed the way
restricting `atomlist` does — it returns forces on every atom and makes that
approximation unnecessary. Agreement with the direct path is ~1e-6, set by the
krylov tolerance, and the direct path itself is checked against finite
differences of $\partial B_{AB}/\partial R_k$.

## Code map

| equation | variable in `bo_flux_gradient` |
| --- | --- |
| $g$, $Y_D$, $Y_R$ | `g`, `YD`, `YR` |
| $W^\alpha,W^\beta$, $W^S$ | `W[0]`, `W[1]`, `WS` |
| $L^\sigma$ | `Lmo` |
| $1/(\varepsilon_a-\varepsilon_i)$ | `eai` |
| $Z$, $V(Z)$ | `Z`, `VZ` |
| $h^{1,k}$, $S^{1,k}$ | `h1`, `s1` (from `make_h1`, `int1e_ipovlp`) |
