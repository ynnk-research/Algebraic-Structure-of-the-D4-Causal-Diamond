# Algebraic Structure of the D4 Causal Diamond
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19343721.svg)](https://doi.org/10.5281/zenodo.19343721) 
[![Code License: Apache 2.0](https://img.shields.io/badge/Code_License-Apache_2.0-blue.svg)](LICENSE)
[![Doc License: CC BY 4.0](https://img.shields.io/badge/Doc_License-CC_BY_4.0-green.svg)](LICENSE-CC-BY.txt)

**Subtitle:** Spectrum, Symmetry, Codes, and Gravitational Phenomenology from a Single Lorentzian Lattice  
**Author:** Yannick Schmitt  
**Date:** March 2026  
**Status:** Preprint 1.0.0


## Overview

This paper synthesises and extends the geometric foundations of the D4 causal diamond — the 2-complex built from the twelve lightlike nearest-neighbour vectors of the ternary Minkowski lattice `{-1, 0, +1}^4` under `η = diag(-1, +1, +1, +1)` — into five mutually reinforcing algebraic and phenomenological results. Every quantity derived in the paper follows from a single combinatorial input: the plaquette Laplacian `K = MM^T` with spectrum `{0^4, 6^2, 8^3, 10^2, 28^1}` and the root count `|Φ(D4)| = 24`. No free parameter is introduced at any stage.

The five main results are:

1. **Symmetry group** (Theorem 3.2): The stabiliser of the twelve links under Minkowski isometries is a group `G` of order 96. The eigenspace decomposition of `K` into `G`-irreducibles is determined exactly: `V_8` uniquely carries the time-reversal-odd representation `ρ_pvec` (angular momentum), and `V_0` carries the time-reversal-even `ρ_vec` (linear momentum).

2. **CSS quantum error-correcting codes** (Theorem 4.3): The GF(2) rank gap `rank_R(M) - rank_{F2}(M) = 1`, forced by the Lorentzian temporal charge `n^0_eff = 12`, generates a complete family of four CSS codes: `[[12, 1, (4,3)]]`, `[[12, 4, (4,2)]]`, and their Wick-rotated duals.

3. **Boundary coupling derivation** (Theorem 5.3): Every plaquette bivector is automatically Plebanski-simple (`Pf = 0`), ruling out simplicity-defect coupling. The boundary eigenvalue `λ_s = 2/13` follows uniquely from the SU(2) `j = 1` Casimir distributed over `N_eff + 1 = 13` entities.

4. **Mass renormalisation fixed point** (Theorem 6.4): The D4 root count and boundary denominator determine an exact fixed point `γ* = 24/13` at which the Lindblad decoherence mass and the geometric inertial mass coincide. The algebraic identities `λ_s · m* = 1/6` and `λ_s · N_eff = γ*` hold to machine precision.

5. **MOND interpolation** (Theorem 8.2, 8.4): The BF partition function produces the interpolation function `ν_BF(x) = 1/r(β_c √x)`, where `r(β) = I_1(β)/I_0(β)` and `β_c = 2.7364`. Tested against 1617 data points from the McGaugh et al. (2016) Radial Acceleration Relation dataset, `ν_BF` matches the empirical McGaugh function to within 0.001 dex in shape, with the sole discrepancy being the geometrically predicted deep-MOND prefactor `2/β_c = 0.731`.

A spectral rigidity theorem (Theorem 9.2) confirms that no perturbation — removal of any single plaquette or replacement of any link — simultaneously preserves all four defining constraints, establishing the D4 causal diamond as the unique object satisfying them.


## Repository Structure
* `/paper` - LaTeX source files and PDF pre-print of the manuscript.
* `/script` - Verification script


## Verification Script

All numerical claims in the paper are verified by `verification_AS_D4_CD_paper.py`. The geometry is re-bootstrapped from axiomatic first principles on every run — no matrix is hardcoded — and 78 independent checks are performed across all eight paper sections. Every check prints `PASS` or `FAIL`; a failure includes an upstream pointer identifying which input has drifted.

### What the script verifies

| Section | Content |
|---|---|
| Sec 2 — Causal Diamond Geometry | Lightlike enumeration, plaquette count, Laplacian spectrum, boundary vector |
| Sec 3 — Symmetry Group | Group order 96, conjugacy classes, `K`-commutativity, eigenspace `G`-irreducible decomposition |
| Sec 4 — CSS Codes & GF(2) Rank Gap | Rank gap = 1; all four code parameters `[[12,4,(4,2)]]`, `[[12,1,(4,3)]]`, duals |
| Sec 5 — Boundary Coupling | Universal Plebanski simplicity; SU(2) Casimir derivation of `λ_s = 2/13`; `K_total` spectrum |
| Sec 6 — Mass Fixed Point | Inertial mass `m_I = 13/12`; Lindblad solution; fixed point `γ* = 24/13`; algebraic identities |
| Sec 7 — Entanglement Entropy | Future/past bipartition entropies; area law slope `α = 0.266 nats/link` |
| Sec 8 — MOND Interpolation | `ν_BF` vs McGaugh on 1617 RAR data points; shape residual 0.001 dex; `χ²/N = 9.63` |
| Sec 9 — Spectral Rigidity | 21 plaquette removals + 1 non-lightlike link; 0/22 perturbations preserve `C1∧C4` |

### Requirements

```
numpy
scipy
```

### Running the script

```bash
python verification_AS_D4_CD_paper.py
```

The script has a fast mode for a quick smoke-test (~15 seconds) and a full mode for complete paper-value verification (~90 seconds, dominated by group closure, code search, and the rigidity scan). To enable fast mode, set `FAST = True` near the top of the script.

## Key Numerical Constants

| Quantity | Symbol | Value |
|---|---|---|
| Lightlike link count | `N_eff` | 12 |
| Plaquette count | `Q` | 21 |
| Boundary denominator | `N_bd` | 13 |
| D4 root count | `\|Φ(D4)\|` | 24 |
| Boundary coupling | `λ_s` | 2/13 ≈ 0.1538 |
| BF crossover | `β_c` | 2.7364 |
| Mass fixed point | `γ*` | 24/13 ≈ 1.8462 |
| Inertial mass | `m_I` | 13/12 ≈ 1.0833 |
| Deep-MOND prefactor | `2/β_c` | 0.7309 |
| Symmetry group order | `\|G\|` | 96 |
| MOND fit `χ²/N` (rescaled BF) | — | 9.63 |
| MOND fit `χ²/N` (McGaugh) | — | 9.79 |

## Related Papers

- **Paper 1**: *Exact Discretisation and Boundary Observables in Lorentzian Causal Diamonds* — establishes the geometry, Laplacian spectrum, and BF crossover. Yannick Schmitt. (2026). Zenodo. https://doi.org/10.5281/zenodo.19338306
- **Paper 3**: *A Lorentzian CSS Duality in Causal Diamond Quantum Error-Correcting Codes* — derives the four-code CSS family in full, including No-Go theorems and circuit-level thresholds. Yannick Schmitt. (2026). Zenodo. https://doi.org/10.5281/zenodo.19343889

## Citation

If you use this work, please cite it as:

> Yannick Schmitt. (2026). Algebraic Structure of the D4 Causal Diamond. Zenodo. https://doi.org/10.5281/zenodo.19343721

## License
 * The source code in this repository is licensed under the [Apache License 2.0](LICENSE).
 * The documentation, LaTeX source files, and PDF papers are licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE-CC-BY.txt).
