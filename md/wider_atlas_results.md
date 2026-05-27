# Wider ATLAS demo — bundle robustness across folds

Each row is the chimerax-vampnet bundle's full analysis of one
public MD trajectory from ATLAS (Vander Meersche et al. 2024) —
no MD generation, no manual prep. The fetcher + bundle handle
download, parse, feature, fit, and timescale extraction.

| PDB | Protein | L | α% | β% | Fold | States populations (%) | Slowest IT (ns) | wall (s) |
|---|---|---:|---:|---:|---|---|---:|---:|
| `1ail_A` | Non-structural protein 1 | 73 | 82 | 0 | all-alpha | 14/54/20/12 | 37.8, 30.2, 22.4 | 41.4 |
| `2ppp_A` | Peptidyl-prolyl cis-trans isomerase FKBP1A | 107 | 13 | 38 | mostly-beta | 28/48/15/9 | 74.5, 35.7, 21.0 | 46.5 |
| `1k5n_A` | HLA class I histocompatibility antigen, B alpha chain | 276 | 28 | 38 | mixed-alpha/beta | 19/16/33/32 | 60.6, 42.3 | 55.1 |
| `4dja_A` | (6-4) photolyase | 518 | 53 | 6 | large-mostly-alpha | 32/22/26/19 | 56.5, 46.7 | 100.0 |

Raw results: [
  {
    "pdb_chain": "1ail_A",
    "protein": "Non-structural protein 1",
    "length": 73,
    "alpha_pct": 82,
    "beta_pct": 0,
    "rg_A": 13.6,
    "rmsf_A": 1.42,
    "populations_pct": [
      13.9,
      53.9,
      20.1,
      12.1
    ],
    "implied_timescales_ns": [
      37.8,
      30.2,
      22.4
    ],
    "elapsed_s": 41.4,
    "expected_fold": "all-alpha"
  },
  {
    "pdb_chain": "2ppp_A",
    "protein": "Peptidyl-prolyl cis-trans isomerase FKBP1A",
    "length": 107,
    "alpha_pct": 13,
    "beta_pct": 38,
    "rg_A": 13.44,
    "rmsf_A": 0.72,
    "populations_pct": [
      28.2,
      47.9,
      15.1,
      8.9
    ],
    "implied_timescales_ns": [
      74.5,
      35.7,
      21.0
    ],
    "elapsed_s": 46.5,
    "expected_fold": "mostly-beta"
  },
  {
    "pdb_chain": "1k5n_A",
    "protein": "HLA class I histocompatibility antigen, B alpha chain",
    "length": 276,
    "alpha_pct": 28,
    "beta_pct": 38,
    "rg_A": 25.17,
    "rmsf_A": 3.12,
    "populations_pct": [
      18.9,
      15.8,
      33.3,
      32.0
    ],
    "implied_timescales_ns": [
      60.6,
      42.3
    ],
    "elapsed_s": 55.1,
    "expected_fold": "mixed-alpha/beta"
  },
  {
    "pdb_chain": "4dja_A",
    "protein": "(6-4) photolyase",
    "length": 518,
    "alpha_pct": 53,
    "beta_pct": 6,
    "rg_A": 24.42,
    "rmsf_A": 1.05,
    "populations_pct": [
      32.5,
      22.5,
      26.1,
      18.9
    ],
    "implied_timescales_ns": [
      56.5,
      46.7
    ],
    "elapsed_s": 100.0,
    "expected_fold": "large-mostly-alpha"
  }
]