# Curated completed experiment

Original run identifier: `experiment_20260918T190433_115826Z_f3ea25fa`.

These are byte-for-byte curated copies of selected outputs from the completed frozen experiment. Authoritative originals remain in the local results directory unchanged. This folder omits downloaded data, checkpoints, personal interpreter-path metadata, smoke artifacts and supplementary figures.

## Execution and measured outcomes

CPU-only Windows execution on Intel Core i5-1340P; approximately 32 GB RAM. Python 3.11.0, torch 2.9.1+cpu, torchvision 0.24.1+cpu, NumPy 2.4.6, Matplotlib 3.11.2. See the repository's installed-packages.txt for the historical package inventory.

Seed 42; 5,000 training / 1,000 validation / 10,000 official test images, resized to 32 x 32. Both models continued from 10 to 30 total epochs because validation reconstruction MSE improved by 32.85% for AE and 11.77% for LoRAE under the predefined comparison. No test data determined training duration. No experiments were rerun to create this package.

| Metric | AE | LoRAE |
|---|---:|---:|
| Final epoch | 30 | 30 |
| Test reconstruction MSE | 0.0025550804536789654 | 0.014270503836870193 |
| Parameters | 2041057 | 2057441 |
| Training + validation seconds | 241.86617659998592 | 221.42575409993879 |
| 95% effective latent dimension | 29 | 7 |
| 99% effective latent dimension | 38 | 7 |

Final checkpoint nuclear norm of M: 1.158299207687378. Weighted penalty: 0.001158299207687378. These are recorded in verification.json, not metrics.json. Epoch-average nuclear norms in lorae/history.csv are different quantities. Timings exclude checkpoint/CSV writes and final evaluation.

## Contents and interpretation

- config.json and dataset_indices.npz: frozen configuration and exact split indices.
- extension_decision.json and status.json: duration decision and completed status.
- ae/history.csv and lorae/history.csv: epoch reconstruction, objectives and timings.
- evaluation/metrics.json and metrics.csv: final held-out comparison.
- evaluation/*_latent_analysis.npz: covariance spectra and cumulative variance.
- evaluation/M_analysis.npz: initial/final M and their singular values.
- Four selected figures: reconstruction_curves.png, latent_covariance_spectra.png, matched_reconstructions.png and M_singular_values.png, all under evaluation/.
- verification.json: historical verification of the original full output set. It references all original plots/files, including omitted supplementary artifacts; it is not a manifest or fresh verification of this smaller curated folder.

source_sha256.json records experiment-time source, including the README as it existed then. It does not include later documentation-only changes or newly added packaging files. The seven Python files remain unchanged; an updated README need not match its historical hash. Do not rewrite that historical record to match publication documentation.

The results show lower reconstruction error for AE and stronger latent variance concentration for LoRAE in this run. Effective dimension is a variance threshold, not exact rank. Single-seed, small-scale results do not establish universal superiority, statistical significance, convergence or full paper reproduction. Licensing/provenance notes in ../ATTRIBUTION.md also apply to the package context.
