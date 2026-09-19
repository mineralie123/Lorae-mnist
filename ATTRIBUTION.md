# Attribution and provenance

This is an AI-generated educational implementation produced with Codex for CISC3024 AI Assignment #1. The original paper and official implementation were consulted. It is not the authors' official code and is not a fully independent implementation from the paper alone. Similarity review cannot establish the origin of every generated line.

## Original research

Alokendu Mazumder, Tirthajit Baruah, Bhartendu Kumar, Rishab Sharma, Vishwajeet Pattanaik and Punit Rathore. Learning Low-Rank Latent Spaces With Simple Deterministic Autoencoder: Theoretical and Empirical Insights. IEEE/CVF WACV 2024; arXiv:2310.16194 (2023 preprint).

Paper: https://arxiv.org/abs/2310.16194

The LoRAE research contribution, including the nuclear-norm-regularized latent transformation, belongs to the paper authors; it is not an assignment invention.

## Official implementation

Repository: https://github.com/tirthajit/LoRAE_WACV24

Inspected revision: `89f38e5f3b02302328bcafd1f587b77cac68f81e`. This is the provenance-review reference, not proof of the precise revision originally consulted during implementation.

The upstream LICENSE states Apache License 2.0 and includes `Copyright 2024 Tirthajit Baruah`. Its exact text is preserved in [third_party_licenses/LoRAE-Apache-2.0.txt](third_party_licenses/LoRAE-Apache-2.0.txt). No separate upstream root NOTICE file was found at that revision.

## Acknowledged earlier code lineage

LoRAE's README acknowledges code adopted from https://github.com/facebookresearch/irmae.

IRMAE reference: Li Jing, Jure Zbontar and Yann LeCun, Implicit Rank-Minimizing Autoencoder (2020), https://arxiv.org/abs/2010.00679.

Inspected IRMAE revision: `41c6c28985a05e8a549b8b0879d697bd7bd964b2`. Its model.py carries: `Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved`.

IRMAE's license is Creative Commons Attribution-NonCommercial 4.0 International; the exact text is preserved in [third_party_licenses/IRMAE-CC-BY-NC-4.0.txt](third_party_licenses/IRMAE-CC-BY-NC-4.0.txt). Its MNIST encoder matches the corresponding LoRAE section and its decoder is nearly identical, with an additional VAE option.

## Assignment implementation mapping

| Assignment section | Upstream reference | Relationship / changes |
|---|---|---|
| models.py Encoder and Decoder | LoRAE model.py MNIST_Encoder / MNIST_Decoder; earlier IRMAE model.py | Close structural adaptation of layer specifications. Fixed dimension 128, loop-built encoder, renamed members, flatten/reshape operations and non-in-place ReLU. Decoder includes identical convolution constructor expressions. |
| models.py LoRAE and loss_terms | LoRAE model.py MLP / AE; train.py nuclear penalty | Same algorithmic mechanism, rewritten interface: explicit single M, reconstruction-returning forward and separately returned objective components. |
| models.py StandardAE / make_models | LoRAE AE baseline concept | Explicit baseline without M/penalty; assignment-specific matched, unshared initialization. |
| data.py | LoRAE train.py MNIST preprocessing | Same resize/ToTensor approach using framework APIs; assignment-specific deterministic subset, isolated validation, explicit interpolation and seeded loaders. |
| evaluation.py covariance analysis | LoRAE singular.py | Same covariance/SVD diagnostic concept; rewritten collection and plotting, additional effective dimensions, cumulative variance and numerical safeguards. |
| evaluation.py interpolation | LoRAE generation.py interpolation branch | Same latent interpolation concept; assignment-specific fixed digit pairs and endpoint-inclusive vectorized grid. |
| config.py, training.py, run.py | Method settings and generic training concepts | Assignment-specific frozen configuration, paired CPU runs, separate logs, RNG/checkpoint continuation, extension rule, timing, manifests and source hashes. Ordinary optimizer operations are generic framework code. |
| smoke_test.py and remaining evaluation diagnostics | No close upstream test counterpart identified | Assignment-specific split, shape, gradient, continuation and plotting checks; matched figures and structured metrics. |

No substantial verbatim multi-line implementation block was identified in the seven files. This does not prove independent authorship or eliminate adaptation concerns. The strongest similarity is the MNIST model structure and decoder expressions. No upstream files are bundled wholesale as experiment source.

## Licensing scope remains unresolved

The two license texts document upstream terms; their inclusion does not grant a new license, remove inherited conditions, or establish an Apache-only licensing chain. The inspected public material does not establish whether separate permission covers the IRMAE-derived portions. We therefore do not assert a blanket repository-wide license or unrestricted commercial reuse permission.

Before publication/relicensing, clarify the applicable scope for adapted model material and preserve all applicable notices and modification disclosures. This document records the adaptation and AI role; it does not claim ownership over the original research or upstream code. Assignment-specific code and experimental outputs are distinguished from upstream material, but no separate blanket license has been selected for them.
