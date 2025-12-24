# Unsupervised Image Segmentation and Denoising

A comparative study of two unsupervised image segmentation methods: EAGLE (Eigen Aggregation Learning) and TokenCut (Self-Supervised Transformers for Object Discovery).

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Installation](#installation)
- [Datasets](#datasets)
- [Methodology](#methodology)

## Overview

This repository contains implementations and evaluations of two distinct approaches to unsupervised semantic segmentation.

**EAGLE (Eigen Aggregation Learning)**
- Learned, self-supervised method
- Combines spectral clustering (EiCue) with contrastive learning (ObjNCELoss)
- Requires training on COCO-Stuff-27
- Inference time: ~0.13s per image

**TokenCut**
- Training-free, graph-based method
- Uses Normalized Cuts on DINO features
- No training required
- Inference time: ~1.5s per image

## Results

### Binary Segmentation on COCO-Stuff-27

| Model | mIoU | Inference Time | Training |
|-------|------|----------------|----------|
| EAGLE | 62.03% | 0.13s | Required |
| TokenCut | 31.75% | 1.50s | None |

### EAGLE Performance

| Probe Type | mIoU | Pixel Accuracy |
|------------|------|----------------|
| Linear Probe | 41.19% | 68.65% |
| Cluster Probe | 24.40% | 50.79% |
| Binary (Thing/Stuff) | 62.03% | - |

### TokenCut Performance

| Dataset | mIoU |
|---------|------|
| COCO-Stuff-27 | 31.75% |
| ECSSD | 42.82% |
| CUB-200-2011 | 32.97% |

## Installation

### Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- PyTorch 1.9+

## Datasets

**COCO-Stuff-27**
- 27 super-categories (15 "Thing" + 12 "Stuff")
- 5,000 validation images
- Primary benchmark for binary segmentation

**ECSSD**
- 1,000 salient object images
- Used for TokenCut evaluation

**CUB-200-2011**
- 11,788 bird images across 200 species
- Fine-grained segmentation benchmark

All datasets are downloaded automatically via the notebooks.

## Methodology

### EAGLE

**EiCue (Spectral Initialization)**

Constructs a Laplacian matrix combining color and semantic affinities:

```
A = A_color + A_seg
```

Performs spectral decomposition and clusters eigenvectors rather than raw features, producing coherent object masks.

**ObjNCELoss (Contrastive Learning)**

```
L_ObjNCE = -Σ log(exp(f_i · c_yi / τ) / Σ_j exp(f_i · c_j / τ))
```

where:
- f_i: pixel feature vector
- c_yi: prototype of assigned class
- τ: temperature parameter

This loss pulls pixels toward their object prototype while pushing them away from other object centers.

### TokenCut

**Feature Extraction**

Uses DINO ViT-S/16 pre-trained features to capture semantic relationships between image patches.

**Normalized Cut**

Solves the graph partitioning problem:

```
Ncut(A,B) = cut(A,B)/assoc(A,V) + cut(A,B)/assoc(B,V)
```

Relaxed to eigenvalue problem:

```
L_sym z = λz
```

The second smallest eigenvector (Fiedler vector) provides the optimal bipartition.


## Key Findings

1. EAGLE achieves 30% higher mIoU than TokenCut on COCO-Stuff-27
2. Spectral initialization (EiCue) is critical for generating coherent masks
3. Contrastive learning refines features for semantic consistency
4. TokenCut performs better on simple, salient objects (ECSSD: 42.82% mIoU)
5. EAGLE is approximately 10x faster at inference


