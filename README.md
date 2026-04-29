# PLMGH: What Matters in PLM-GNN Hybrids for Code Classification and Vulnerability Detection
[![arXiv](https://img.shields.io/badge/arXiv-2604.25599-b31b1b.svg)](https://arxiv.org/abs/2604.25599)

Official repository for the paper:

**PLMGH: What Matters in PLM-GNN Hybrids for Code Classification and Vulnerability Detection** 

## Overview

Code understanding models increasingly rely on two complementary families of methods:

- **Pretrained Language Models (PLMs)**, which capture rich semantic information from code tokens
- **Graph Neural Networks (GNNs)**, which exploit structural information from program representations such as Abstract Syntax Trees (ASTs)

This repository accompanies a controlled empirical study of **PLM→GNN hybrids**, where pretrained code representations are injected into downstream graph models for code classification and vulnerability detection.

The goal of this work is to better understand **what actually matters** in PLM-GNN hybrid pipelines:
- Do hybrids consistently outperform PLM-only and GNN-only baselines?
- What computational costs do they introduce?
- How robust are they under identifier obfuscation?
- Does performance depend more on the PLM feature source or on the GNN architecture?

## Main Idea

This repo's  setting follows a simple and practical pipeline:

1. Parse source code into an **AST**
2. Run a frozen **code-specialized PLM** on the source code
3. Align token-level PLM embeddings to AST nodes
4. Feed the resulting node features into a **GNN**
5. Perform downstream prediction for code classification or vulnerability detection


## Study Scope

We evaluate PLM→GNN hybrids across:
- **Code classification** on **Java250**
- **Vulnerability detection** on **Devign**
- **Out-of-distribution robustness** on Devign with **identifier obfuscation**

### PLMs
- DeepSeek-Coder-1.3B
- StarCoder2-3B
- Qwen2.5-Coder-0.5B

### GNNs
- GCN
- GAT
- Graph Transformer

### Baselines
- GNN-only models
- PLM-only frozen models
- PLM-only finetuned models (when applicable)

## Third-Party Code

Parts of this repository include code copied and adapted from:

- **code_ast** — Cedric Richter
- Source repository: `cedricrupb/code_ast`
- License: MIT
