# FIGNN
### EEG Representation Learning for Cross-Subject Workload Understanding and Personalization

<p align="center">
  <b>Frequency-Informed Graph Neural Network</b>
</p>

<p align="center">
  Learning transferable EEG representations and adapting them to individual users.
</p>

---

## Overview

Electroencephalographic (EEG) signals vary significantly between individuals. A model trained on one person may not generalize well to another without additional personalization.

**FIGNN** investigates a graph-based deep learning approach for learning common cognitive workload representations across individuals and adapting them to new users with limited personal information.

The project focuses on understanding how workload-related patterns are represented in EEG and how these representations can be transferred and personalized across subjects.

> **Research status:** The associated research manuscript is currently under preparation/review. Detailed experimental findings are not disclosed in this repository until publication.

---

## Core Idea

Instead of treating EEG as a collection of independent signals, FIGNN represents the brain as a graph:

- EEG electrodes are represented as graph nodes.
- Frequency-based brain activity is used as node information.
- Functional connectivity between electrodes is represented as graph edges.
- A Graph Neural Network learns a compact representation of the brain state.
- The learned representation is studied across individuals and adapted for personalization.

The broader objective is to understand population-level workload patterns while retaining the ability to personalize them for individual users.

---

## Key Features

- Cross-subject EEG representation learning
- Graph-based modeling of brain connectivity
- Frequency-informed EEG analysis
- Cognitive workload representation
- Population-level learning
- Subject-specific personalization
- Leave-one-subject-out evaluation
- Deep learning with PyTorch and PyTorch Geometric

---

## Technology Stack

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-Graph%20Learning-3C78A9)
![MNE](https://img.shields.io/badge/MNE-EEG%20Processing-6A5ACD)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-Scientific%20Computing-8CAAE6)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)

</p>

| Area | Technologies |
|------|--------------|
| Programming | Python |
| Deep learning | PyTorch |
| Graph learning | PyTorch Geometric |
| EEG processing | MNE-Python |
| Signal analysis | NumPy, SciPy |
| Visualization | Matplotlib |
| Evaluation | Subject-independent EEG experiments |

---

## Dataset

The project uses the publicly available **COG-BCI dataset**, which contains EEG recordings collected during multiple cognitive tasks.

The current research focuses mainly on:

- N-back cognitive workload conditions
- Resting-state EEG recordings
- Cross-subject representation learning
- Subject-specific personalization

Please refer to the original dataset publication for access and usage information.

### Dataset Reference

Hinss et al.,  
**“Open multi-session and multi-task EEG cognitive dataset for passive brain-computer interface applications.”**  
*Scientific Data*, 2023.

---

## Research Workflow

```text
EEG Recordings
      ↓
EEG Preprocessing
      ↓
Frequency-Based Brain Features
      ↓
Brain Connectivity Graphs
      ↓
Frequency-Informed Graph Neural Network
      ↓
Latent EEG Representation
      ↓
Cross-Subject Evaluation
      ↓
Subject-Specific Personalization
