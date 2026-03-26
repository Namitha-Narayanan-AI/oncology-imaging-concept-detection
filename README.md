# Oncology Imaging Concept Detection

A research-focused repository for building reproducible deep learning workflows in medical imaging, with an initial focus on oncology image analysis and concept detection.

## Project Overview

This repository is intended to support structured experimentation in medical imaging using reproducible deep learning workflows. The initial focus is on establishing a clean project foundation for image classification tasks that can later be extended toward more advanced analysis settings relevant to oncology imaging.

## Objectives

- Build a clean and reproducible AI research workspace
- Develop a baseline image classification pipeline
- Establish a structured workflow for experimentation and result tracking
- Create a foundation that can be extended toward oncology imaging and radiomics-related tasks

## Repository Structure

```text
oncology-imaging-concept-detection/
├── .venv/                  # Local virtual environment
├── configs/                # Configuration files
├── data/                   # Dataset storage
├── docs/                   # Research notes and documentation
├── experiments/            # Experiment metadata and tracking
├── logs/                   # Training and run logs
├── notebooks/              # Exploratory notebooks
├── results/                # Saved outputs and results
├── src/                    # Source code
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Environment Setup

Create and activate a virtual environment, then install the required dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

```

## Status

This repository is under active development