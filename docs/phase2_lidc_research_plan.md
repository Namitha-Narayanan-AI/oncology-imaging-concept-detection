# Phase 2 LIDC-IDRI Research Plan

## Research Direction

**Robust Multi-Task Pulmonary Nodule Characterisation and Radiologist-Assessed Malignancy-Risk Prediction from 3D CT.**

Phase 2 moves from the Phase 1 chest X-ray baseline to lesion-level thoracic CT analysis using radiologist-annotated 3D pulmonary nodule crops from LIDC-IDRI.

This phase must consistently describe the primary outcome as **radiologist-assessed malignancy-risk** or **malignancy-risk**, not definitive cancer diagnosis. LIDC-IDRI malignancy ratings are subjective radiologist assessments and are not universal pathology-confirmed cancer labels.

## Research Question

Can a 3D CT model jointly predict radiologist-assessed pulmonary nodule malignancy-risk and clinically interpretable nodule attributes from LIDC-IDRI nodule crops while preserving patient-level separation and transparent handling of reader disagreement?

## Hypothesis

A multi-task 3D model trained to predict malignancy-risk together with visual attributes such as spiculation, lobulation, texture, margin, and sphericity will learn more clinically meaningful nodule representations than a single-task malignancy-risk baseline.

The expected benefit is not necessarily higher headline accuracy alone. The desired improvements are better calibrated risk estimates, more stable performance across reader-disagreement strata, and interpretable auxiliary predictions that make model behavior easier to audit.

## Dataset Rationale

LIDC-IDRI is appropriate for Phase 2 because it provides thoracic CT scans with pulmonary nodule annotations from multiple radiologists. It supports lesion-level experiments using 3D CT context and includes radiologist-assessed semantic attributes relevant to nodule characterization.

The dataset is especially suitable for this project because it aligns with the repository's Phase 1 emphasis on clinically meaningful evaluation, error analysis, threshold behavior, and interpretability.

Important caveat: LIDC-IDRI is not a clean pathology-confirmed cancer dataset for every nodule. Malignancy labels should be treated as reader-assessed risk scores, not ground-truth cancer status.

## Targets

### Primary Target

- Radiologist-assessed malignancy-risk score from LIDC-IDRI nodule annotations.

Initial experiments may model this target as:

- ordinal classification over rating levels;
- binary or grouped risk classification for a minimal baseline;
- regression against an aggregated reader score.

The chosen representation must be documented in the experiment config and results.

### Auxiliary Targets

Candidate auxiliary targets:

- spiculation;
- lobulation;
- texture;
- margin;
- sphericity.

Other LIDC-IDRI attributes may be added later if the annotation coverage, label quality, and modeling objective justify them.

## Baseline Models

Initial baselines should be intentionally modest and reproducible:

1. Single-task 3D CNN predicting radiologist-assessed malignancy-risk from fixed-size nodule crops.
2. Single-task 3D CNNs for each selected auxiliary attribute.
3. Optional non-image baseline using simple crop metadata or radiomic-style summary features, if implemented later.

These baselines establish whether the multi-task setup adds value beyond a straightforward 3D lesion classifier.

## Proposed Multi-Task Model

The planned Phase 2 model should use a shared 3D convolutional backbone with task-specific prediction heads:

- one primary head for radiologist-assessed malignancy-risk;
- auxiliary heads for nodule attributes such as spiculation, lobulation, texture, margin, and sphericity.

The training objective should support missing labels through masked losses, because not every annotation aggregation strategy will necessarily provide every target for every nodule.

Initial implementation should prioritize clear data flow, reproducibility, and leakage control before exploring larger architectures.

## Evaluation Metrics

Primary malignancy-risk evaluation may include:

- accuracy for grouped risk labels;
- macro F1 and weighted F1;
- balanced accuracy;
- ROC AUC when binary risk grouping is used;
- mean absolute error or quadratic weighted kappa when ordinal/regression formulations are used;
- calibration metrics or calibration plots where probability estimates are used.

Auxiliary attribute evaluation may include:

- per-attribute accuracy;
- macro F1;
- balanced accuracy;
- confusion matrices;
- missing-label coverage.

Clinical/research reporting should include:

- performance by patient-level split;
- confidence intervals where practical;
- reader-disagreement subgroup analysis;
- threshold behavior selected on validation data only and evaluated once on test data.

## Leakage Controls

Required controls:

- split by patient, not by crop, image, annotation, or radiologist reading;
- keep all nodules from the same patient in the same split;
- keep all annotations for the same nodule in the same split;
- generate crop manifests before training and record split membership explicitly;
- avoid computing normalization statistics from validation or test data;
- select decision thresholds on validation data only;
- run final test evaluation once per frozen model and frozen threshold policy.

Derived artifacts such as crops, cached tensors, features, and visualizations must retain patient ID, series ID, nodule ID, and split metadata.

## Uncertainty And Reader-Disagreement Strategy

LIDC-IDRI includes multiple radiologist assessments per nodule. Reader disagreement is scientifically meaningful and should not be silently erased.

Planned handling:

- store individual reader ratings when available;
- store aggregate targets separately, such as median, mean, or consensus grouping;
- record disagreement summaries, such as rating range or standard deviation;
- evaluate performance separately for low-disagreement and high-disagreement nodules;
- consider soft labels or distributional targets after the first vertical slice;
- avoid presenting high-disagreement cases as if they had a single objective label.

Uncertainty reporting should distinguish model uncertainty from reader disagreement.

## Expected Limitations

Expected limitations include:

- malignancy-risk ratings are subjective radiologist assessments, not universal pathology-confirmed cancer diagnoses;
- LIDC-IDRI annotations vary across readers and nodules;
- fixed-size crops may lose broader anatomical or clinical context;
- small sample sizes may limit reliable subgroup conclusions;
- scanner, reconstruction, and slice-thickness variation may affect learned features;
- model explanations for 3D CT require careful visualization and should not be treated as proof of causal reasoning;
- external generalization will remain unproven until evaluated on an independent cohort.

## Smallest End-To-End Vertical Slice

The first vertical slice should prove the full Phase 2 data path without implementing full model training yet:

1. Select a tiny LIDC-IDRI subset.
2. Inspect one DICOM CT series and its annotation metadata.
3. Build a manifest with patient ID, series ID, nodule ID, reader ratings, target fields, and planned split fields.
4. Create patient-level train, validation, and test split files.
5. Define fixed crop geometry and preprocessing parameters in config.
6. Verify that target names use radiologist-assessed malignancy-risk terminology.
7. Prepare placeholders for later 3D crop loading, multi-task model training, evaluation, and explainability.

After this slice is complete, the next implementation step should be DICOM reading and annotation parsing, followed by manifest generation and split validation.
