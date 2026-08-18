# Phase 2 Label Semantics

## Core Rule

LIDC-IDRI malignancy ratings are **subjective radiologist assessments**. They are not universal pathology-confirmed cancer diagnoses.

All Phase 2 code, documentation, configs, metrics, plots, and filenames should use terminology such as:

- radiologist-assessed malignancy-risk;
- malignancy-risk;
- reader-assessed malignancy-risk;
- LIDC malignancy rating.

Avoid terminology that implies definitive diagnostic truth, such as:

- cancer label;
- cancer diagnosis;
- malignant cancer ground truth;
- benign versus cancer ground truth;
- pathology-confirmed malignancy, unless a separate pathology-confirmed source is explicitly introduced and documented.

## Why This Matters

The LIDC-IDRI annotation process captures reader interpretation of pulmonary nodules on CT. A model trained on these labels is learning to approximate radiologist-assessed imaging risk patterns, not to diagnose cancer with universal pathological certainty.

This distinction affects:

- dataset documentation;
- model and output naming;
- evaluation text;
- error analysis;
- conclusions and limitations;
- any clinical claims.

## Recommended Target Names

Preferred config and code names:

- `malignancy_risk`
- `radiologist_assessed_malignancy_risk`
- `reader_malignancy_rating`
- `malignancy_rating_mean`
- `malignancy_rating_median`
- `malignancy_rating_disagreement`

Avoid:

- `cancer`
- `cancer_label`
- `is_cancer`
- `cancer_prediction`
- `true_malignancy`

## Attribute Labels

LIDC-IDRI nodule attributes such as spiculation, lobulation, texture, margin, and sphericity should also be described as radiologist-assessed imaging attributes.

Recommended naming:

- `spiculation`
- `lobulation`
- `texture`
- `margin`
- `sphericity`
- `reader_attribute_rating`

## Aggregation Semantics

When multiple radiologist ratings exist for a nodule, the repository should preserve both:

- individual reader ratings;
- the selected aggregate target used for modeling.

Aggregation choices must be explicit. Examples:

- median reader rating;
- mean reader rating;
- majority grouped risk category;
- full reader-rating distribution.

High reader disagreement should be treated as uncertainty or ambiguity, not as simple label noise.

## Reporting Language

Acceptable:

- "The model predicts radiologist-assessed malignancy-risk from 3D CT nodule crops."
- "Performance is evaluated against aggregated LIDC-IDRI reader ratings."
- "Errors may reflect model mistakes, reader disagreement, or ambiguity in the nodule appearance."

Not acceptable:

- "The model diagnoses lung cancer."
- "The model predicts true cancer status."
- "False cancer predictions indicate incorrect pathology."

## Phase 2 Interpretation Boundary

Phase 2 results should be framed as computational imaging research on radiologist-assessed nodule characterization. They should not be framed as clinical diagnosis, pathology prediction, or deployable cancer screening without additional validated datasets and clinical evidence.

## Initial Single-Task Baseline

The first Phase 2 baseline uses one characterized reader-level nodule annotation as one modelling record. Original malignancy ratings are retained in metadata, while the binary operational target is:

- ratings 1–2: low risk (`0`);
- rating 3: indeterminate and excluded from binary modelling;
- ratings 4–5: high risk (`1`).

This reader-level sampling does not cluster multiple radiologists' annotations of the same physical nodule. Patient-level splitting keeps related annotations in one data split, but repeated descriptions of a lesion can still give that lesion more influence within its split. This is an explicit limitation of the initial baseline.
