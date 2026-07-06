# Error Analysis Notes

## Purpose

This analysis examines the clinical behaviour of the chest X-ray baseline beyond aggregate accuracy. The goal is to identify the types of mistakes made by the model and inspect whether Grad-CAM visualisations highlight clinically relevant image regions.

## Summary of Results

The model produced 167 misclassified examples on the test set:

| Error type | Count |
|---|---:|
| False positives | 165 |
| False negatives | 2 |

The model therefore behaves like a high-sensitivity screening classifier: it rarely misses pneumonia cases, but it frequently over-predicts pneumonia in normal cases.

## Clinical Metrics

| Metric | Value |
|---|---:|
| Sensitivity | 99.49% |
| Specificity | 29.49% |
| False-negative rate | 0.51% |
| False-positive rate | 70.51% |
| Positive predictive value | 70.16% |
| Negative predictive value | 97.18% |
| Balanced accuracy | 64.49% |

These results show why accuracy alone is insufficient for medical imaging evaluation. The model's overall accuracy is moderate, but its clinical behaviour is asymmetric: it strongly favours disease detection over normal-case rejection.

## Grad-CAM Interpretation

Grad-CAM visualisations were generated for selected false-positive and false-negative examples.

For false positives, several heatmaps show broad activation across the lung fields. This suggests that the model is attending to anatomically relevant regions, but the activation is not focused on a specific abnormality. The model may be responding to general lung texture, rib patterns, contrast, or acquisition-related visual cues rather than a clearly disease-specific finding.

Some false-positive heatmaps also show activation around ribs, image borders, and lower lung boundaries. This suggests that the model may be partly sensitive to non-pathology cues.

For false negatives, the heatmaps show weaker or less clinically focused activation. One false-negative example has a strong activation near the image marker/top-left region rather than the lung fields, suggesting possible shortcut learning or artifact sensitivity.

## Interpretation

The model has learned useful disease-sensitive patterns, but it is not yet clinically reliable. Its high sensitivity is promising for screening-style use, but its low specificity would create substantial unnecessary follow-up in a real clinical workflow.

The Grad-CAM outputs suggest that the model sometimes attends to relevant anatomy, but may also rely on broad texture patterns or non-clinical artifacts. This motivates stronger preprocessing, better validation, threshold analysis, and comparison against correct predictions.

## Limitations

- Grad-CAM provides coarse visual explanations and should not be treated as definitive proof of model reasoning.
- The examples analysed here are selected from model errors and do not represent the full test distribution.
- The current model is a simple CNN baseline, not a clinically validated system.
- The current task is pneumonia classification and is used as a medical-imaging baseline before moving toward oncology-specific imaging.

## Next Steps

1. Compare Grad-CAM outputs for correct and incorrect predictions.
2. Select examples by confidence level rather than only taking the first errors.
3. Add threshold analysis to study the sensitivity-specificity trade-off.
4. Improve preprocessing and consider lung-field-focused analysis.
5. Move toward an oncology-relevant dataset after completing this baseline evaluation.
