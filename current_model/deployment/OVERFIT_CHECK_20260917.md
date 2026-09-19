# Overfit check for deployment-matched candidate

Candidate: `20260917_001333/separable_cnn_64`  
Dataset: `01_data/dataset_deployment_matched_v4`

## Evidence checked

| Check | Result | Interpretation |
|---|---:|---|
| Matched test set | 38/38 = 100.00% | Perfect on the deployment-matched holdout |
| Original 96x96 crop test set | 36/38 = 94.74% | Performance remains high under the previous crop style |
| Final training accuracy | about 92.5% | Does not greatly exceed validation |
| Final validation accuracy | 91.18% | Small train/validation gap |
| Duplicate raw filenames across splits | 0 | No exact filename duplication found |

The original crop test confusion matrix for this candidate is:

```text
              predicted
              overripe ripe unripe
true overripe      2     0      0
true ripe          1    16      1
true unripe        0     0     18
```

## Conclusion

There is no strong evidence of conventional overfitting from these checks. The
candidate also generalizes reasonably well to the old crop style, where it
achieves 94.74% instead of collapsing to one class.

This is not a final field-validation result: the test set has only 38 images,
only 2 are `overripe`, and the images belong to the same original capture
collection. The decisive check remains a real OV2640 board test after the
candidate is converted to the firmware C-array and flashed.
