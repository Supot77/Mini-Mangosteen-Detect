# Original-crop evaluation set

This is the 38-image held-out test set used for the primary presentation metric.
The files are copied from the project's original 96x96 crop dataset and are
used without augmentation:

- `overripe`: 2 images
- `ripe`: 18 images
- `unripe`: 18 images

The current Separable CNN 64 Keras and Full INT8 artifacts both score 36/38
(94.74%) on this input protocol. The deployment-matched crop result of 38/38
(100.00%) remains a separate preprocessing check. The source copy under
`training/Mangosteen_EdgeAI/01_data/dataset/test/` is preserved for provenance.
