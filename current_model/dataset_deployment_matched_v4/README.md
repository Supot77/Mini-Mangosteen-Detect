# Deployment-matched crop dataset

Canonical images are cropped from the project's raw originals using a compact foreground detector, then resized to 96x96 to match the board model input. Existing `aug_` files are copied only for the training split as internal augmentation. No external images are added.
