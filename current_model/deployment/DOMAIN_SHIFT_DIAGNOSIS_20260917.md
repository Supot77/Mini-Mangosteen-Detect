# Board domain-shift diagnosis

วันที่: 2026-09-17

## Evidence

| Input path | Result for `unripe` test | Meaning |
|---|---:|---|
| Deployment-matched fruit crop | 18/18 correct | Model can learn the class from fruit-centric input |
| Original 96x96 crop | 18/18 correct | Model is not dependent on only the new crop generator |
| Simulated RGB565 full-frame raw input | 3/18 correct | Full-frame framing/background causes the failure |
| Live board RGB888 center crop | `unripe` still missed | Fixed center crop is not sufficient |
| Live board RGB565 full-frame | `unripe` still missed | Changing pixel format alone is not sufficient |

The simulated RGB565 full-frame test produced accuracy 55.26% (21/38), with
15 of 18 `unripe` images predicted as `ripe`. This reproduces the important
failure mode before flashing and shows that the model is not seeing the same
fruit-centric input used during training.

## Dataset count clarification

The deployment-matched train split contains 78 canonical `unripe` images and
80 canonical `ripe` images. After the internal generated augmentation files
are included, the effective counts are 56 `overripe`, 80 `ripe`, and 78
`unripe`. Therefore `unripe` is not actually the largest class, but its
offline recall is already 100%; adding more of the same visual domain is not
the immediate fix.

## Recommended next fix

Do not keep changing RGB565/RGB888. The next controlled change should be an
automatic fruit ROI step in firmware: detect the compact fruit region, make a
square crop around it, then resize that crop to 96x96. A center-crop fallback
should remain for frames where detection is uncertain. This makes the board
input match the training crop without adding external images.
