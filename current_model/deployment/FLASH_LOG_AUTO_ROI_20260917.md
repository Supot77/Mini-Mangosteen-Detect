# Flash log: RGB565 automatic fruit ROI

วันที่: 2026-09-17  
Board: LilyGO T-SIMCAM / ESP32-S3  
Port: `COM9`

## A/B change

- Model: deployment-matched Separable CNN 64 candidate; model bytes unchanged
- Camera input: `240x240 RGB565`
- New preprocessing: scan the RGB565 frame at low resolution, select a plausible
  fruit-colour region, expand it to a square crop, then resize to `96x96`
- Fallback: centered `160x160` crop when the automatic ROI is not plausible
- Dashboard response: RGB565 is converted to JPEG only for preview
- No external images were added

## Build and upload

- PlatformIO build: success
- RAM: 63,384 / 327,680 bytes (19.3%)
- Flash: 1,165,517 / 6,553,600 bytes (17.8%)
- Upload through `COM9`: success
- `Hash of data verified`: bootloader, partition table, and app image
- Board reset: completed via RTS

## Test status

The board is ready for a real OV2640 test. This environment does not open the
board SoftAP because doing so would interrupt the active connection. The user
should test unripe, ripe, and overripe fruit and send a screenshot containing
the prediction plus the diagnostic `ROI` value. Live accuracy and whether the
ROI is selected or falls back to the center crop remain pending that test.
