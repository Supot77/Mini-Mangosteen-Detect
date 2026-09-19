# Flash log: current model with the original full-frame preprocessing

วันที่: 2026-09-17  
Board: LilyGO T-SIMCAM / ESP32-S3  
Port: `COM9`

## Change applied

- Model: current deployment-matched Separable CNN 64 INT8; model bytes unchanged
- Preprocessing restored to the original MobileNetV2 Alpha 0.5 deployment path:
  `RGB565 240x240 -> RGB888 -> full-frame 96x96 -> signed INT8 (channel - 128)`
- Removed automatic fruit ROI selection
- Removed center-crop and ROI fallback logic
- Dashboard still converts the RGB565 frame to JPEG for preview only
- No external images were added

## Build and upload

- PlatformIO build: success
- RAM: 63,384 / 327,680 bytes (19.3%)
- Flash: 1,164,621 / 6,553,600 bytes (17.8%)
- Upload through `COM9`: success
- `Hash of data verified`: bootloader, partition table, and app image
- Board reset: completed via RTS

## Test status

The current model is now running with the original full-frame camera
preprocessing. Live OV2640 accuracy remains pending a test from another device;
this environment does not open the board SoftAP because doing so would interrupt
the active connection. The dashboard diagnostic should show `ROI: 240x240 at
(0,0) [full-frame]`.
