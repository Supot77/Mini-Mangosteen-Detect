# Flash log: runtime camera configuration panel

วันที่: 2026-09-17  
Board: LilyGO T-SIMCAM / ESP32-S3  
Port: `COM9`

## Change applied

- Model: current Separable CNN 64 INT8; model bytes unchanged
- Preprocessing: unchanged original full-frame baseline
  `RGB565 240x240 -> RGB888 -> 96x96 -> INT8`
- Added a web Camera Config panel for runtime OV2640 settings:
  brightness, contrast, saturation, sharpness, auto white balance, AWB gain,
  and auto exposure
- Added `GET /camera-config` endpoint to read/apply settings immediately
- Settings are runtime-only and return to defaults after board restart
- Frame format, resolution, and model input path remain locked
- No external images were added

## Build and upload

- PlatformIO build: success
- RAM: 63,424 / 327,680 bytes (19.4%)
- Flash: 1,177,841 / 6,553,600 bytes (18.0%)
- Upload through `COM9`: success
- `Hash of data verified`: bootloader, partition table, and app image
- Board reset: completed via RTS

## Test status

The firmware is ready for a user-side UI test. The board SoftAP was not opened
from this environment because doing so would interrupt the active connection.
