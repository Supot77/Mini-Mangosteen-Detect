# Flash log: RGB565 preprocessing probe

วันที่: 2026-09-17  
Board: LilyGO T-SIMCAM / ESP32-S3  
Port: `COM9`

## A/B change

- Model: deployment-matched Separable CNN 64 candidate
- Model unchanged from the previous flash
- Preprocessing changed from `JPEG -> RGB888 -> center crop 160x160` to the original `RGB565 full frame 240x240`
- Input conversion: RGB565 byte swap, expand to RGB888, then `240x240 -> 96x96`
- Camera output preview: RGB565 converted to JPEG only for the web response

## Build and upload

- PlatformIO build: success
- RAM: 63,384 / 327,680 bytes (19.3%)
- Flash: 1,164,573 / 6,553,600 bytes (17.8%)
- Upload through `COM9`: success
- `Hash of data verified`: bootloader, partition table, app image
- Board reset: completed via RTS

## Test status

The board is ready for a real OV2640 test. This environment does not open the
board SoftAP because doing so would interrupt the active connection. Live
prediction accuracy therefore remains pending the user's test result.
