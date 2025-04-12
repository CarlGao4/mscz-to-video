# 0.3
Release date: Apr. 12, 2025

## Updates

1. Be able to set note highlight area to have fixed width, instead of adjusting to actual note width provided by MuseScore. Default fixed width is calculated from the original mscz file.
2. Added a button to link audio delay to render offset.
3. Will not use CPU for render if GPU is available by default.

# 0.2
Release date: Mar. 31, 2025

## Updates

1. Will keep note highlight area moving to the end of the bar if the bar is the last one in one row.
2. Added "Copy all" button in log window right-click menu.
3. Uses torch 2.6.0 so Intel GPU acceleration coverage is improved.
4. Added `prores_videotoolbox` codec on macOS.

## Fixes

1. Will fallback to CPU when GPU acceleration is not available. (Especially for Intel GPU and MPS)

# 0.1
Release date: Mar. 15, 2025

## Updates

1. Initial release.
