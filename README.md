# mscz-to-video
Render a MuseScore file to a video file

<video controls playsinline style="width:100%;height:fit-content;padding-bottom:56.25%;overflow-y:hidden" class="video-js" data-setup="{}"><source src="https://mscz-video.carlgao4.workers.dev/FlowerDance/FlowerDance.m3u8">Your browser does not support the video tag.</video>

*Here should be a sample video. Please go to [GitHub Pages](https://carlgao4.github.io/mscz-to-video/) to view the video.*

## Features

- [x] Export MuseScore file to video
- [x] Show current note and bar with different highlight color
- [x] Manually set highlight color and transparency
- [x] Smooth cursor movement between notes
- [x] Parallel rendering
- [x] Accelerate with PyTorch including GPU support and JIT compilation (Speed can reach about 4K 52fps on single 4060m GPU)
- [x] Resize function to crop or rescale each page so current note and bar will always be in the center
- [x] Multi GPU support
- [x] Graphical user interface (UI) version
- [x] Automatically add audio to the video (UI version only)
- [x] Realtime preview (UI version only)
- [ ] Automatically audio support

## Requirements

**Only Windows 10 and macOS 10.15 or later are supported. Only 64-bit systems are supported.**

- MuseScore
- ffmpeg
- Python 3 (Python Libraries see below)
  - `numpy<2`
  - `Pillow`
  - `webcolors`
  - If you want faster rendering, you can install `torch` following the instructions at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
  - If you want to render using svg, you need to install `cairosvg`

---

Extra requirements for the UI version:
- `PySide6`
- `psutil`
- `torch`
- `cairosvg` is not required

If you want to accelerate with Intel GPU, you can install [`intel_extension_for_pytorch`](https://github.com/intel/intel-extension-for-pytorch) with XPU support.

## UI Usage

1. Download the binaries from the [release page](https://github.com/CarlGao4/mscz-to-video/releases) according to your platform.
   1. For Windows, if you have Intel integrated GPU (Core 11th Gen or later), Intel or Nvidia discrete GPU, download `mscz2video_xxx_cuda_mkl.7z` and extract it. If you do not have GPUs matching the above, download `mscz2video_xxx_cpu.7z` and extract it. Both versions can use GPU accelerated encoding, but only the former can use GPU accelerated rendering.
   2. For macOS, download `mscz2video_xxx_macOS_[ARCH].dmg` and open it. Drag the app to the Applications folder. `[ARCH]` is the architecture of your Mac, which can be `arm64` or `x86_64`. Apple Developer is expensive so I can't sign the app, so if the app is blocked by macOS, you can refer to [Note for macOS users](#CannotOpen) below.
2. Create a MuseScore file and format it to fit your video, or use the provided example file `Flower Dance.mscz` (Requires MuseScore 4.4 or later). If you don't want your video to scroll, you need to set your page ratio same as your video resolution. You can do this by going to `Format` → `Page Settings` → `Page Size` → `Custom` → `Width` and `Height` to your desired ratio. Please do not change them too small as the default output size is 330 dpi (pixels per inch, wich means that 1 inch is 330 pixels). You can also change `Staff space` and add new lines to make each page shows better. Personally, I'd also recommend setting `Format` → `Style` → `Header & Footer` to make odd/even pages have the same header and footer.
3. Open the program (On Windows, you can just double-click `mscz2video.exe`). If supported Intel GPU is detected and you've downloaded the mkl version, and AOT is not enabled, a popup dialog will appear to ask you to download the AOT version. For more information, you can refer to [AOT docs](MKL-AOT.md).
4. Wait for the program to start up. Before the process is done, you can't load a mscz file. During start up, the program will search for MuseScore and FFmpeg. I've packed ffmpeg along with the program so you don't need to download it, but you need to download MuseScore yourself. The program will search for `C:\Program Files\MuseScore 4\bin\MuseScore4.exe` and `C:\Program Files\MuseScore 3\bin\MuseScore3.exe` on Windows, and `/Applications/MuseScore 4.app/Contents/MacOS/mscore` and `/Applications/MuseScore 3.app/Contents/MacOS/mscore` on macOS. If the program fails to find MuseScore, a dialog will pop up to ask you to set the path to MuseScore manually, just choose the correct path to MuseScore executable. On macOS, you need to enter the application package and find the executable in the `Contents/MacOS` folder. To open an application package, you need to press <kbd>Cmd</kbd> + <kbd>Shift</kbd> + <kbd>G</kbd> in Finder and enter the path.
5. Load the mscz file by clicking `Load MuseScore file` or dragging the file to the button. Before the file is loaded, you can't start rendering. During this time you can change other settings.
    1. **Video settings**: You can set the resolution and framerate of the video. The default resolution is 1920x1080 and the default framerate is 60fps. Other common resolutions are: `1280x720` (720p), `1920x1080` (1080p), `2560x1440` (1440p or 2K), `3840x2160` (4K), `7680x4320` (8K).
    2. **Bar and note highlight color**: the background color of the current bar and current note. You can set the color and transparency, and view the effects on the label.
    3. **Video range**: Start offset and end offset are the time before the first note and after the last note press (*NOT* after the last note release). You can also set the start time and duration of the video. Start time is calculated before the start offset, so if you set start time to 1 and start offset to 1, the video will start at 0 second. The video will end either when the duration is reached or when the last note is pressed, whichever comes first. The start time and duration are used by the program, but not arguments passed to ffmpeg.
    4. **parallel jobs**: The number of parallel jobs to render the video. The default is 1 on each device. Only cpu is available if you've downloaded the CPU version. On macOS, mps is always available if you have M-series SoC or AMD GPU. You can hover over the label to see the name of the device instead of the device ID.
    5. **Cache**: Cache limit will limit the number of frames stored in memory. Repeat frames may occur during rendering, when the program finds a frame that is already rendered, it will use the cached frame instead of rendering it again. However, if the cache limit is reached, the program will delete the oldest frame in the cache. Besides, since parallel rendering is supported, frames may not be rendered in order, so the program will also use the cache to store frames that are rendered but not yet used. The default cache limit is 60. Use device cache will store the original images in device memory, which will speed up the rendering process by avoiding transferring images from CPU to GPU memory. However, this will use more device memory. If your device memory is not sufficient, you can disable this option.
    6. **Smooth cursor**: Smooth cursor movement will make the cursor move smoothly between notes. Without smooth cursor, the note highlight aria will jump to the next note when it is pressed. Just like the playback cursor inside MuseScore, MuseScore 3 does not have a smooth cursor, but MuseScore 4 does. Smooth cursor will make the render process much slower as almost no same frames can be cached.
    7. **Resize function**: Resize function will resize each page to the target size. Crop will crop each page to the largest possible size with the same ratio while keeping the current note in the center. Rescale will resize each page to the target size, ignoring the ratio.
    8. **Encoder settings**: Please remember that this program WILL NOT verify whether your choices can produce valid video!
        - **Muxing audio into the video** This program WILL NOT render audio automatically. You need to export the audio from MuseScore manually first and load it here (or drag it to the button). Please don't forget to set audio delay. Usually, you need to set audio delay same as the start offset, unless you are using other audio sources or start time is not 0.
        - **Video encoder settings** You can choose the video codec and encoder. On Windows and Linux, both Intel, Nvidia, AMD graphic card acceleration are supported (Even you've downloaded the CPU version). Choose encoder ends with `_nvenc` for Nvidia, `_amf` for AMD, `_qsv` for Intel. Please note that the encoder may not be available on your system even it is listed here. For codec, you can choose `h264` (`libx264`, `h264_*`), `hevc` (`libx265`, `hevc_*`), `vp9` (`libvpx-vp9`, `vp9_qsv`), `av1` (`libaom-av1`, `libsvtav1`, `av1_*`) and `prores`. On macOS, GPU acceleration codecs are end with `_videotoolbox`.
        - **Video bitrate control** You can set the video bitrate and quality. You can choose `VBR` (Variable Bitrate) and `CQP` (or `CRF` depending on the codec) for quality control.
        - **Audio codecs** You can choose `aac` (the most common audio codec in videos), `libopus`, `flac` (lossless audio codec), `pcm` (lossless and uncompressed), `alac` (Apple Lossless Audio Codec), `mp3` (lossy audio codec) and `vorbis`. Bitrate control is not available for lossless audio codecs.
6. Now you can start rendering. The preview window will show the realtime rendering frame, refreshing twice a second for better rendering performance. Showing the log window will also slow down the render process, so you can hide it if you don't need it.

<details id="CannotOpen">
  <summary>Note for macOS users</summary>

If the application cannot be launched due to the Mac's security protection feature, try the following:

For macOS versions below 15.0:

1. Right-click on the mscz2video app icon and select "Open".
2. Click "Open" again in the window that appears as follows.

For macOS versions 15.0 or greater:
1. On your Mac, go to System Settings > Privacy & Security > Scroll to the Security section.
2. If you see a message stating "'mscz2video.app' was blocked to protect your Mac." - to the right of this message, click "Open Anyway".
3. Enter your login password, then click OK. This will create an override in Gatekeeper, allowing mscz2video to run.

Similar steps with screenshots can be found on my other project [Demucs-GUI](https://github.com/CarlGao4/Demucs-GUI#CannotOpen).

</details>

## Commandline usage

1. Clone this repo. There are two required files: `mscz2video.py` (Conversion commandline interface script) and `convert_core.py` (Conversion codes), so you can also just download these two files directly and put them in the same directory.
2. Create a MuseScore file. You can use the provided example file `Flower Dance.mscz` (Requires MuseScore 4.4 or later)
3. If you don't want your video to scroll, you need to set your page ratio same as your video resolution. You can do this by going to `Format` → `Page Settings` → `Page Size` → `Custom` → `Width` and `Height` to your desired ratio. Please do not change them too small as the default output size is 330 dpi for PNG and 360 dpi for SVG. You can also change `Staff space` and add new lines to make each page shows better. Personally, I'd also recommend setting `Format` → `Style` → `Header & Footer` to make odd/even pages have the same header and footer.
4. Prepare FFmpeg and MuseScore. You can install them using your package manager, or download [MuseScore](https://musescore.org) and [FFmpeg Windows Build By Gyan.dev](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip). Find the path to `ffmpeg` and `MuseScore` executable. You need to set the path to `ffmpeg` and `MuseScore` with `--ffmpeg-path` and `--musescore-path` respectively later.
5. Before converting the file, you may need to learn some basic usage of FFMpeg as this script only passes frames and output file name to FFMpeg and the frames are in `RGB24` format but usually videos are encoded in `YUV420p` format. You may also read help of this script by running `python3 mscz2video.py --help` as it has a lot of options to customize the output video. Refer to [Command Line Arguments](#command-line-arguments) for more information.
6. Now you can convert the file by running `python3 mscz2video.py "Flower Dance.mscz" "Flower Dance.mp4" --ffmpeg-path "path/to/ffmpeg" --musescore-path "path/to/MuseScore" --start-offset 1 --end-offset 5 -r 30 -s 1920x1080 -j 4 --smooth-cursor "--" -i "Flower Dance.flac" -c:v libx265 -b:v 768k -c:a aac -b:a 128k -pix_fmt yuv420p -tag:v hvc1` to create a video with 30 fps, 1920x1080 resolution, 1 second wait time before the first note, 5 seconds wait time after the last note, 4 parallel jobs, smooth cursor movement, and encoded with libx265 and aac codec, just like the video above. This script does not automatically add audio to the video, so I added the audio file by passing additional arguments to ffmpeg (all arguments after `"--"` will not be parsed and are passed to ffmpeg directly). Remember to export the audio file from MuseScore first manually.
7. You can also use PyTorch (which supports GPU) for faster rendering. For usage, you can read the script help.

## Command Line Arguments
```
usage: mscz2video.py [-h] [-r FPS] [-s SIZE] [--bar-color COLOR] [--bar-alpha UINT8] [--note-color COLOR] [--note-alpha UINT8] [--ffmpeg-path PATH] [--musescore-path PATH] [--start-offset FLOAT] [--end-offset FLOAT] [-ss FLOAT] [-t FLOAT] [--ffmpeg-help] [-j UINT] [--cache-limit UINT] [--use-torch] [--torch-devices STR] [--no-device-cache] [--resize-function {crop,rescale}] [--use-svg] [--allow-large-picture] [--smooth-cursor] input_mscz output_video

Convert MuseScore files to video

positional arguments:
  input_mscz            Input MuseScore file
  output_video          Output video file

options:
  -h, --help                       show this help message and exit
  -r FPS, --fps FPS                Framerate, default 60
  -s SIZE                          Resolution in widthxheight (like 1920x1080), default size of first page
  --bar-color COLOR                Color of current bar, default red, support 3/6 digits rgb (begin with #) and color names in HTML format
  --bar-alpha UINT8                Alpha of current bar, default 85/255
  --note-color COLOR               Color of current note, default cyan, support 3/6 digits rgb (begin with #) and color names in HTML format
  --note-alpha UINT8               Alpha of current note, default 85/255
  --ffmpeg-path PATH               Path to ffmpeg, default ffmpeg
  --musescore-path PATH            Path to MuseScore, default musescore
  --start-offset FLOAT             Wait time before first note, default 0.0
  --end-offset FLOAT               Wait time after last note, default 0.0
  -ss FLOAT                        Start time offset in seconds, default 0.0, include start offset (start_offset=1 and ss=1 will result no wait time)
  -t FLOAT                         Duration in seconds, default to the end of the song
  --ffmpeg-help                    Print help for ffmpeg arguments
  -j UINT, --jobs UINT             Number of parallel jobs, default 1
  --cache-limit UINT               Cache same frames limit in memory, default 100
  --use-torch                      Use PyTorch for image processing, faster and with GPU support
  --torch-devices STR              PyTorch devices, separated with colon, default cpu only. You can use a comma to set max parallel jobs on each device, like cuda:0,1;cpu,4 and sum of max jobs must be greater than or equal to parallel jobs
  --no-device-cache                Do not cache original images to every device. Load from memory every time. May slower but use less device memory.
  --resize-function {crop,rescale} Resize function to use, crop will crop each page to the largest possible size with the same ratio, rescale will resize each page to target size, default crop
  --use-svg                        Use SVG exported by MuseScore instead of PNG. May clearer and requires CairoSVG but may fail sometimes
  --allow-large-picture            Allow reading large picture with PIL
  --smooth-cursor                  Smooth cursor movement
```