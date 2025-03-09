# mscz-to-video
Render a MuseScore file to a video file

<video src="Flower Dance.mp4" controls="controls" width="100%">Your browser does not support the video tag.</video>

## Features

- [x] Export MuseScore file to video
- [x] Show current note and bar with different highlight color
- [x] Manually set highlight color and transparency
- [x] Smooth cursor movement between notes
- [x] Parallel rendering
- [x] Accelerate with PyTorch including GPU support and JIT compilation 
- [x] Multi GPU support
- [ ] Automatically audio support

## Requirements
- MuseScore
- ffmpeg
- Python 3 (Python Libraries see below)
  - `numpy<2`
  - `Pillow`
  - `webcolors`
  - If you want faster rendering, you can install `torch` following the instructions at https://pytorch.org/get-started/locally/
  - If you want to render using svg, you need to install `cairosvg`

## Usage

1. Create a MuseScore file. You can use the provided example file `Flower Dance.mscz` (Requires MuseScore 4.4 or later)
2. If you don't want your video to scroll, you need to set your page ratio same as your video resolution. You can do this by going to `Format` -> `Page Settings` -> `Page Size` -> `Custom` -> `Width` and `Height` to your desired ratio. Please do not change them too small as the default output size is 330 dpi for PNG and 360 dpi for SVG. You can also change `Staff space` and add new lines to make each page shows better. Personally, I'd also recommend setting `Format` -> `Style` -> `Header & Footer` to make odd/even pages have the same header and footer.
3. Prepare FFmpeg and MuseScore. You can install them using your package manager, or download [MuseScore](https://musescore.org) and [FFmpeg Windows Build By Gyan.dev](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip). Find the path to `ffmpeg` and `MuseScore` executable. You need to set the path to `ffmpeg` and `MuseScore` with `--ffmpeg-path` and `--musescore-path` respectively later.
4. Before converting the file, you may need to learn some basic usage of FFMpeg as this script only passes frames and output file name to FFMpeg and the frames are in `RGB24` format but usually videos are encoded in `YUV420p` format. You may also read help of this script by running `python3 mscz2video.py --help` as it has a lot of options to customize the output video. Refer to [Command Line Arguments](#command-line-arguments) for more information.
5. Now you can convert the file by running `python3 mscz2video.py "Flower Dance.mscz" "Flower Dance.mp4" --ffmpeg-path "path/to/ffmpeg" --musescore-path "path/to/MuseScore" --start-offset 1 --end-offset 5 -r 30 -s 1920x1080 -j 4 --smooth-cursor --ffmpeg-arg-ext '-i "Flower Dance.flac" -c:v libx265 -b:v 768k -c:a aac -b:a 128k -pix_fmt yuv420p -tag:v hvc1'` to create a video with 30 fps, 1920x1080 resolution, 1 second wait time before the first note, 5 seconds wait time after the last note, 4 parallel jobs, smooth cursor movement, and encoded with libx265 and aac codec, just like the video above. This script does not automatically add audio to the video, so I added the audio file using `--ffmpeg-arg-ext` option. Remember to export the audio file from MuseScore first manually.
6. You can also use PyTorch (which supports GPU) for faster rendering. For usage, you can read the script help.

## Command Line Arguments
```
usage: mscz2video.py [-h] [-r FPS] [-s SIZE] [--bar-color COLOR] [--bar-alpha UINT8] [--note-color COLOR] [--note-alpha UINT8] [--ffmpeg-path PATH] [--musescore-path PATH] [--start-offset FLOAT] [--end-offset FLOAT] [-ss FLOAT] [-t FLOAT] [--ffmpeg-arg-ext STR] [--ffmpeg-help] [-j UINT] [--cache-limit UINT] [--use-torch] [--torch-devices STR] [--no-device-cache] [--resize-function {crop,rescale}] [--use-svg] [--allow-large-picture] [--smooth-cursor] input_mscz output_video

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
  --ffmpeg-arg-ext STR             Extra ffmpeg arguments. Use --ffmpeg-help for more information
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