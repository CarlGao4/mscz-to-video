#!/usr/bin/python3

# mscz-to-video
# Render MuseScore files to video
# Copyright (C) 2025  GitHub CarlGao4

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import pathlib
import sys
import webcolors

import convert_core


class FFmpegHelpAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super(FFmpegHelpAction, self).__init__(
            option_strings=option_strings, dest=dest, default=default, nargs=0, help=help
        )

    def __call__(self, parser, namespace, values, option_string=None):
        print(file=sys.stderr)
        print(
            "This program will use FFMpeg to encode the video. "
            'You can pass extra arguments to ffmpeg, all arguments after "--" will be passed to ffmpeg.',
            file=sys.stderr,
        )
        print("By default, the ffmpeg command will be:", file=sys.stderr)
        print(
            "\x1b[1;37mffmpeg -y -r {fps} -f rawvideo -s {width}x{height}"
            " -pix_fmt rgb24 -i - [Your arguments here] {output}\x1b[0m",
            file=sys.stderr,
        )
        print("Your arguments will be inserted after the input file, but before the output file.", file=sys.stderr)
        print("So you can add things like audio input, encoder settings, etc.", file=sys.stderr)
        print(
            "Before passing to ffmpeg, your command will be splited with shlex4all, so you can use quotes and escapes.",
            file=sys.stderr,
        )
        print(file=sys.stderr)
        parser.exit()


parser = argparse.ArgumentParser(description="Convert MuseScore files to video")
parser.add_argument("input_mscz", type=pathlib.Path, help="Input MuseScore file")
parser.add_argument("output_video", type=pathlib.Path, help="Output video file")
parser.add_argument("-r", "--fps", type=int, dest="fps", default=60, help="Framerate, default 60")
parser.add_argument(
    "-s",
    default=None,
    type=lambda x: tuple(map(int, x.split("x"))),
    dest="size",
    help="Resolution in widthxheight (like 1920x1080), default size of first page",
)
parser.add_argument(
    "--bar-color",
    type=webcolors.html5_parse_legacy_color,
    default=(255, 0, 0),
    dest="bar_color",
    metavar="COLOR",
    help="Color of current bar, default red, support 3/6 digits rgb (begin with #) and color names in HTML format",
)
parser.add_argument(
    "--bar-alpha", type=int, default=85, dest="bar_alpha", metavar="UINT8", help="Alpha of current bar, default 85/255"
)
parser.add_argument(
    "--note-color",
    type=webcolors.html5_parse_legacy_color,
    default=(0, 255, 255),
    dest="note_color",
    metavar="COLOR",
    help="Color of current note, default cyan, support 3/6 digits rgb (begin with #) and color names in HTML format",
)
parser.add_argument(
    "--note-alpha",
    type=int,
    default=85,
    dest="note_alpha",
    metavar="UINT8",
    help="Alpha of current note, default 85/255",
)
parser.add_argument(
    "--ffmpeg-path",
    type=str,
    default="ffmpeg",
    dest="ffmpeg_path",
    metavar="PATH",
    help="Path to ffmpeg, default ffmpeg",
)
parser.add_argument(
    "--musescore-path",
    type=str,
    default="musescore",
    dest="musescore_path",
    metavar="PATH",
    help="Path to MuseScore, default musescore",
)
parser.add_argument(
    "--start-offset",
    type=float,
    default=0.0,
    dest="start_offset",
    metavar="FLOAT",
    help="Wait time before first note, default 0.0",
)
parser.add_argument(
    "--end-offset",
    type=float,
    default=0.0,
    dest="end_offset",
    metavar="FLOAT",
    help="Wait time after last note, default 0.0",
)
parser.add_argument(
    "-ss",
    type=float,
    default=0.0,
    dest="ss",
    metavar="FLOAT",
    help="Start time offset in seconds, default 0.0, include start offset "
    "(start_offset=1 and ss=1 will result no wait time)",
)
parser.add_argument(
    "-t",
    type=float,
    default=float("inf"),
    dest="t",
    metavar="FLOAT",
    help="Duration in seconds, default to the end of the song",
)
parser.add_argument("--ffmpeg-help", action=FFmpegHelpAction, help="Print help for ffmpeg arguments")
parser.add_argument(
    "-j", "--jobs", type=int, default=1, dest="jobs", metavar="UINT", help="Number of parallel jobs, default 1"
)
parser.add_argument(
    "--cache-limit",
    type=int,
    default=60,
    dest="cache_limit",
    metavar="UINT",
    help="Cache same frames limit in memory, default 100",
)
parser.add_argument(
    "--use-torch",
    action="store_true",
    dest="use_torch",
    help="Use PyTorch for image processing, faster and with GPU support",
)
parser.add_argument(
    "--torch-devices",
    type=str,
    default="cpu",
    dest="torch_devices",
    metavar="STR",
    help="PyTorch devices, separated with colon, default cpu only. "
    "You can use a comma to set max parallel jobs on each device, like cuda:0,1;cpu,4 and "
    "sum of max jobs must be greater than or equal to parallel jobs",
)
parser.add_argument(
    "--no-device-cache",
    action="store_true",
    dest="no_device_cache",
    help="Do not cache original images to every device. Load from memory every time. "
    "May slower but use less device memory.",
)
parser.add_argument(
    "--resize-function",
    type=str,
    default="crop",
    dest="resize_function",
    choices=["crop", "rescale"],
    help="Resize function to use, crop will crop each page to the largest possible size with the same ratio, "
    "rescale will resize each page to target size, default crop",
)
parser.add_argument(
    "--use-svg",
    action="store_true",
    dest="use_svg",
    help="Use SVG exported by MuseScore instead of PNG. May clearer and requires CairoSVG but may fail sometimes",
)
parser.add_argument("--smooth-cursor", action="store_true", dest="smooth_cursor", help="Smooth cursor movement")
parser.add_argument("--version", action="version", version=f"%(prog)s {convert_core.__version__}")
if "--" in sys.argv:
    args = parser.parse_args(sys.argv[1 : sys.argv.index("--")])
    ffmpeg_arg_ext = sys.argv[sys.argv.index("--") + 1 :]
else:
    args = parser.parse_args()
    ffmpeg_arg_ext = []

converter = convert_core.Converter(
    use_torch=args.use_torch,
    ffmpeg_path=args.ffmpeg_path,
    musescore_path=args.musescore_path,
)
converter.load_score(args.input_mscz, use_svg=args.use_svg)
converter.convert(
    args.output_video,
    cache_limit=args.cache_limit,
    torch_devices=args.torch_devices,
    smooth_cursor=args.smooth_cursor,
    size=args.size,
    bar_color=args.bar_color,
    bar_alpha=args.bar_alpha,
    note_color=args.note_color,
    note_alpha=args.note_alpha,
    start_offset=args.start_offset,
    end_offset=args.end_offset,
    ss=args.ss,
    jobs=args.jobs,
    resize_method=args.resize_function,
    fps=args.fps,
    t=args.t,
    ffmpeg_arg_ext=ffmpeg_arg_ext,
    no_device_cache=args.no_device_cache,
    wait=True,
)
