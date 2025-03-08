#!/usr/bin/python3

# mscz2video.py
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
import base64
import bisect
import collections
import io
import json
import numpy as np
import pathlib
import PIL.Image
import PIL.ImageFile
import shlex4all as shlex
import subprocess
import sys
import threading
import time
import webcolors
import xml.etree.ElementTree as ET

__version__ = "0.1"


class FFmpegHelpAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super(FFmpegHelpAction, self).__init__(
            option_strings=option_strings, dest=dest, default=default, nargs=0, help=help
        )

    def __call__(self, parser, namespace, values, option_string=None):
        print(file=sys.stderr)
        print(
            "This program will use FFMpeg to encode the video. "
            "You can pass extra arguments to ffmpeg using --ffmpeg-arg-ext.",
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
    type=str,
    default="#f00",
    dest="bar_color",
    metavar="COLOR",
    help="Color of current bar, default red, support 3/6 digits rgb (begin with #) and color names in HTML format",
)
parser.add_argument(
    "--bar-alpha", type=int, default=85, dest="bar_alpha", metavar="UINT8", help="Alpha of current bar, default 85/255"
)
parser.add_argument(
    "--note-color",
    type=str,
    default="#0ff",
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
parser.add_argument(
    "--ffmpeg-arg-ext",
    type=str,
    default="",
    dest="ffmpeg_arg_ext",
    metavar="STR",
    help="Extra ffmpeg arguments. Use --ffmpeg-help for more information",
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
parser.add_argument(
    "--allow-large-picture",
    action="store_true",
    dest="allow_large_picture",
    help="Allow reading large picture with PIL",
)
parser.add_argument("--smooth-cursor", action="store_true", dest="smooth_cursor", help="Smooth cursor movement")
parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
args = parser.parse_args()

if args.allow_large_picture:
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
    PIL.Image.MAX_IMAGE_PIXELS = None

# Load MuseScore file data
print("Converting MuseScore file to json...", file=sys.stderr)
musescore = subprocess.Popen(
    [
        args.musescore_path,
        "--score-media",
        str(args.input_mscz.resolve()),
    ],
    stdout=subprocess.PIPE,
)

data = json.loads("{" + musescore.communicate()[0].decode("utf-8").split("{", 1)[1])

# Convert svg to png
pngs = []
if args.use_svg:
    import cairosvg

    for page in data["svgs"]:
        print(f"Converting svg to png... {len(pngs)} / {len(data['svgs'])}", end="\r", file=sys.stderr)
        b = io.BytesIO()
        cairosvg.svg2png(bytestring=base64.b64decode(page), write_to=b)
        b.seek(0)
        pngs.append(np.array(PIL.Image.open(b).convert("RGBA")).astype(np.float16) / 255)
    print("Converting svg to png... Done          ", file=sys.stderr)
    highlight_ratio = (11, 11)
else:
    for page in data["pngs"]:
        print(f"Reading png... {len(pngs)} / {len(data['pngs'])}", end="\r", file=sys.stderr)
        b = io.BytesIO(base64.b64decode(page))
        pngs.append(np.array(PIL.Image.open(b).convert("RGBA")).astype(np.float16) / 255)
    print("Reading png... Done          ", file=sys.stderr)
    highlight_ratio = (12, 12)

if args.size is None:
    args.size = pngs[0].shape[1], pngs[0].shape[0]

print("Original page size:", pngs[0].shape, file=sys.stderr)

# Get times of bars and notes
bars: list[tuple[int, int]] = []
notes: list[tuple[int, int]] = []
mposXML = ET.fromstring(base64.b64decode(data["mposXML"]))
sposXML = ET.fromstring(base64.b64decode(data["sposXML"]))
for bar in mposXML.find(".//events"):
    bars.append((int(bar.attrib["position"]), int(bar.attrib["elid"])))
for note in sposXML.find(".//events"):
    notes.append((int(note.attrib["position"]), int(note.attrib["elid"])))
bars.sort()
notes.sort()


class CacheWithQueue:
    def __init__(self, limit):
        self._cache = collections.OrderedDict()
        self._reuse = {}
        self._reuse_counter = {}
        self._limit = limit

    def __contains__(self, key):
        return key in self._cache or key in self._reuse

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]
        if key in self._reuse:
            ret = self._reuse[key]
            self._reuse_counter[key] -= 1
            if self._reuse_counter[key] == 0:
                del self._reuse_counter[key]
                self._cache[key] = self._reuse.pop(key)
                if len(self._cache) > self._limit:
                    self._cache.popitem(last=False)
            return ret
        raise KeyError(key)

    def __setitem__(self, key, value):
        if key in self._reuse:
            self._reuse[key] = value
        else:
            self._cache[key] = value
        if len(self._cache) + len(self._reuse) > self._limit and len(self._cache) > 1:
            self._cache.popitem(last=False)

    def add_reuse(self, key):
        self._reuse_counter[key] = self._reuse_counter.get(key, 0) + 1
        if key in self._reuse:
            return
        self._reuse[key] = self._cache.pop(key)


def resize_and_crop_to(img: PIL.Image.Image, w, h, l, t, r, b):
    # First resize the input image to ensure img.width == w or img.height == h
    # Image ratio is preserved, but the other dimension may be larger than the target
    if img.width / w > img.height / h:
        # Target ratio is taller than the image
        rescale_ratio = h / img.height
        img = img.resize((int(img.width * h / img.height), h), resample=PIL.Image.BICUBIC)
        x_center = (l + r) / 2 * rescale_ratio
        x_left = int(x_center - w / 2)
        if x_left < 0:
            x_left = 0
        elif x_left + w > img.width:
            x_left = img.width - w
        x_right = x_left + w
        return img.crop((x_left, 0, x_right, h)), (
            int(x_left / rescale_ratio),
            0,
            int(x_right / rescale_ratio),
            img.height,
        )
    else:
        # Target ratio is wider than the image
        rescale_ratio = w / img.width
        img = img.resize((w, int(img.height * w / img.width)), resample=PIL.Image.BICUBIC)
        y_center = (t + b) / 2 * rescale_ratio
        y_top = int(y_center - h / 2)
        if y_top < 0:
            y_top = 0
        elif y_top + h > img.height:
            y_top = img.height - h
        y_bottom = y_top + h
        return img.crop((0, y_top, w, y_bottom)), (
            0,
            int(y_top / rescale_ratio),
            img.width,
            int(y_bottom / rescale_ratio),
        )


def direct_resize_to(img: PIL.Image.Image, w, h, l, t, r, b):
    return img.resize((w, h), resample=PIL.Image.BICUBIC), (0, 0, img.width, img.height)


if args.use_torch:
    import torch

    if "xpu" in args.torch_devices:
        import intel_extension_for_pytorch as ipex

    torch_devices = {}
    for device in args.torch_devices.split(";"):
        if "," in device:
            device, max_jobs = device.split(",")
            max_jobs = int(max_jobs)
            torch_devices[device] = {
                "device": torch.device(device),
                "max_jobs": max_jobs,
                "current_jobs": 0,
                "name": device,
            }
        else:
            torch_devices[device] = {
                "device": torch.device(device),
                "max_jobs": float("inf"),
                "current_jobs": 0,
                "name": device,
            }
    if sum(d["max_jobs"] for d in torch_devices.values()) < args.jobs:
        raise ValueError("Not enough max jobs in torch devices")

    @torch.jit.script
    def resize_and_crop_to_torch(
        img: torch.Tensor, w: int, h: int, l: int, t: int, r: int, b: int, fallback_cpu: bool = False
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        # First resize the input image to ensure img.width == w or img.height == h
        # Image ratio is preserved, but the other dimension may be larger than the target
        width = img.shape[1]
        height = img.shape[0]
        if width / w > height / h:
            # Target ratio is taller than the image
            rescale_ratio = h / height
            if fallback_cpu:
                img = (
                    torch.nn.functional.interpolate(
                        img[None, ...].permute(0, 3, 1, 2).cpu().to(torch.float32),
                        size=(h, int(width * rescale_ratio)),
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    )
                    .to(img.device)
                    .permute(0, 2, 3, 1)
                    .squeeze(0)
                    .clamp(0, 1)
                    .to(torch.float16)
                )
            else:
                img = (
                    torch.nn.functional.interpolate(
                        img[None, ...].permute(0, 3, 1, 2).to(torch.float32),
                        size=(h, int(width * rescale_ratio)),
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    )
                    .permute(0, 2, 3, 1)
                    .squeeze(0)
                    .clamp(0, 1)
                    .to(torch.float16)
                )
            width, height = img.shape[1], img.shape[0]
            x_center = (l + r) / 2 * rescale_ratio
            x_left = int(x_center - w / 2)
            if x_left < 0:
                x_left = 0
            elif x_left + w > width:
                x_left = width - w
            x_right = x_left + w
            return img[:, x_left:x_right, :], (int(x_left / rescale_ratio), 0, int(x_right / rescale_ratio), height)
        else:
            # Target ratio is wider than the image
            rescale_ratio = w / width
            img = (
                torch.nn.functional.interpolate(
                    img[None, ...].permute(0, 3, 1, 2).to(torch.float32),
                    size=(int(height * rescale_ratio), w),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                .permute(0, 2, 3, 1)
                .squeeze(0)
                .clamp(0, 1)
                .to(torch.float16)
            )
            width, height = img.shape[1], img.shape[0]
            y_center = (t + b) / 2 * rescale_ratio
            y_top = int(y_center - h / 2)
            if y_top < 0:
                y_top = 0
            elif y_top + h > height:
                y_top = height - h
            y_bottom = y_top + h
            return img[y_top:y_bottom, :], (0, int(y_top / rescale_ratio), width, int(y_bottom / rescale_ratio))

    @torch.jit.script
    def direct_resize_to_torch(
        img: torch.Tensor, w: int, h: int, l: int, t: int, r: int, b: int, fallback_cpu: bool = False
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        if fallback_cpu:
            return (
                torch.nn.functional.interpolate(
                    img[None, ...].permute(0, 3, 1, 2).cpu().to(torch.float32),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                .to(img.device)
                .permute(0, 2, 3, 1)
                .squeeze(0)
                .clamp(0, 1)
                .to(torch.float16),
                (0, 0, img.shape[1], img.shape[0]),
            )
        else:
            return (
                torch.nn.functional.interpolate(
                    img[None, ...].permute(0, 3, 1, 2).to(torch.float32),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                .permute(0, 2, 3, 1)
                .squeeze(0)
                .clamp(0, 1)
                .to(torch.float16)
            ), (0, 0, img.shape[1], img.shape[0])

    if args.resize_function == "crop":
        resize_torch = resize_and_crop_to_torch
    else:
        resize_torch = direct_resize_to_torch


if args.resize_function == "crop":
    resize = resize_and_crop_to
else:
    resize = direct_resize_to

print("Rescaling MuseScore pages...", end="\r", file=sys.stderr)
if args.resize_function == "rescale":
    highlight_ratio = (
        highlight_ratio[0] / args.size[0] * pngs[0].shape[1],
        highlight_ratio[1] / args.size[1] * pngs[0].shape[0],
    )
    pngs = [
        (
            print(f"Rescaling MuseScore pages... {i} / {len(pngs)}", end="\r", file=sys.stderr),
            np.array(
                direct_resize_to(PIL.Image.fromarray((png * 255).astype(np.uint8)), *args.size, 0, 0, 0, 0)[0]
            ).astype(np.float16)
            / 255,
        )[1]
        for (i, png) in enumerate(pngs)
    ]
else:
    if pngs[0].shape[1] / args.size[0] > pngs[0].shape[0] / args.size[1]:
        # Target ratio is taller than the image
        rescale_ratio = args.size[1] / pngs[0].shape[0]
    else:
        # Target ratio is wider than the image
        rescale_ratio = args.size[0] / pngs[0].shape[1]
    pngs = [
        (
            print(f"Rescaling MuseScore pages... {i} / {len(pngs)}", end="\r", file=sys.stderr),
            np.array(
                direct_resize_to(
                    PIL.Image.fromarray((png * 255).astype(np.uint8)),
                    int(pngs[0].shape[1] * rescale_ratio),
                    int(pngs[0].shape[0] * rescale_ratio),
                    0,
                    0,
                    0,
                    0,
                )[0]
            ).astype(np.float16)
            / 255,
        )[1]
        for (i, png) in enumerate(pngs)
    ]
    highlight_ratio = (highlight_ratio[0] / rescale_ratio, highlight_ratio[1] / rescale_ratio)
print("Rescaling MuseScore pages... Done          ", file=sys.stderr)

# Get bars and notes positions
bar_pos: dict[int, dict[str, int]] = {}
note_pos: dict[int, dict[str, int]] = {}
for bar in mposXML.find(".//elements"):
    bar_pos[int(bar.attrib["id"])] = {
        "x": int(float(bar.attrib["x"]) / highlight_ratio[0]),
        "y": int(float(bar.attrib["y"]) / highlight_ratio[1]),
        "width": int(float(bar.attrib["sx"]) / highlight_ratio[0]),
        "height": int(float(bar.attrib["sy"]) / highlight_ratio[1]),
        "page": int(bar.attrib["page"]),
    }
for note in sposXML.find(".//elements"):
    note_pos[int(note.attrib["id"])] = {
        "x": int(float(note.attrib["x"]) / highlight_ratio[0]),
        "y": int(float(note.attrib["y"]) / highlight_ratio[1]),
        "width": int(float(note.attrib["sx"]) / highlight_ratio[0]),
        "height": int(float(note.attrib["sy"]) / highlight_ratio[1]),
        "page": int(note.attrib["page"]),
    }
print("Bar count:", len(bars), file=sys.stderr)
print("Note time count:", len(notes), file=sys.stderr)

pending_frames = set()
cached_frames = CacheWithQueue(args.cache_limit)
same_frame_queue = {}
frame_key_map = {}
first_frame = (
    resize(PIL.Image.fromarray((pngs[0].copy() * 255).astype(np.uint8)), *args.size, 0, 0, 0, 0)[0]
    .convert("RGB")
    .tobytes()
)
lock = threading.RLock()

if args.use_torch:
    if args.no_device_cache:
        pngs_torch = [torch.from_numpy(png) for png in pngs]
        pngs = {device: pngs_torch for device in torch_devices}
    else:
        pngs = {
            device: [torch.from_numpy(png).to(torch_devices[device]["device"]) for png in pngs]
            for device in torch_devices
        }

program_status = ""
ffmpeg_status = b""
update_status = threading.Event()


def calc_highlight_pos(
    t: float,
    bar_idx: int,
    note_idx: int,
    page: int,
    smooth_cursor: bool,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    note = note_pos[notes[note_idx][1]]
    bar = bar_pos[bars[bar_idx][1]]
    bar_box = (bar["x"], bar["y"], bar["x"] + bar["width"], bar["y"] + bar["height"])
    if not smooth_cursor:
        note_box = (note["x"], note["y"], note["x"] + note["width"], note["y"] + note["height"])
    else:
        next_note = note_pos[notes[note_idx + 1][1]] if note_idx + 1 < len(notes) else note
        next_page = next_note["page"]
        next_bar_idx = (
            (bisect.bisect_right(bars, notes[note_idx + 1][0], key=lambda x: x[0]) - 1)
            if note_idx + 1 < len(notes)
            else bar_idx
        )
        if page != next_page or note_idx + 1 >= len(notes):
            note_box = (note["x"], note["y"], note["x"] + note["width"], note["y"] + note["height"])
        else:
            note_box_current = (note["x"], note["y"], note["x"] + note["width"], note["y"] + note["height"])
            note_box_next = (
                next_note["x"],
                next_note["y"],
                next_note["x"] + next_note["width"],
                next_note["y"] + next_note["height"],
            )
            if (
                note_box_next[1] >= note_box_current[3]
                or note_box_next[2] <= note_box_current[0]
                or note_box_next[3] <= note_box_current[1]
                or not (bars[bar_idx][1] <= bars[next_bar_idx][1] <= bars[bar_idx][1] + 1)
            ):
                note_box = note_box_current
            else:
                current_note_time = notes[note_idx][0]
                next_note_time = notes[note_idx + 1][0]
                note_box = tuple(
                    int(
                        (note_box_current[i] * (next_note_time - t) + note_box_next[i] * (t - current_note_time))
                        / (next_note_time - current_note_time)
                    )
                    for i in range(4)
                )
    return bar_box, note_box


if args.use_torch:

    @torch.jit.script
    def process_torch(
        img: torch.Tensor,
        bar_box: tuple[int, int, int, int],
        note_box: tuple[int, int, int, int],
        out_size: tuple[int, int],
        device: torch.device,
        bar_color: torch.Tensor,
        note_color: torch.Tensor,
    ) -> torch.Tensor:
        img, box = resize_torch(img, *out_size, *note_box, "xpu" in device.type)
        overlay = torch.zeros_like(img, dtype=torch.float16, device=device)
        bar_box_offseted = (
            max(bar_box[0] - box[0], 0),
            max(bar_box[1] - box[1], 0),
            min(bar_box[2] - box[0], out_size[0]),
            min(bar_box[3] - box[1], out_size[1]),
        )
        note_box_offseted = (
            max(note_box[0] - box[0], 0),
            max(note_box[1] - box[1], 0),
            min(note_box[2] - box[0], out_size[0]),
            min(note_box[3] - box[1], out_size[1]),
        )
        # Draw current bar
        overlay[bar_box_offseted[1] : bar_box_offseted[3], bar_box_offseted[0] : bar_box_offseted[2]] = bar_color
        # Draw current note
        overlay[note_box_offseted[1] : note_box_offseted[3], note_box_offseted[0] : note_box_offseted[2]] = note_color
        overlay_area = (
            min(bar_box_offseted[1], note_box_offseted[1]),
            max(bar_box_offseted[3], note_box_offseted[3]),
            min(bar_box_offseted[0], note_box_offseted[0]),
            max(bar_box_offseted[2], note_box_offseted[2]),
        )
        if overlay.any():
            overlay /= torch.tensor(255.0, dtype=torch.float16, device=device)
            overlay[overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3] *= img[
                overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3
            ]
            overlay[overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3] *= overlay[
                overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], 3, None
            ]
            img[overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3] *= (
                torch.tensor(1.0, dtype=torch.float16, device=device)
                - overlay[overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], 3, None]
            )
            img[overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3] += overlay[
                overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3
            ]
            img[..., 3] = torch.tensor(1.0, dtype=torch.float16, device=device)
        return (img[..., :3] * torch.tensor(255.0, dtype=torch.float16, device=device)).to(torch.uint8).cpu()


def get_frame(frame_id, t):
    global next_to_send, program_status
    while len(frame_key_map) > args.cache_limit:
        time.sleep(0.1)
    program_status = "%40s" % f"Generating frame at {t:.0f}ms"
    update_status.set()
    with lock:
        actual_time = t - args.start_offset * 1000 + args.ss * 1000
        if actual_time < 0:
            frame_key_map[frame_id] = (-1, -1, -1)
            cached_frames[(-1, -1, -1)] = first_frame
            cached_frames.add_reuse((-1, -1, -1))
            send_event.set()
            return
        if actual_time > notes[-1][0] + args.end_offset * 1000:
            raise StopIteration
        bar_idx = bisect.bisect_right(bars, actual_time, key=lambda x: x[0]) - 1
        note_idx = bisect.bisect_right(notes, actual_time, key=lambda x: x[0]) - 1
        if note_idx < 0:
            note_idx = 0
        if bar_idx < 0:
            bar_idx = 0
        # Get current page
        page = bar_pos[bars[bar_idx][1]]["page"]
        if not args.smooth_cursor:
            frame_key = (bar_idx, note_idx, page)
            if frame_key in cached_frames:
                frame_key_map[frame_id] = frame_key
                cached_frames.add_reuse(frame_key)
                send_event.set()
                return
        bar_box, note_box = calc_highlight_pos(actual_time, bar_idx, note_idx, page, args.smooth_cursor)
        if args.smooth_cursor:
            frame_key = (bar_idx, note_idx, page, bar_box, note_box)
        if frame_key in cached_frames:
            frame_key_map[frame_id] = frame_key
            cached_frames.add_reuse(frame_key)
            send_event.set()
            return
        same_frame_queue.setdefault(frame_key, []).append(frame_id)
        if frame_key in pending_frames:
            return
        pending_frames.add(frame_key)
        if args.use_torch:
            # Find the device with the least current jobs
            device_value = min(
                torch_devices.values(),
                key=lambda x: x["current_jobs"] if x["current_jobs"] < x["max_jobs"] else float("inf"),
            )
            device = device_value["device"]
            device_value["current_jobs"] += 1
    # Draw current bar and note
    img = pngs[page].copy() if not args.use_torch else pngs[device_value["name"]][page].to(device).clone()
    program_status = "%40s" % f"Processing note {note_idx+1}/{len(notes)} at {t:.0f}ms"
    update_status.set()

    # Draw bar and note highlights
    if not args.use_torch:
        img, box = resize(PIL.Image.fromarray((img * 255).astype(np.uint8)), *args.size, *note_box)
        img = np.array(img).astype(np.float16) / 255
        overlay = np.zeros_like(img, dtype=np.float16)
        # Draw current bar
        if args.bar_alpha > 0:
            color = webcolors.html5_parse_legacy_color(args.bar_color) + (args.bar_alpha,)
            bar_box_offseted = (
                max(bar_box[0] - box[0], 0),
                max(bar_box[1] - box[1], 0),
                min(bar_box[2] - box[0], args.size[0]),
                min(bar_box[3] - box[1], args.size[1]),
            )
            overlay[bar_box_offseted[1] : bar_box_offseted[3], bar_box_offseted[0] : bar_box_offseted[2]] = color
        # Draw current note
        if args.note_alpha > 0:
            color = webcolors.html5_parse_legacy_color(args.note_color) + (args.note_alpha,)
            note_box_offseted = (
                max(note_box[0] - box[0], 0),
                max(note_box[1] - box[1], 0),
                min(note_box[2] - box[0], args.size[0]),
                min(note_box[3] - box[1], args.size[1]),
            )
            overlay[note_box_offseted[1] : note_box_offseted[3], note_box_offseted[0] : note_box_offseted[2]] = color
        overlay_area = (
            min(bar_box_offseted[1], note_box_offseted[1]),
            max(bar_box_offseted[3], note_box_offseted[3]),
            min(bar_box_offseted[0], note_box_offseted[0]),
            max(bar_box_offseted[2], note_box_offseted[2]),
        )
        if overlay.any():
            overlay /= 255
            overlay[overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3] *= img[
                overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3
            ]
            overlay[overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3] *= overlay[
                overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], 3, None
            ]
            img[overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3] *= (
                1 - overlay[overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], 3, None]
            )
            img[overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3] += overlay[
                overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3
            ]
            img[..., 3] = 1
            img = PIL.Image.fromarray((img * 255).astype(np.uint8))
            out_bytes = img.convert("RGB").tobytes()
    else:
        out_bytes = (
            process_torch(
                img,
                bar_box,
                note_box,
                args.size,
                device,
                torch.tensor(
                    webcolors.html5_parse_legacy_color(args.bar_color) + (args.bar_alpha,),
                    dtype=torch.float16,
                    device=device,
                ),
                torch.tensor(
                    webcolors.html5_parse_legacy_color(args.note_color) + (args.note_alpha,),
                    dtype=torch.float16,
                    device=device,
                ),
            )
            .numpy()
            .tobytes()
        )
    assert (
        len(out_bytes) == args.size[0] * args.size[1] * 3
    ), f"Image shape: {img.shape if args.use_torch else img.size}"

    with lock:
        cached_frames[frame_key] = out_bytes
        for i in same_frame_queue.pop(frame_key, []):
            frame_key_map[i] = frame_key
            cached_frames.add_reuse(frame_key)
            if i == next_to_send:
                send_event.set()
        pending_frames.remove(frame_key)
        if args.use_torch:
            device_value["current_jobs"] -= 1
    return


# Generate video
ffmpeg_command = [
    args.ffmpeg_path,
    "-y",
    "-r",
    str(args.fps),
    "-f",
    "rawvideo",
    "-s",
    f"{args.size[0]}x{args.size[1]}",
    "-pix_fmt",
    "rgb24",
    "-i",
    "-",
    *shlex.split(args.ffmpeg_arg_ext),
    str(args.output_video.resolve()),
]
print("ffmpeg command:", ffmpeg_command, file=sys.stderr)

ffmpeg = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
current_frame = 0
next_to_send = 0
send_event = threading.Event()
stop = False
all_done = False


def worker_thread():
    global current_frame, stop, all_done
    while True:
        with lock:
            t = current_frame
            current_frame += 1
        if t > args.t * args.fps:
            stop = True
            all_done = True
            break
        if stop:
            break
        try:
            get_frame(t, t * 1000 / args.fps)
        except StopIteration:
            stop = True
            all_done = True
            break


def send_thread():
    global next_to_send, stop, cached_frames_old, cached_frames_reusing, frame_reuse_counter
    try:
        while True:
            send_event.wait(1)
            with lock:
                if stop and not all_done and next_to_send not in frame_key_map:
                    break
                if all_done and next_to_send not in frame_key_map:
                    break
                while next_to_send in frame_key_map:
                    key = frame_key_map.pop(next_to_send)
                    if key in cached_frames:
                        ffmpeg.stdin.write(cached_frames[key])
                    ffmpeg.stdin.flush()
                    next_to_send += 1
                    send_event.clear()
    finally:
        ffmpeg.stdin.close()
        stop = True


def ffmpeg_output_thread():
    global ffmpeg_status
    ffmpeg_output = b""
    while True:
        o = ffmpeg.stderr.read(1)
        if not o:
            break
        if o == b"\r":
            if ffmpeg_output.startswith(b"frame=") and not b"Lsize=" in ffmpeg_output:
                ffmpeg_status = ffmpeg_output
            else:
                sys.stderr.buffer.write(ffmpeg_output + b"\n")
                sys.stderr.buffer.flush()
            ffmpeg_output = b""
        elif o == b"\n":
            pass
        else:
            ffmpeg_output += o
    if ffmpeg_output:
        sys.stderr.buffer.write(ffmpeg_output + b"\n")
        sys.stderr.buffer.flush()


def print_thread():
    global program_status, ffmpeg_status
    while True:
        update_status.wait(1)
        print(program_status, end=" ", file=sys.stderr, flush=True)
        sys.stderr.buffer.write(ffmpeg_status + b"     \r")
        sys.stderr.buffer.flush()
        update_status.clear()


try:
    threads = []
    t = threading.Thread(target=send_thread, daemon=True)
    send_thread_obj = t
    t.start()
    threads.append(t)
    t = threading.Thread(target=ffmpeg_output_thread, daemon=True)
    t.start()
    threads.append(t)
    t = threading.Thread(target=print_thread, daemon=True)
    t.start()
    threads.append(t)
    for _ in range(args.jobs):
        t = threading.Thread(target=worker_thread, daemon=True)
        t.start()
        threads.append(t)
    while sum(1 for t in threads[3:] if t.is_alive()) > 0:
        for t in threads:
            t.join(0.1)
    send_thread_obj.join()
    print("\nDone", file=sys.stderr)
except KeyboardInterrupt:
    print("\nInterrupted, stopping...", file=sys.stderr)
    stop = True
    send_thread_obj.join()
finally:
    ffmpeg.stdin.close()
    ffmpeg.wait()
