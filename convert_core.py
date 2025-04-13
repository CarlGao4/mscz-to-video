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

import base64
import bisect
import collections
import io
import itertools
import json
import numpy as np
import packaging.version
import pathlib
import PIL.Image
import PIL.ImageFile
import subprocess
import sys
import threading
import time
import traceback
import typing
import xml.etree.ElementTree as ET
import zipfile

__version__ = "0.3.1"

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = None


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


class Converter:
    def __init__(
        self,
        use_torch: bool = False,
        ffmpeg_path: str = "ffmpeg",
        musescore_path: str = "musescore",
    ):
        self._use_torch = use_torch
        self._ffmpeg_path = ffmpeg_path
        self._musescore_path = musescore_path
        self._convert_lock = threading.Lock()
        self._lock = threading.RLock()
        self._ffmpeg_status = b""
        self._program_status = ""
        self._update_status = threading.Event()
        if self._use_torch:
            global torch, _process_torch, _resize_torch, _direct_resize_to_torch, _resize_and_crop_to_torch
            import torch

            if "_resize_torch" not in globals():

                @torch.jit.script
                def _resize_and_crop_to_torch(
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
                                .to(torch.float16 if "xpu" not in img.device.type else torch.float32)
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
                                .to(torch.float16 if "xpu" not in img.device.type else torch.float32)
                            )
                        width, height = img.shape[1], img.shape[0]
                        x_center = (l + r) / 2 * rescale_ratio
                        x_left = int(x_center - w / 2)
                        if x_left < 0:
                            x_left = 0
                        elif x_left + w > width:
                            x_left = width - w
                        x_right = x_left + w
                        return img[:, x_left:x_right, :], (
                            int(x_left / rescale_ratio),
                            0,
                            int(x_right / rescale_ratio),
                            height,
                        )
                    else:
                        # Target ratio is wider than the image
                        rescale_ratio = w / width
                        if fallback_cpu:
                            img = (
                                torch.nn.functional.interpolate(
                                    img[None, ...].permute(0, 3, 1, 2).cpu().to(torch.float32),
                                    size=(int(height * rescale_ratio), w),
                                    mode="bicubic",
                                    align_corners=False,
                                    antialias=True,
                                )
                                .to(img.device)
                                .permute(0, 2, 3, 1)
                                .squeeze(0)
                                .clamp(0, 1)
                                .to(torch.float16 if "xpu" not in img.device.type else torch.float32)
                            )
                        else:
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
                                .to(torch.float16 if "xpu" not in img.device.type else torch.float32)
                            )
                        width, height = img.shape[1], img.shape[0]
                        y_center = (t + b) / 2 * rescale_ratio
                        y_top = int(y_center - h / 2)
                        if y_top < 0:
                            y_top = 0
                        elif y_top + h > height:
                            y_top = height - h
                        y_bottom = y_top + h
                        return img[y_top:y_bottom, :], (
                            0,
                            int(y_top / rescale_ratio),
                            width,
                            int(y_bottom / rescale_ratio),
                        )

                @torch.jit.script
                def _direct_resize_to_torch(
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
                            .to(torch.float16 if "xpu" not in img.device.type else torch.float32),
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
                            .to(torch.float16 if "xpu" not in img.device.type else torch.float32),
                            (0, 0, img.shape[1], img.shape[0]),
                        )

            _resize_torch = _resize_and_crop_to_torch

            if "_process_torch" not in globals():

                @torch.jit.script
                def _process_torch(
                    img: torch.Tensor,
                    bar_box: tuple[int, int, int, int],
                    note_box: tuple[int, int, int, int],
                    out_size: tuple[int, int],
                    device: torch.device,
                    bar_color: torch.Tensor,
                    note_color: torch.Tensor,
                    fallback_cpu: bool = False,
                ) -> torch.Tensor:
                    img, box = _resize_torch(
                        img,
                        *out_size,
                        *note_box,
                        fallback_cpu=fallback_cpu,
                    )
                    overlay = torch.zeros_like(img, dtype=img.dtype, device=device)
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
                    overlay[bar_box_offseted[1] : bar_box_offseted[3], bar_box_offseted[0] : bar_box_offseted[2]] = (
                        bar_color
                    )
                    # Draw current note
                    overlay[
                        note_box_offseted[1] : note_box_offseted[3], note_box_offseted[0] : note_box_offseted[2]
                    ] = note_color
                    overlay_area = (
                        min(bar_box_offseted[1], note_box_offseted[1]),
                        max(bar_box_offseted[3], note_box_offseted[3]),
                        min(bar_box_offseted[0], note_box_offseted[0]),
                        max(bar_box_offseted[2], note_box_offseted[2]),
                    )
                    if (overlay.any() if "xpu" not in device.type else overlay.cpu().any()).item():
                        overlay /= torch.tensor(255.0, dtype=img.dtype, device=device)
                        overlay[overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3] *= img[
                            overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3
                        ]
                        overlay[overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3] *= overlay[
                            overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], 3, None
                        ]
                        img[overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3] *= (
                            torch.tensor(1.0, dtype=img.dtype, device=device)
                            - overlay[overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], 3, None]
                        )
                        img[overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3] += overlay[
                            overlay_area[0] : overlay_area[1], overlay_area[2] : overlay_area[3], :3
                        ]
                        img[..., 3] = torch.tensor(1.0, dtype=img.dtype, device=device)
                    if "xpu" in device.type:
                        return (
                            (img[..., :3] * torch.tensor(255.0, dtype=img.dtype, device=device)).cpu().to(torch.uint8)
                        )
                    return (img[..., :3] * torch.tensor(255.0, dtype=img.dtype, device=device)).to(torch.uint8).cpu()

    def load_score(self, input_mscz: pathlib.Path, use_svg: bool = False):
        self._input_mscz = input_mscz
        self._use_svg = use_svg
        musescore = subprocess.Popen(
            [
                self._musescore_path,
                "--score-media",
                str(input_mscz.resolve()),
            ],
            stdout=subprocess.PIPE,
        )
        data = json.loads("{" + musescore.communicate()[0].decode("utf-8").split("{", 1)[1])
        # Convert svg to png
        self._pngs: list[np.ndarray] = []
        if self._use_svg:
            import cairosvg  # type: ignore[import]

            for page in data["svgs"]:
                print(f"Converting svg to png... {len(self._pngs)} / {len(data['svgs'])}", end="\r", file=sys.stderr)
                b = io.BytesIO()
                cairosvg.svg2png(bytestring=base64.b64decode(page), write_to=b)
                b.seek(0)
                self._pngs.append(np.array(PIL.Image.open(b).convert("RGBA")).astype(np.float16) / 255)
            print("Converting svg to png... Done          ", file=sys.stderr)
            self._highlight_ratio: tuple[float, float] = (11, 11)
        else:
            for page in data["pngs"]:
                print(f"Reading png... {len(self._pngs)} / {len(data['pngs'])}", end="\r", file=sys.stderr)
                b = io.BytesIO(base64.b64decode(page))
                self._pngs.append(np.array(PIL.Image.open(b).convert("RGBA")).astype(np.float16) / 255)
            print("Reading png... Done          ", file=sys.stderr)
            self._highlight_ratio = (12, 12)

        self._bars: list[tuple[int, int]] = []
        self._notes: list[tuple[int, int]] = []
        self._mposXML = ET.fromstring(base64.b64decode(data["mposXML"]))
        self._sposXML = ET.fromstring(base64.b64decode(data["sposXML"]))
        for bar in self._mposXML.find(".//events"):
            self._bars.append((int(bar.attrib["position"]), int(bar.attrib["elid"])))
        for note in self._sposXML.find(".//events"):
            self._notes.append((int(note.attrib["position"]), int(note.attrib["elid"])))
        self._bars.sort()
        self._notes.sort()

    def _resize_and_crop_to(self, img: PIL.Image.Image, w, h, l, t, r, b):
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

    def _direct_resize_to(self, img: PIL.Image.Image, w, h, l, t, r, b):
        return img.resize((w, h), resample=PIL.Image.BICUBIC), (0, 0, img.width, img.height)

    def _get_frame(self, frame_id, t):
        while len(self._frame_key_map) > self._cache_limit:
            time.sleep(0.1)
        self._program_status = "%40s" % f"Generating frame {frame_id}/{self._total_frames} ({t/1000:.2f}s)"
        self._update_status.set()
        with self._lock:
            actual_time = t - self._start_offset * 1000 + self._ss * 1000
            if actual_time < 0:
                self._frame_key_map[frame_id] = (-1, -1, -1)
                self._cached_frames[(-1, -1, -1)] = self._first_frame
                self._cached_frames.add_reuse((-1, -1, -1))
                self._send_event.set()
                return
            if actual_time > self._notes[-1][0] + self._end_offset * 1000:
                raise StopIteration
            bar_idx = bisect.bisect_right(self._bars, actual_time, key=lambda x: x[0]) - 1
            note_idx = bisect.bisect_right(self._notes, actual_time, key=lambda x: x[0]) - 1
            if note_idx < 0:
                note_idx = 0
            if bar_idx < 0:
                bar_idx = 0
            # Get current page
            page = self._bar_pos[self._bars[bar_idx][1]]["page"]
            if not self._smooth_cursor:
                frame_key = (bar_idx, note_idx, page)
                if frame_key in self._cached_frames:
                    self._frame_key_map[frame_id] = frame_key
                    self._cached_frames.add_reuse(frame_key)
                    self._send_event.set()
                    return
            bar_box, note_box = self._calc_highlight_pos(actual_time, bar_idx, note_idx, page, self._smooth_cursor)
            if self._smooth_cursor:
                frame_key = (bar_idx, note_idx, page, bar_box, note_box)
            if frame_key in self._cached_frames:
                self._frame_key_map[frame_id] = frame_key
                self._cached_frames.add_reuse(frame_key)
                self._send_event.set()
                return
            self._same_frame_queue.setdefault(frame_key, []).append(frame_id)
            if frame_key in self._pending_frames:
                return
            self._pending_frames.add(frame_key)
            if self._use_torch:
                # Find the device with the least current jobs
                device_value = min(
                    self._torch_devices.values(),
                    key=lambda x: x["current_jobs"] if x["current_jobs"] < x["max_jobs"] else float("inf"),
                )
                device = device_value["device"]
                device_value["current_jobs"] += 1
        # Draw current bar and note
        img = (
            self._pngs[page].copy()
            if not self._use_torch
            else self._pngs[device_value["name"]][page].to(device).clone()
        )

        # Draw bar and note highlights
        if not self._use_torch:
            img, box = self._resize(PIL.Image.fromarray((img * 255).astype(np.uint8)), *self._size, *note_box)
            img = np.array(img).astype(np.float16) / 255
            overlay = np.zeros_like(img, dtype=np.float16)
            # Draw current bar
            if self._bar_alpha > 0:
                color = self._bar_color + (self._bar_alpha,)
                bar_box_offseted = (
                    max(bar_box[0] - box[0], 0),
                    max(bar_box[1] - box[1], 0),
                    min(bar_box[2] - box[0], self._size[0]),
                    min(bar_box[3] - box[1], self._size[1]),
                )
                overlay_area = bar_box_offseted
                overlay[bar_box_offseted[1] : bar_box_offseted[3], bar_box_offseted[0] : bar_box_offseted[2]] = color
            # Draw current note
            if self._note_alpha > 0:
                color = self._note_color + (self._note_alpha,)
                note_box_offseted = (
                    max(note_box[0] - box[0], 0),
                    max(note_box[1] - box[1], 0),
                    min(note_box[2] - box[0], self._size[0]),
                    min(note_box[3] - box[1], self._size[1]),
                )
                overlay_area = note_box_offseted
                overlay[note_box_offseted[1] : note_box_offseted[3], note_box_offseted[0] : note_box_offseted[2]] = (
                    color
                )
            if self._bar_alpha > 0 and self._note_alpha > 0:
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
                _process_torch(
                    img,
                    bar_box,
                    note_box,
                    self._size,
                    device,
                    torch.tensor(
                        self._bar_color + (self._bar_alpha,),
                        dtype=img.dtype,
                        device=device,
                    ),
                    torch.tensor(
                        self._note_color + (self._note_alpha,),
                        dtype=img.dtype,
                        device=device,
                    ),
                    "cpu" not in device.type
                    and "cuda" not in device.type
                    and packaging.version.Version(torch.__version__) < packaging.version.Version("2.5"),
                )
                .numpy()
                .tobytes()
            )
        assert (
            len(out_bytes) == self._size[0] * self._size[1] * 3
        ), f"Image shape: {img.shape if self._use_torch else img.size}"

        with self._lock:
            self._cached_frames[frame_key] = out_bytes
            for i in self._same_frame_queue.pop(frame_key, []):
                self._frame_key_map[i] = frame_key
                self._cached_frames.add_reuse(frame_key)
                if i == self._next_to_send:
                    self._send_event.set()
            self._pending_frames.remove(frame_key)
            if self._use_torch:
                device_value["current_jobs"] -= 1
        return

    def _calc_highlight_pos(
        self,
        t: float,
        bar_idx: int,
        note_idx: int,
        page: int,
        smooth_cursor: bool,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        note = self._note_pos[self._notes[note_idx][1]]
        bar = self._bar_pos[self._bars[bar_idx][1]]
        bar_box = (bar["x"], bar["y"], bar["x"] + bar["width"], bar["y"] + bar["height"])
        if not smooth_cursor:
            note_box = (note["x"], note["y"], note["x"] + note["width"], note["y"] + note["height"])
        else:
            if note_idx + 1 >= len(self._notes):
                note_box = (note["x"], note["y"], note["x"] + note["width"], note["y"] + note["height"])
            else:
                next_note = self._note_pos[self._notes[note_idx + 1][1]]
                next_page = next_note["page"]
                next_bar_idx = bisect.bisect_right(self._bars, self._notes[note_idx + 1][0], key=lambda x: x[0]) - 1
                note_box_current = (note["x"], note["y"], note["x"] + note["width"], note["y"] + note["height"])
                note_box_next = (
                    next_note["x"],
                    next_note["y"],
                    next_note["x"] + next_note["width"],
                    next_note["y"] + next_note["height"],
                )
                if (
                    page != next_page
                    or note_box_next[1] >= note_box_current[3]
                    or note_box_next[2] <= note_box_current[0]
                    or note_box_next[3] <= note_box_current[1]
                    or not (self._bars[bar_idx][1] <= self._bars[next_bar_idx][1] <= self._bars[bar_idx][1] + 1)
                ):
                    note_box_next = (
                        note["x"] + bar_box[2] - note_box_current[2],
                        note["y"],
                        bar_box[2],
                        note["y"] + note["height"],
                    )
                current_note_time = self._notes[note_idx][0]
                next_note_time = self._notes[note_idx + 1][0]
                note_box = tuple(
                    int(
                        (note_box_current[i] * (next_note_time - t) + note_box_next[i] * (t - current_note_time))
                        / (next_note_time - current_note_time)
                    )
                    for i in range(4)
                )
        return bar_box, note_box

    def convert(
        self,
        output_path: pathlib.Path,
        cache_limit: int = 100,
        smooth_cursor: bool = False,
        fixed_note_width: typing.Optional[typing.Union[int, float]] = None,
        extra_note_width_ratio: float = 0.0,
        size: typing.Optional[tuple[int, int]] = None,
        bar_alpha: int = 85,
        bar_color: tuple[int, int, int] = (255, 0, 0),
        note_alpha: int = 85,
        note_color: tuple[int, int, int] = (0, 255, 255),
        start_offset: float = 0.0,
        end_offset: float = 0.0,
        ss: float = 0.0,
        jobs: int = 1,
        resize_method: typing.Union[typing.Literal["crop", "rescale"]] = "rescale",
        torch_devices: str = "cpu",
        fps: int = 60,
        t: float = float("inf"),
        no_device_cache: bool = False,
        ffmpeg_arg_ext: list[str] = [],
        callback: typing.Optional[typing.Callable[[int, int, int, int, bytes], None]] = None,
    ):

        with self._convert_lock:
            self._cache_limit = cache_limit
            self._smooth_cursor = smooth_cursor
            self._fixed_note_width = fixed_note_width
            self._extra_note_width_ratio = extra_note_width_ratio
            self._size = size
            self._bar_alpha = bar_alpha
            self._bar_color = bar_color
            self._note_alpha = note_alpha
            self._note_color = note_color
            self._start_offset = start_offset
            self._end_offset = end_offset
            self._ss = ss
            self._jobs = jobs
            self._resize_method = resize_method
            self._callback = callback

            if self._use_torch:
                global _resize_torch
                self._torch_devices = {}
                if "xpu" in torch_devices and packaging.version.Version(torch.__version__) < packaging.version.Version(
                    "2.2"
                ):
                    import intel_extension_for_pytorch as ipex  # type: ignore[import]
                for device in torch_devices.split(";"):
                    if "," in device:
                        device, max_jobs = device.split(",")
                        max_jobs = int(max_jobs)
                        self._torch_devices[device] = {
                            "device": torch.device(device),
                            "max_jobs": max_jobs,
                            "current_jobs": 0,
                            "name": device,
                        }
                    else:
                        self._torch_devices[device] = {
                            "device": torch.device(device),
                            "max_jobs": float("inf"),
                            "current_jobs": 0,
                            "name": device,
                        }

                if sum(d["max_jobs"] for d in self._torch_devices.values()) < self._jobs:
                    raise ValueError("Not enough max jobs in torch devices")

                if resize_method == "crop":
                    _resize_torch = _resize_and_crop_to_torch
                elif resize_method == "rescale":
                    _resize_torch = _direct_resize_to_torch
                else:
                    raise ValueError("Invalid resize method")

            if resize_method == "crop":
                self._resize = self._resize_and_crop_to
            elif resize_method == "rescale":
                self._resize = self._direct_resize_to
            else:
                raise ValueError("Invalid resize method")

            if self._size is None:
                self._size = self._pngs[0].shape[1], self._pngs[0].shape[0]

            self._fps = fps
            self._t = t
            ffmpeg_command = [
                self._ffmpeg_path,
                "-y",
                "-r",
                str(self._fps),
                "-f",
                "rawvideo",
                "-s",
                f"{self._size[0]}x{self._size[1]}",
                "-pix_fmt",
                "rgb24",
                "-i",
                "-",
                *ffmpeg_arg_ext,
                str(output_path.resolve()),
            ]
            print("ffmpeg command:", ffmpeg_command, file=sys.stderr)

            self._ffmpeg = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            self._current_frame = 0
            self._next_to_send = 0
            self._send_event = threading.Event()
            self._stop = False
            self._all_done = False
            self._total_frames = (
                int(
                    (
                        self._t
                        if self._t != float("inf")
                        else (self._notes[-1][0] / 1000 + self._start_offset + self._end_offset - self._ss)
                    )
                    * self._fps
                )
                + 1
            )

            print("Exported page size", tuple(reversed(self._pngs[0].shape[:2])), file=sys.stderr)
            print("Rescaling MuseScore pages...", end="\r", file=sys.stderr)
            if self._resize_method == "rescale":
                self._highlight_ratio = (
                    self._highlight_ratio[0] / self._size[0] * self._pngs[0].shape[1],
                    self._highlight_ratio[1] / self._size[1] * self._pngs[0].shape[0],
                )
                self._pngs = [
                    (
                        print(f"Rescaling MuseScore pages... {i} / {len(self._pngs)}", end="\r", file=sys.stderr),
                        np.array(
                            self._direct_resize_to(
                                PIL.Image.fromarray((png * 255).astype(np.uint8)), *self._size, 0, 0, 0, 0
                            )[0]
                        ).astype(np.float16)
                        / 255,
                    )[1]
                    for (i, png) in enumerate(self._pngs)
                ]
            else:
                if self._pngs[0].shape[1] / self._size[0] > self._pngs[0].shape[0] / self._size[1]:
                    # Target ratio is taller than the image
                    rescale_ratio = self._size[1] / self._pngs[0].shape[0]
                else:
                    # Target ratio is wider than the image
                    rescale_ratio = self._size[0] / self._pngs[0].shape[1]
                self._pngs = [
                    (
                        print(f"Rescaling MuseScore pages... {i} / {len(self._pngs)}", end="\r", file=sys.stderr),
                        np.array(
                            self._direct_resize_to(
                                PIL.Image.fromarray((png * 255).astype(np.uint8)),
                                int(self._pngs[0].shape[1] * rescale_ratio),
                                int(self._pngs[0].shape[0] * rescale_ratio),
                                0,
                                0,
                                0,
                                0,
                            )[0]
                        ).astype(np.float16)
                        / 255,
                    )[1]
                    for (i, png) in enumerate(self._pngs)
                ]
                self._highlight_ratio = (
                    self._highlight_ratio[0] / rescale_ratio,
                    self._highlight_ratio[1] / rescale_ratio,
                )
            print("Rescaling MuseScore pages... Done          ", file=sys.stderr)

            # Get bars and notes positions
            self._bar_pos: dict[int, dict[str, int]] = {}
            self._note_pos: dict[int, dict[str, int]] = {}
            for bar in self._mposXML.find(".//elements"):
                self._bar_pos[int(bar.attrib["id"])] = {
                    "x": int(float(bar.attrib["x"]) / self._highlight_ratio[0]),
                    "y": int(float(bar.attrib["y"]) / self._highlight_ratio[1]),
                    "width": int(float(bar.attrib["sx"]) / self._highlight_ratio[0]),
                    "height": int(float(bar.attrib["sy"]) / self._highlight_ratio[1]),
                    "page": int(bar.attrib["page"]),
                }
            for note in self._sposXML.find(".//elements"):
                self._note_pos[int(note.attrib["id"])] = {
                    "x": int(float(note.attrib["x"]) / self._highlight_ratio[0]),
                    "y": int(float(note.attrib["y"]) / self._highlight_ratio[1]),
                    "width": int(float(note.attrib["sx"]) / self._highlight_ratio[0]),
                    "height": int(float(note.attrib["sy"]) / self._highlight_ratio[1]),
                    "page": int(note.attrib["page"]),
                }
            if self._fixed_note_width is not None:
                if self._fixed_note_width == 0:
                    print("Getting fixed note width because it is set to 0", file=sys.stderr)
                    found_width = False
                    # First try reading with xml, assuming input file is mscx
                    try:
                        with open(self._input_mscz, "rb") as f:
                            decoded_mscx = ET.fromstring(f.read())
                            # Just in case the mscx is not capitalized
                            for spatium_text in (
                                "".join(comb)
                                for comb in itertools.product(*((char.lower(), char.upper()) for char in "spatium"))
                            ):
                                spatium_element = decoded_mscx.find(f".//{spatium_text}")
                                if spatium_element is not None:
                                    break
                            if spatium_element is None:
                                raise ValueError("Spatium element not found")
                            self._fixed_note_width = float(spatium_element.text)  # Unit: mm
                            found_width = True
                    except ET.ParseError:
                        pass
                    except Exception:
                        print("Error reading mscx file", file=sys.stderr)
                        print(traceback.format_exc(), file=sys.stderr)
                    if not found_width:
                        # Now we assume the input file is mscz, unzip it and read the mscx file first
                        with zipfile.ZipFile(self._input_mscz, "r") as zf:
                            for name in zf.namelist():
                                if name.lower().endswith(".mscx"):
                                    with zf.open(name) as f:
                                        try:
                                            decoded_mscx = ET.fromstring(f.read())
                                        except ET.ParseError:
                                            continue
                                        # Just in case the mscx is not capitalized
                                        for spatium_text in (
                                            "".join(comb)
                                            for comb in itertools.product(
                                                *((char.lower(), char.upper()) for char in "spatium"))
                                        ):
                                            spatium_element = decoded_mscx.find(f".//{spatium_text}")
                                            if spatium_element is not None:
                                                break
                                        if spatium_element is None:
                                            continue
                                        self._fixed_note_width = float(spatium_element.text)  # Unit: mm
                                        found_width = True
                                        break
                            for name in zf.namelist():
                                if not found_width and name.lower().endswith(".mss"):  # For MuseScore Studio 4
                                    with zf.open(name) as f:
                                        try:
                                            decoded_mscx = ET.fromstring(f.read())
                                        except ET.ParseError:
                                            continue
                                        # Just in case the mscx is not capitalized
                                        for spatium_text in (
                                            "".join(comb)
                                            for comb in itertools.product(
                                                *((char.lower(), char.upper()) for char in "spatium"))
                                        ):
                                            spatium_element = decoded_mscx.find(f".//{spatium_text}")
                                            if spatium_element is not None:
                                                break
                                        if spatium_element is None:
                                            continue
                                        self._fixed_note_width = float(spatium_element.text)  # Unit: mm
                                        found_width = True
                    if not found_width:
                        print("Spatium element not found in input file. Using default value of 1.75", file=sys.stderr)
                        self._fixed_note_width = 1.75
                    else:
                        print("Found spatium", self._fixed_note_width, file=sys.stderr)
                    # Calc actual note width in pixels
                    # 325 / 265 is ratio of single note head
                    # 265 / 252 is ratio of note height to staff line space
                    # 1.008 is ratio of note font size in MuseScore
                    # 25.4 is mm to inch
                    # 330 is default png dpi in MuseScore
                    self._fixed_note_width *= 325 / 265 * 265 / 252 * 1.008 / 25.4 * 330 / self._highlight_ratio[0] * 10
                for note in self._note_pos.values():
                    note["x"] = int(note["x"] - self._fixed_note_width * self._extra_note_width_ratio / 2)
                    note["width"] = int(self._fixed_note_width * (1 + self._extra_note_width_ratio))
            print("Bar count:", len(self._bars), file=sys.stderr)
            print("Note time count:", len(self._notes), file=sys.stderr)

            self._pending_frames = set()
            self._cached_frames = CacheWithQueue(self._cache_limit)
            self._same_frame_queue = {}
            self._frame_key_map = {}
            self._first_frame = (
                self._resize(
                    PIL.Image.fromarray((self._pngs[0].copy() * 255).astype(np.uint8)), *self._size, 0, 0, 0, 0
                )[0]
                .convert("RGB")
                .tobytes()
            )

            if self._use_torch:
                if no_device_cache:
                    pngs_torch = [torch.from_numpy(png) for png in self._pngs]
                    self._pngs = {
                        device: (pngs_torch if "xpu" not in device else [png.to(torch.float32) for png in pngs_torch])
                        for device in self._torch_devices
                    }
                else:
                    self._pngs = {
                        device: [
                            torch.from_numpy(png)
                            .to(torch.float32 if "xpu" in device else torch.float16)
                            .to(self._torch_devices[device]["device"])
                            for png in self._pngs
                        ]
                        for device in self._torch_devices
                    }

            try:
                threads = []
                t = threading.Thread(target=self._send_thread, daemon=True)
                send_thread_obj = t
                t.start()
                threads.append(t)
                t = threading.Thread(target=self._ffmpeg_output_thread, daemon=True)
                t.start()
                threads.append(t)
                t = threading.Thread(target=self._print_thread, daemon=True)
                t.start()
                threads.append(t)
                for _ in range(self._jobs):
                    t = threading.Thread(target=self._worker_thread, daemon=True)
                    t.start()
                    threads.append(t)
                while sum(1 for t in threads[3:] if t.is_alive()) > 0:
                    for t in threads:
                        t.join(0.1)
                send_thread_obj.join()
                print("\nDone", file=sys.stderr)
            except KeyboardInterrupt:
                print("\nInterrupted, stopping...", file=sys.stderr)
                self._stop = True
                send_thread_obj.join()
            finally:
                try:
                    self._ffmpeg.stdin.close()
                    self._ffmpeg.wait()
                except Exception:
                    pass

    def _worker_thread(self):
        while True:
            with self._lock:
                t = self._current_frame
                self._current_frame += 1
            if t > min(self._t * self._fps, self._total_frames):
                self._stop = True
                self._all_done = True
                break
            if self._stop:
                break
            try:
                self._get_frame(t, t * 1000 / self._fps)
            except StopIteration:
                self._stop = True
                self._all_done = True
                break

    def _send_thread(self):
        try:
            while True:
                self._send_event.wait(1)
                with self._lock:
                    if self._stop and not self._all_done and self._next_to_send not in self._frame_key_map:
                        break
                    if self._all_done and self._next_to_send not in self._frame_key_map:
                        break
                    while self._next_to_send in self._frame_key_map:
                        key = self._frame_key_map.pop(self._next_to_send)
                        frame = self._cached_frames[key]
                        if key in self._cached_frames:
                            self._ffmpeg.stdin.write(frame)
                        self._ffmpeg.stdin.flush()
                        if self._callback:
                            self._callback(self._next_to_send, self._total_frames, *self._size, frame)
                        self._next_to_send += 1
                        self._send_event.clear()
        finally:
            self._ffmpeg.stdin.close()
            self._stop = True
            self._all_done = True

    def _ffmpeg_output_thread(self):
        ffmpeg_output = b""
        just_newline = False
        while True:
            if self._stop:
                break
            o = self._ffmpeg.stderr.read(1)
            if not o:
                break
            if o == b"\r":
                just_newline = False
                if ffmpeg_output.startswith(b"frame=") and not b"Lsize=" in ffmpeg_output:
                    self._ffmpeg_status = ffmpeg_output
                else:
                    sys.stderr.buffer.write(ffmpeg_output + b"\n")
                    sys.stderr.buffer.flush()
                    just_newline = True
                ffmpeg_output = b""
            elif o == b"\n" and (ffmpeg_output and ffmpeg_output[-1] < 128 or not ffmpeg_output):
                if just_newline:
                    just_newline = False
                else:
                    ffmpeg_output += b"\n"
                    sys.stderr.buffer.write(ffmpeg_output)
                    sys.stderr.buffer.flush()
                    ffmpeg_output = b""
            else:
                just_newline = False
                ffmpeg_output += o
        if ffmpeg_output:
            sys.stderr.buffer.write(ffmpeg_output + b"\n")
            sys.stderr.buffer.flush()

    def _print_thread(self):
        while True:
            self._update_status.wait(1)
            sys.stderr.buffer.write(self._program_status.encode("utf-8") + b" " + self._ffmpeg_status + b"     \r")
            sys.stderr.buffer.flush()
            self._update_status.clear()
            if self._stop:
                break
