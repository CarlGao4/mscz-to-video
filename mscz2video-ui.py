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

import collections
import functools
import logging
import numpy as np
import os
import packaging.version
import pathlib
import platform
import psutil
from PySide6 import QtGui
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QStyleFactory,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6_modified import TextWrappedQLabel
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
import webcolors

import convert_core

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS") and sys.platform == "win32":
    # Popen should be wrapped to avoid WinError 50
    subprocess._Popen = subprocess.Popen

    class wrapped_Popen(subprocess._Popen):
        def __init__(self, *args, **kwargs):
            if "stdout" in kwargs and kwargs["stdout"] is not None:
                if "stderr" not in kwargs or kwargs["stderr"] is None:
                    kwargs["stderr"] = subprocess.PIPE
                if "stdin" not in kwargs or kwargs["stdin"] is None:
                    kwargs["stdin"] = subprocess.PIPE
            if "stderr" in kwargs and kwargs["stderr"] is not None:
                if "stdout" not in kwargs or kwargs["stdout"] is None:
                    kwargs["stdout"] = subprocess.PIPE
                if "stdin" not in kwargs or kwargs["stdin"] is None:
                    kwargs["stdin"] = subprocess.PIPE
            if "stdin" in kwargs and kwargs["stdin"] is not None:
                if "stdout" not in kwargs or kwargs["stdout"] is None:
                    kwargs["stdout"] = subprocess.PIPE
                if "stderr" not in kwargs or kwargs["stderr"] is None:
                    kwargs["stderr"] = subprocess.PIPE
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            super().__init__(*args, **kwargs)

    subprocess.Popen = wrapped_Popen


def thread_wrapper(*args_thread, no_log=False, **kw_thread):
    if "target" in kw_thread:
        kw_thread.pop("target")
    if "args" in kw_thread:
        kw_thread.pop("args")
    if "kwargs" in kw_thread:
        kw_thread.pop("kwargs")

    def thread_func_wrapper(func):
        if not hasattr(thread_wrapper, "index"):
            thread_wrapper.index = 0

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            thread_wrapper.index += 1
            stack = "".join(traceback.format_list(traceback.extract_stack()[:-1]))

            def run_and_log(idx=thread_wrapper.index):
                if not no_log:
                    logging.info(
                        "[%d] Thread %s (%s) starts"
                        % (idx, func.__name__, pathlib.Path(func.__code__.co_filename).name)
                    )
                try:
                    func(*args, **kwargs)
                except Exception:
                    logging.error(
                        "[%d] Thread %s (%s) failed:\n%s%s"
                        % (
                            idx,
                            func.__name__,
                            pathlib.Path(func.__code__.co_filename).name,
                            stack,
                            traceback.format_exc(),
                        )
                    )
                finally:
                    if not no_log:
                        logging.info(
                            "[%d] Thread %s (%s) ends"
                            % (idx, func.__name__, pathlib.Path(func.__code__.co_filename).name)
                        )

            t = threading.Thread(target=run_and_log, *args_thread, **kw_thread)
            t.start()
            return t

        return wrapper

    return thread_func_wrapper


class VirtualTerminal:
    def __init__(self, callback, send_to_stderr=False):
        self.callback = callback
        self.data_buffer = ""
        self.closed = False
        self.end_with_r = False
        self.send_to_stderr = send_to_stderr

    def write(self, data):
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if self.send_to_stderr:
            sys.__stderr__.write(data)
        len_before = len(self.data_buffer)
        if self.end_with_r:
            self.data_buffer += "\r"
        self.end_with_r = False
        data = re.sub(r"\r+", "\r", data)
        if not self.data_buffer:
            data = re.sub(r"\r*\n", "\n", data)
            if "\r" in data:
                if data[-1] != "\r":
                    for line in data.split("\n"):
                        self.data_buffer += line.rsplit("\r", 1)[-1] + "\n"
                    self.data_buffer = self.data_buffer[:-1]
                else:
                    for line in data.split("\n"):
                        self.data_buffer += line.rsplit("\r", 1)[-1] + "\n"
                    self.data_buffer = self.data_buffer[:-1] + line.rsplit("\r", 2)[-2]
                    self.end_with_r = True
            else:
                self.data_buffer += data
        else:
            last_newline = self.data_buffer.rsplit("\n", 1)
            if len(last_newline) == 1:
                data = self.data_buffer + re.sub(r"\r*\n", "\n", data)
                self.data_buffer = ""
            else:
                self.data_buffer = last_newline[0] + "\n"
                data = last_newline[1] + re.sub(r"\r*\n", "\n", data)
            if "\r" in data:
                if data[-1] != "\r":
                    for line in data.split("\n"):
                        self.data_buffer += line.rsplit("\r", 1)[-1] + "\n"
                    self.data_buffer = self.data_buffer[:-1]
                else:
                    for line in data.split("\n"):
                        self.data_buffer += line.rsplit("\r", 1)[-1] + "\n"
                    self.data_buffer = self.data_buffer[:-1] + line.rsplit("\r", 2)[-2]
                    self.end_with_r = True
            else:
                self.data_buffer += data
        if "\n" in self.data_buffer or "\r" in self.data_buffer:
            try:
                self.callback(self.data_buffer)
            except Exception:
                return 0
        return len(self.data_buffer) - len_before

    def flush(self):
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if self.send_to_stderr:
            sys.__stderr__.flush()
        try:
            self.callback(self.data_buffer)
        except Exception:
            pass

    @property
    def buffer(self):
        obj = collections.namedtuple("buffer", ["write", "flush", "closed"])
        obj.write = lambda x: self.write(x.decode(errors="replace"))
        obj.flush = lambda: (sys.__stderr__.buffer.flush() if self.send_to_stderr else None)
        obj.closed = self.closed
        return obj

    def close(self):
        self.closed = True


class MainWindow(QWidget):
    _execInMainThreadSignal = Signal()
    _execInMainThreadFunc = None
    _execInMainThreadResult = None
    _execInMainThreadSuccess = False
    _execInMainThreadLock = threading.Lock()
    _execInMainThreadResultEvent = threading.Event()

    def __init__(self):
        super().__init__()
        self._execInMainThreadSignal.connect(self._exec_in_main_thread_executor)
        self.setWindowTitle("mscz-to-video")
        self.setWindowIcon(QtGui.QIcon(str(pathlib.Path(__file__).parent / "icon/icon.ico")))

        self.preview_lock = threading.Lock()
        self.update_preview_event = threading.Event()

        self.preview_frame = QWidget(self)
        self.preview_frame.setObjectName("preview_frame")
        self.preview_frame.setStyleSheet("#preview_frame{border: 2px dotted #888;}")
        self.preview_frame.setMinimumSize(240, 160)
        self.preview_window = QLabel(self)
        self.preview_window.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.preview_window.setText("Start rendering to show preview")
        self.preview_window.setWordWrap(True)
        self.preview_window.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_window.resizeEvent = lambda e: self.update_preview_event.set()
        self.render_button = QPushButton("Render", self)
        self.render_button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.render_button.setDisabled(True)
        self.render_button.setToolTip("Please load MuseScore file first")
        self.render_button.clicked.connect(self.render)
        self.files_frame = QWidget(self)
        self.files_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.controls_frame = QWidget(self)
        self.controls_frame.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.progress_bar.setTextVisible(False)
        self.progess_text = QLabel(self)
        self.progess_text.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.progess_text.setText("No job running")
        self.close_show_log = QPushButton("Show log", self)
        self.close_show_log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.close_show_log.clicked.connect(self.toggle_log)
        self.log_text = QTextEdit(self)
        self.log_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("No log available")
        self.log_text.setWordWrapMode(QtGui.QTextOption.WrapMode.NoWrap)
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.log_text.setFixedHeight(100)
        self.log_text.keyPressEvent = lambda e: None
        self.log_text.inputMethodEvent = lambda e: None
        self.log_text.setFontFamily("Consolas" if sys.platform == "win32" else "Courier")

        def show_context_menu(pos):
            menu = self.log_text.createStandardContextMenu()
            menu.addSeparator()
            copy_all_action = menu.addAction("Copy All")
            copy_all_action.triggered.connect(lambda: QApplication.clipboard().setText(self.log_text.toPlainText()))
            menu.exec(self.log_text.mapToGlobal(pos))

        self.log_text.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.log_text.customContextMenuRequested.connect(show_context_menu)
        self.log = ""

        self.stop_button_frame = QWidget(self)
        self.stop_button_frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.stop_button = QPushButton("Stop", self.stop_button_frame)
        self.stop_button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.stop_button.clicked.connect(self.ask_stop)

        self.load_mscz_button = QPushButton("Load MuseScore file", self)
        self.load_mscz_button.clicked.connect(self.ask_mscz)
        self.load_mscz_button.setDisabled(True)
        self.load_mscz_button.setToolTip("Please wait while loading devices")
        self.load_mscz_button.setAcceptDrops(True)
        self.load_mscz_button.dragEnterEvent = lambda e: (
            e.accept() if e.mimeData().hasUrls() and self.load_mscz_button.isEnabled() else None
        )
        self.load_mscz_button.dropEvent = lambda e: self.load_mscz(e.mimeData().urls()[0].toLocalFile())
        self.current_mscz_label = TextWrappedQLabel("No MuseScore file loaded", self)
        self.load_audio_button = QPushButton("Select audio file", self)
        self.load_audio_button.clicked.connect(self.select_audio)
        self.load_audio_button.setAcceptDrops(True)
        self.load_audio_button.dragEnterEvent = lambda e: (
            e.accept() if e.mimeData().hasUrls() and self.load_audio_button.isEnabled() else None
        )
        self.load_audio_button.dropEvent = lambda e: self.current_audio_label.setText(
            e.mimeData().urls()[0].toLocalFile()
        )
        self.clear_audio_button = QPushButton("Clear audio file", self)
        self.clear_audio_button.clicked.connect(self.clear_audio)
        self.current_audio_label = TextWrappedQLabel("No audio file selected", self)
        self.audio_delay_label = QLabel("Audio delay:", self)
        self.audio_delay_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.audio_delay_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.audio_delay_label.setToolTip("Delay of audio will not be automatically adjusted according to start offset")
        self.audio_delay = QDoubleSpinBox(self)
        self.audio_delay.setRange(-1000, 1000)
        self.audio_delay.setValue(0)
        self.audio_delay.setSingleStep(0.01)
        self.audio_delay.setDecimals(2)
        self.audio_delay.setSuffix(" s")
        self.video_encoder_label = QLabel("Video encoder:", self)
        self.video_encoder_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.video_encoder_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.video_encoder = QComboBox(self)
        self.video_encoder.addItem("libx264")
        self.video_encoder.addItem("libx265")
        if sys.platform != "darwin":
            self.video_encoder.addItem("h264_nvenc")
            self.video_encoder.addItem("hevc_nvenc")
            self.video_encoder.addItem("av1_nvenc")
            self.video_encoder.addItem("h264_qsv")
            self.video_encoder.addItem("hevc_qsv")
            self.video_encoder.addItem("av1_qsv")
            self.video_encoder.addItem("vp9_qsv")
            self.video_encoder.addItem("h264_amf")
            self.video_encoder.addItem("hevc_amf")
            self.video_encoder.addItem("av1_amf")
        else:
            self.video_encoder.addItem("h264_videotoolbox")
            self.video_encoder.addItem("hevc_videotoolbox")
            self.video_encoder.addItem("prores_videotoolbox")
        self.video_encoder.addItem("libvpx-vp9")
        self.video_encoder.addItem("libaom-av1")
        self.video_encoder.addItem("libsvtav1")
        self.video_encoder.addItem("prores")
        self.video_encoder_method_label = QLabel("Video bitrate control:", self)
        self.video_encoder_method_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.video_encoder_method_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.video_encoder_method = QComboBox(self)
        self.video_encoder_method.addItem("VBR")
        self.video_encoder_method.addItem("CQP")
        self.video_encoder_method.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.video_bitrate = QSpinBox(self)
        self.video_bitrate.setSingleStep(1)
        self.video_bitrate.setAccelerated(True)
        self.video_bitrate.setSuffix(" kbps")
        self.video_encoder_method.currentIndexChanged.connect(self.switch_video_bitrate_range)
        self.switch_video_bitrate_range()
        self.audio_encoder_label = QLabel("Audio encoder:", self)
        self.audio_encoder_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.audio_encoder_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.audio_encoder = QComboBox(self)
        self.audio_encoder.addItem("aac")
        self.audio_encoder.addItem("libopus")
        self.audio_encoder.addItem("flac")
        self.audio_encoder.addItem("pcm_s16le")
        self.audio_encoder.addItem("pcm_s24le")
        self.audio_encoder.addItem("alac")
        self.audio_encoder.addItem("mp3")
        self.audio_encoder.addItem("vorbis")
        self.audio_bitrate = QSpinBox(self)
        self.audio_bitrate.setRange(1, 3072)
        self.audio_bitrate.setValue(192)
        self.audio_bitrate.setSingleStep(1)
        self.audio_bitrate.setAccelerated(True)
        self.audio_bitrate.setSuffix(" kbps")

        self.size_label = QLabel("Size:", self)
        self.size_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.size_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.size_x = QSpinBox(self)
        self.size_x.setRange(1, 16384)
        self.size_x.setValue(1920)
        self.size_x.setSingleStep(1)
        self.size_x.setAccelerated(True)
        self.size_multiplier = QLabel("×", self)
        self.size_multiplier.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.size_multiplier.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.size_y = QSpinBox(self)
        self.size_y.setRange(1, 16384)
        self.size_y.setValue(1080)
        self.size_y.setSingleStep(1)
        self.size_y.setAccelerated(True)
        self.size_label.setBuddy(self.size_x)
        self.fps_label = QLabel("FPS:", self)
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.fps_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.fps = QSpinBox(self)
        self.fps.setRange(1, 240)
        self.fps.setValue(60)
        self.fps.setSingleStep(1)
        self.fps.setAccelerated(True)
        self.fps.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.fps_label.setBuddy(self.fps)
        self.bar_color_label_container = QWidget(self)
        self.bar_color_label_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.bar_color_label_container.setStyleSheet("background-color: #FFFFFFFF;")
        self.bar_color_label = QLabel(" Bar color & alpha: ", self)
        self.bar_color_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.bar_color_label.setStyleSheet("background-color: #55FF0000; color: black; padding: 3px;")
        self.bar_color = QPushButton("Choose color", self)
        self.bar_color.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.bar_color.clicked.connect(lambda: self.update_bar_color(True))
        self.bar_alpha = QSpinBox(self)
        self.bar_alpha.setRange(0, 255)
        self.bar_alpha.setValue(85)
        self.bar_alpha.setSingleStep(1)
        self.bar_alpha.setAccelerated(True)
        self.bar_alpha.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.bar_alpha.valueChanged.connect(lambda: self.update_bar_color(False))
        self.note_color_label_container = QWidget(self)
        self.note_color_label_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.note_color_label_container.setStyleSheet("background-color: #FFFFFFFF;")
        self.note_color_label = QLabel(" Note color & alpha: ", self)
        self.note_color_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.note_color_label.setStyleSheet("background-color: #5500FFFF; color: black; padding: 3px;")
        self.note_color = QPushButton("Choose color", self)
        self.note_color.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.note_color.clicked.connect(lambda: self.update_note_color(True))
        self.note_alpha = QSpinBox(self)
        self.note_alpha.setRange(0, 255)
        self.note_alpha.setValue(85)
        self.note_alpha.setSingleStep(1)
        self.note_alpha.setAccelerated(True)
        self.note_alpha.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.note_alpha.valueChanged.connect(lambda: self.update_note_color(False))
        self.start_offset_label = QLabel("Start offset:", self)
        self.start_offset_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.start_offset_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.start_offset = QDoubleSpinBox(self)
        self.start_offset.setRange(0, 100)
        self.start_offset.setValue(1)
        self.start_offset.setSingleStep(0.1)
        self.start_offset.setDecimals(1)
        self.start_offset.setSuffix(" s")
        self.start_offset.setAccelerated(True)
        self.start_offset.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.end_offset_label = QLabel("End offset:", self)
        self.end_offset_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.end_offset_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.end_offset = QDoubleSpinBox(self)
        self.end_offset.setRange(0, 100)
        self.end_offset.setValue(5)
        self.end_offset.setSingleStep(0.1)
        self.end_offset.setDecimals(1)
        self.end_offset.setSuffix(" s")
        self.end_offset.setAccelerated(True)
        self.end_offset.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.from_label = QLabel("From:", self)
        self.from_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.from_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.from_time = QDoubleSpinBox(self)
        self.from_time.setRange(0, float("inf"))
        self.from_time.setValue(0)
        self.from_time.setSingleStep(0.1)
        self.from_time.setDecimals(1)
        self.from_time.setSuffix(" s")
        self.from_time.setAccelerated(True)
        self.from_time.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.from_time.valueChanged.connect(
            lambda: self.from_time.setValue(float("inf")) if self.from_time.value() > 1e8 else None
        )
        self.total_time_label = QLabel("Total:", self)
        self.total_time_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.total_time_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.total_time = QDoubleSpinBox(self)
        self.total_time.setRange(0, float("inf"))
        self.total_time.setValue(float("inf"))
        self.total_time.setSingleStep(0.1)
        self.total_time.setDecimals(1)
        self.total_time.setSuffix(" s")
        self.total_time.setAccelerated(True)
        self.total_time.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.total_time.valueChanged.connect(
            lambda: self.total_time.setValue(float("inf")) if self.total_time.value() > 1e8 else None
        )
        self.jobs_label = QLabel("Parallel jobs:", self)
        self.jobs_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.jobs_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.jobs = QSpinBox(self)
        self.jobs.setRange(1, 100)
        self.jobs.setValue(1)
        self.jobs.setSingleStep(1)
        self.jobs.setAccelerated(True)
        self.jobs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.jobs_each_device_frame = QWidget(self)
        self.jobs_each_device_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.jobs_each_device_frame.setObjectName("jobs_each_device_frame")
        self.jobs_each_device_frame.setStyleSheet("#jobs_each_device_frame{border: 2px solid #888;}")
        self.jobs_each_device_layout = QGridLayout(self.jobs_each_device_frame)
        self.jobs_each_device_layout.setContentsMargins(5, 5, 5, 5)
        self.jobs_each_device_frame.setLayout(self.jobs_each_device_layout)
        self.jobs_each_device_label = QLabel("Parallel jobs on each device:", self)
        self.jobs_each_device_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.jobs_each_device_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.jobs_each_device_layout.addWidget(self.jobs_each_device_label, 0, 0, 1, 2)
        self.device_labels = []
        self.device_jobs = []
        self.cache_limie_label = QLabel("Cache limit:", self)
        self.cache_limie_label.setToolTip(
            "Cache same frames limit in memory, "
            "larger will use more memory but may faster if same frames are rendered multiple times"
        )
        self.cache_limie_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.cache_limie_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.cache_limit = QSpinBox(self)
        self.cache_limit.setRange(1, 1000)
        self.cache_limit.setValue(60)
        self.cache_limit.setSingleStep(1)
        self.cache_limit.setAccelerated(True)
        self.cache_limit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.use_device_cache = QCheckBox("Use device cache", self)
        self.use_device_cache.setChecked(True)
        self.use_device_cache.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.use_device_cache.setToolTip(
            "Cache all pages to device memory for faster rendering, requires more device memory"
        )
        self.smooth_cursor = QCheckBox("Smooth cursor", self)
        self.smooth_cursor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.smooth_cursor.setToolTip("Smooth cursor movement")
        self.smooth_cursor.setChecked(True)
        self.fixed_note_width_checkbox = QCheckBox("Fixed note width", self)
        self.fixed_note_width_checkbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.fixed_note_width_checkbox.setToolTip(
            "Note highlight width will not be automatically adjusted to fit each note"
        )
        self.fixed_note_width_checkbox.setChecked(True)
        self.fixed_note_width_label = QLabel("Fixed note width (0 means auto):", self)
        self.fixed_note_width_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.fixed_note_width_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.fixed_note_width = QSpinBox(self)
        self.fixed_note_width.setRange(0, 1000)
        self.fixed_note_width.setValue(0)
        self.fixed_note_width.setSingleStep(1)
        self.fixed_note_width.setAccelerated(True)
        self.fixed_note_width.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.fixed_note_width.setSuffix(" px")
        self.fixed_note_width_label.setBuddy(self.fixed_note_width)
        self.fixed_note_width_checkbox.toggled.connect(
            lambda: self.fixed_note_width.setEnabled(self.fixed_note_width_checkbox.isChecked())
        )
        self.extra_note_width_ratio_label = QLabel("Extra note width ratio:", self)
        self.extra_note_width_ratio_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.extra_note_width_ratio_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.extra_note_width_ratio = QSpinBox(self)
        self.extra_note_width_ratio.setRange(0, 1000)
        self.extra_note_width_ratio.setValue(40)
        self.extra_note_width_ratio.setSingleStep(1)
        self.extra_note_width_ratio.setAccelerated(True)
        self.extra_note_width_ratio.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.extra_note_width_ratio.setSuffix(" %")
        self.extra_note_width_ratio_label.setBuddy(self.extra_note_width_ratio)
        self.resize_function_label = QLabel("Resize function:", self)
        self.resize_function_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.resize_function_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.resize_crop = QRadioButton("Crop", self)
        self.resize_crop.setChecked(True)
        self.resize_crop.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.resize_rescale = QRadioButton("Rescale", self)
        self.resize_rescale.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.resize_crop.toggled.connect(lambda: self.resize_rescale.setChecked(not self.resize_crop.isChecked()))
        self.resize_rescale.toggled.connect(lambda: self.resize_crop.setChecked(not self.resize_rescale.isChecked()))

        self.files_layout = QVBoxLayout(self.files_frame)
        self.files_layout.setContentsMargins(10, 0, 10, 10)
        self.files_layout.addWidget(self.load_mscz_button)
        self.files_layout.addWidget(self.current_mscz_label)
        self.audio_buttons_layout = QHBoxLayout()
        self.audio_buttons_layout.addWidget(self.load_audio_button)
        self.audio_buttons_layout.addWidget(self.clear_audio_button)
        self.files_layout.addLayout(self.audio_buttons_layout)
        self.files_layout.addWidget(self.current_audio_label)
        self.audio_delay_layout = QHBoxLayout()
        self.audio_delay_layout.addWidget(self.audio_delay_label)
        self.audio_delay_layout.addWidget(self.audio_delay)
        self.files_layout.addLayout(self.audio_delay_layout)
        self.video_encoder_layout = QHBoxLayout()
        self.video_encoder_layout.addWidget(self.video_encoder_label)
        self.video_encoder_layout.addWidget(self.video_encoder)
        self.files_layout.addLayout(self.video_encoder_layout)
        self.video_encoder_method_layout = QHBoxLayout()
        self.video_encoder_method_layout.addWidget(self.video_encoder_method_label)
        self.video_encoder_method_layout.addWidget(self.video_encoder_method)
        self.video_encoder_method_layout.addWidget(self.video_bitrate)
        self.files_layout.addLayout(self.video_encoder_method_layout)
        self.audio_encoder_layout = QHBoxLayout()
        self.audio_encoder_layout.addWidget(self.audio_encoder_label)
        self.audio_encoder_layout.addWidget(self.audio_encoder)
        self.audio_encoder_layout.addWidget(self.audio_bitrate)
        self.files_layout.addLayout(self.audio_encoder_layout)

        self.controls_layout = QVBoxLayout(self.controls_frame)
        self.size_fps_layout = QHBoxLayout()
        self.size_fps_layout.addWidget(self.size_label)
        self.size_fps_layout.addWidget(self.size_x)
        self.size_fps_layout.addWidget(self.size_multiplier)
        self.size_fps_layout.addWidget(self.size_y)
        self.size_fps_layout.addWidget(self.fps_label)
        self.size_fps_layout.addWidget(self.fps)
        self.controls_layout.addLayout(self.size_fps_layout)
        self.bar_color_layout = QHBoxLayout()
        self.bar_color_container_layout = QVBoxLayout(self.bar_color_label_container)
        self.bar_color_container_layout.setContentsMargins(0, 0, 0, 0)
        self.bar_color_container_layout.addWidget(self.bar_color_label)
        self.bar_color_layout.addWidget(self.bar_color_label_container)
        self.bar_color_layout.addWidget(self.bar_color)
        self.bar_color_layout.addWidget(self.bar_alpha)
        self.controls_layout.addLayout(self.bar_color_layout)
        self.note_color_layout = QHBoxLayout()
        self.note_color_container_layout = QVBoxLayout(self.note_color_label_container)
        self.note_color_container_layout.setContentsMargins(0, 0, 0, 0)
        self.note_color_container_layout.addWidget(self.note_color_label)
        self.note_color_layout.addWidget(self.note_color_label_container)
        self.note_color_layout.addWidget(self.note_color)
        self.note_color_layout.addWidget(self.note_alpha)
        self.controls_layout.addLayout(self.note_color_layout)
        self.offsets_layout = QHBoxLayout()
        self.offsets_layout.addWidget(self.start_offset_label)
        self.offsets_layout.addWidget(self.start_offset)
        self.offsets_layout.addWidget(self.end_offset_label)
        self.offsets_layout.addWidget(self.end_offset)
        self.controls_layout.addLayout(self.offsets_layout)
        self.from_to_layout = QHBoxLayout()
        self.from_to_layout.addWidget(self.from_label)
        self.from_to_layout.addWidget(self.from_time)
        self.from_to_layout.addWidget(self.total_time_label)
        self.from_to_layout.addWidget(self.total_time)
        self.controls_layout.addLayout(self.from_to_layout)
        self.jobs_layout = QHBoxLayout()
        self.jobs_layout.addWidget(self.jobs_label)
        self.jobs_layout.addWidget(self.jobs)
        self.controls_layout.addLayout(self.jobs_layout)
        self.controls_layout.addWidget(self.jobs_each_device_frame)
        self.cache_limit_layout = QHBoxLayout()
        self.cache_limit_layout.addWidget(self.cache_limie_label)
        self.cache_limit_layout.addWidget(self.cache_limit)
        self.controls_layout.addLayout(self.cache_limit_layout)
        self.controls_layout.addWidget(self.use_device_cache)
        self.controls_layout.addWidget(self.smooth_cursor)
        self.controls_layout.addWidget(self.fixed_note_width_checkbox)
        self.fixed_note_width_layout = QHBoxLayout()
        self.fixed_note_width_layout.addWidget(self.fixed_note_width_label)
        self.fixed_note_width_layout.addWidget(self.fixed_note_width)
        self.controls_layout.addLayout(self.fixed_note_width_layout)
        self.extra_note_width_ratio_layout = QHBoxLayout()
        self.extra_note_width_ratio_layout.addWidget(self.extra_note_width_ratio_label)
        self.extra_note_width_ratio_layout.addWidget(self.extra_note_width_ratio)
        self.controls_layout.addLayout(self.extra_note_width_ratio_layout)
        self.resize_function_layout = QHBoxLayout()
        self.resize_function_layout.addWidget(self.resize_function_label)
        self.resize_function_layout.addWidget(self.resize_crop)
        self.resize_function_layout.addWidget(self.resize_rescale)
        self.controls_layout.addLayout(self.resize_function_layout)

        self.widget_layout = QGridLayout(self)
        self.widget_layout.addWidget(self.preview_frame, 0, 0)
        self.preview_layout = QVBoxLayout(self.preview_frame)
        self.preview_space_top = QWidget(self.preview_frame)
        self.preview_space_top.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.preview_layout.addWidget(self.preview_space_top)
        self.preview_layout.addWidget(self.preview_window)
        self.preview_layout.addWidget(self.render_button, alignment=Qt.AlignmentFlag.AlignCenter)
        self.preview_space_bottom = QWidget(self.preview_frame)
        self.preview_space_bottom.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.preview_layout.addWidget(self.preview_space_bottom)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        self.widget_layout.addWidget(
            self.controls_frame, 0, 1, 3, 1, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.stop_button_layout = QVBoxLayout(self.stop_button_frame)
        self.stop_button_layout.addWidget(self.stop_button)
        self.stop_button_layout.setContentsMargins(10, 10, 10, 0)
        self.widget_layout.addWidget(self.stop_button_frame, 1, 0)
        self.widget_layout.addWidget(self.files_frame, 2, 0)
        self.progress_layout = QHBoxLayout()
        self.progress_layout.addWidget(self.progress_bar)
        self.progress_layout.addWidget(self.progess_text)
        self.widget_layout.addLayout(self.progress_layout, 3, 0, 1, 2)
        self.widget_layout.addWidget(self.close_show_log, 4, 0, 1, 2)
        self.widget_layout.addWidget(self.log_text, 5, 0, 1, 2)
        self.log_text.hide()
        self.setLayout(self.widget_layout)

        sys.stderr = VirtualTerminal(lambda x: self.exec_in_main(lambda: self.set_log(x)))
        logging.basicConfig(
            format="%(asctime)s (%(filename)s) (Line %(lineno)d) [%(levelname)s] : %(message)s",
            level=logging.INFO,
            stream=sys.stderr,
        )
        self.timer = QTimer(self)
        self.timer.singleShot(0, self.search_binaries)
        self.timer.singleShot(0, self.search_devices)

    def set_log(self, log):
        if self.log_text.isVisible():
            pos_y = self.log_text.verticalScrollBar().value()
            pos_x = self.log_text.horizontalScrollBar().value()
            scroll = pos_y == self.log_text.verticalScrollBar().maximum()
            self.log_text.setText(log)
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum() if scroll else pos_y)
            self.log_text.horizontalScrollBar().setValue(pos_x)
        self.log = log
        last_line = log.strip().rsplit("\n", 1)[-1]
        if m := re.search(r"^\s*([a-zA-Z ]*?)\s*\.*\s*(\d+)\s*/\s*(\d+)", last_line):
            self.progress_bar.setMaximum(int(m[3]))
            self.progress_bar.setValue(int(m[2]))
            self.progess_text.setText(f"{m[1]} {m[2]}/{m[3]}")
        elif "done" in last_line.lower():
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.progess_text.setText("No job running")

    def toggle_log(self):
        if self.log_text.isVisible():
            self.log_text.hide()
            self.close_show_log.setText("Show log")
        else:
            self.log_text.show()
            self.log_text.setText(self.log)
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
            self.close_show_log.setText("Hide log")

    def exec_in_main(self, func):
        with self._execInMainThreadLock:
            self._execInMainThreadFunc = func
            self._execInMainThreadResultEvent.clear()
            self._execInMainThreadSignal.emit()
            self._execInMainThreadResultEvent.wait()
            if self._execInMainThreadSuccess:
                ret = self._execInMainThreadResult
                self._execInMainThreadResult = None
                self._execInMainThreadFunc = None
                return ret
            else:
                err = self._execInMainThreadResult
                self._execInMainThreadResult = None
                self._execInMainThreadFunc = None
                raise err

    def _exec_in_main_thread_executor(self):
        try:
            self._execInMainThreadResult = self._execInMainThreadFunc()
            self._execInMainThreadSuccess = True
        except Exception as e:
            self._execInMainThreadResult = e
            self._execInMainThreadSuccess = False
        self._execInMainThreadResultEvent.set()

    @thread_wrapper(daemon=True)
    def search_devices(self):
        global torch, ipex
        try:
            import torch
        except ImportError:
            self.exec_in_main(
                lambda: QMessageBox.critical(
                    self,
                    "Torch not found",
                    "Torch not found. Did you extracted the torch runtime files correctly?",
                    QMessageBox.StandardButton.Ok,
                )
            )
            self.exec_in_main(self.close)

        devices = {"cpu": [f"{psutil.virtual_memory().total // 1048576} MB", platform.processor()]}
        if sys.platform == "darwin":
            if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                devices["mps"] = [f"{psutil.virtual_memory().total // 1048576} MB", "Metal Performance Shaders"]
        else:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    devices[f"cuda:{i}"] = [
                        f"{torch.cuda.get_device_properties(i).total_memory // 1048576} MB",
                        torch.cuda.get_device_name(i),
                    ]
                    print("Found CUDA device", i, ":", torch.cuda.get_device_properties(i), file=sys.stderr)
            try:
                if packaging.version.Version(torch.__version__) < packaging.version.Version("2.2.0"):
                    import intel_extension_for_pytorch as ipex  # type: ignore[import]

                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    for i in range(torch.xpu.device_count()):
                        devices[f"xpu:{i}"] = [
                            f"ipex {torch.xpu.get_device_properties(i).total_memory // 1048576} MB",
                            torch.xpu.get_device_name(i),
                        ]
                        print("Found IPEX device", i, ":", torch.xpu.get_device_properties(i), file=sys.stderr)
            except ImportError:
                pass
        self.exec_in_main(lambda: self.add_devices(devices))

    def add_devices(self, devices):
        for device in devices:
            label = QLabel(f"{device} ({devices[device][0]})", self)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            label.setToolTip(devices[device][1])
            self.device_labels.append(label)
            jobs = QSpinBox(self)
            jobs.setRange(0, 100)
            jobs.setValue(1)
            jobs.setSingleStep(1)
            jobs.setAccelerated(True)
            jobs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            jobs.setToolTip(devices[device][1])
            jobs.valueChanged.connect(lambda: self.jobs.setMaximum(max(1, sum([j.value() for j in self.device_jobs]))))
            self.device_jobs.append(jobs)
            label.setBuddy(jobs)
            self.jobs_each_device_layout.addWidget(label, len(self.device_labels), 0)
            self.jobs_each_device_layout.addWidget(jobs, len(self.device_jobs), 1)
        self.device_jobs[0].valueChanged.emit(self.device_jobs[0].value())
        self.load_mscz_button.setDisabled(False)
        self.load_mscz_button.setToolTip("")

    def switch_video_bitrate_range(self):
        if self.video_encoder_method.currentText() == "VBR":
            self.video_bitrate.setRange(1, 1000000)
            self.video_bitrate.setValue(1000)
            self.video_bitrate.setSuffix(" kbps")
        else:
            self.video_bitrate.setRange(0, 51)
            self.video_bitrate.setValue(18)
            self.video_bitrate.setSuffix("")

    def update_bar_color(self, ask_color=False):
        color = self.bar_color_label.styleSheet().split("background-color: #")[-1][2:8]
        if ask_color:
            new_color = QColorDialog.getColor(f"#{color}", self)
            if new_color.isValid():
                color = new_color.name(QtGui.QColor.NameFormat.HexRgb)[1:]
        self.bar_color_label.setStyleSheet(
            f"background-color: #{self.bar_alpha.value():02X}{color}; color: black; padding: 3px;"
        )

    def update_note_color(self, ask_color=False):
        color = self.note_color_label.styleSheet().split("background-color: #")[-1][2:8]
        if ask_color:
            new_color = QColorDialog.getColor(f"#{color}", self)
            if new_color.isValid():
                color = new_color.name(QtGui.QColor.NameFormat.HexRgb)[1:]
        self.note_color_label.setStyleSheet(
            f"background-color: #{self.note_alpha.value():02X}{color}; color: black; padding: 3px;"
        )

    @thread_wrapper(daemon=True)
    def show_preview(self):
        while True:
            if self.stop:
                self.exec_in_main(lambda: self.preview_window.setPixmap(QtGui.QPixmap()))
                self.exec_in_main(lambda: self.preview_window.setText("Start rendering to show preview"))
                return
            self.update_preview_event.wait(1)
            self.update_preview_event.clear()
            if self.preview_window.text() or not hasattr(self, "frame_preview"):
                continue
            required_size = self.preview_window.size().toTuple()
            if self.frame_preview.shape[0] / required_size[1] > self.frame_preview.shape[1] / required_size[0]:
                new_height = required_size[1]
                new_width = int(self.frame_preview.shape[1] / self.frame_preview.shape[0] * new_height)
            else:
                new_width = required_size[0]
                new_height = int(self.frame_preview.shape[0] / self.frame_preview.shape[1] * new_width)
            new_width = min(new_width, self.frame_preview.shape[1])
            new_height = min(new_height, self.frame_preview.shape[0])
            self.resized_preview = (
                torch.nn.functional.interpolate(
                    torch.from_numpy(self.frame_preview)[None, ...].permute(0, 3, 1, 2).to(torch.float32) / 255,
                    size=(new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )
                .permute(0, 2, 3, 1)
                .squeeze(0)
                .clamp(0, 1)
                .mul(255)
                .to(torch.uint8)
                .numpy()
                .copy()
            )
            preview_qimage = QtGui.QImage(
                self.resized_preview.data, new_width, new_height, 3 * new_width, QtGui.QImage.Format.Format_RGB888
            )
            preview_pixmap = QtGui.QPixmap.fromImage(preview_qimage)
            self.exec_in_main(lambda: self.preview_window.setPixmap(preview_pixmap))

    def update_preview(self, frame_id, total_frames, width, height, frame: bytes):
        if self.stop:
            raise RuntimeError("Stop requested")
        if time.time() - self.last_update < 0.5:
            return
        self.last_update = time.time()
        self.frame_preview = np.frombuffer(frame, np.uint8).copy().reshape((height, width, 3))
        self.exec_in_main(
            lambda: self.preview_window.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )
        self.update_preview_event.set()

    def ask_stop(self):
        self.stop = True

    def closeEvent(self, event):
        self.stop = True
        sys.stderr.close()
        sys.stderr = sys.__stderr__
        return super().closeEvent(event)

    @thread_wrapper(daemon=True)
    def render(self):
        self.exec_in_main(lambda: self.render_button.setDisabled(True))
        output_path = self.exec_in_main(
            lambda: QFileDialog.getSaveFileName(
                self,
                "Save video",
                str(pathlib.Path(self.current_mscz_label.text()).parent),
                "*.mp4;;*.mkv;;*.mov;;*.flv;;*.m4v",
            )[0]
        )
        if not output_path:
            self.exec_in_main(lambda: self.render_button.setDisabled(False))
            return
        self.exec_in_main(lambda: self.controls_frame.setDisabled(True))
        self.exec_in_main(lambda: self.files_frame.setDisabled(True))
        output_path = pathlib.Path(output_path)
        extra_ffmpeg_args = []
        if self.current_audio_label.text() != "No audio file selected":
            if self.audio_delay.value() < 0:
                extra_ffmpeg_args += ["-itsoffset", str(-self.audio_delay.value())]
            extra_ffmpeg_args += ["-i", self.current_audio_label.text()]
            if self.audio_delay.value() > 0:
                extra_ffmpeg_args += ["-af", f"adelay={int(self.audio_delay.value() * 1000)}:all=1"]
        extra_ffmpeg_args += ["-c:a", self.audio_encoder.currentText()]
        extra_ffmpeg_args += ["-b:a", str(self.audio_bitrate.value()) + "k"]
        extra_ffmpeg_args += ["-pix_fmt", "yuv420p"]
        extra_ffmpeg_args += ["-c:v", self.video_encoder.currentText()]
        if "265" in self.video_encoder.currentText() or "hevc" in self.video_encoder.currentText():
            extra_ffmpeg_args += ["-tag:v", "hvc1"]
        if self.video_encoder_method.currentText() == "VBR":
            extra_ffmpeg_args += ["-b:v", str(self.video_bitrate.value()) + "k"]
        else:
            if "nvenc" in self.video_encoder.currentText():
                extra_ffmpeg_args += ["-qp", str(self.video_bitrate.value())]
            else:
                extra_ffmpeg_args += ["-q:v", str(self.video_bitrate.value())]
        extra_ffmpeg_args += ["-g", str(self.fps.value() * 6)]
        self.last_update = 0
        try:
            self.exec_in_main(self.render_button.hide)
            self.exec_in_main(self.preview_space_top.hide)
            self.exec_in_main(self.preview_space_bottom.hide)
            self.stop = False
            self.show_preview()
            self.exec_in_main(lambda: self.preview_window.setText(""))
            self.converter.convert(
                output_path,
                cache_limit=self.cache_limit.value(),
                smooth_cursor=self.smooth_cursor.isChecked(),
                fixed_note_width=self.fixed_note_width.value() if self.fixed_note_width_checkbox.isChecked() else None,
                extra_note_width_ratio=self.extra_note_width_ratio.value() / 100,
                size=(self.size_x.value(), self.size_y.value()),
                bar_color=webcolors.hex_to_rgb(
                    "#" + self.bar_color_label.styleSheet().split("background-color: #")[-1][2:8]
                ),
                bar_alpha=self.bar_alpha.value(),
                note_color=webcolors.hex_to_rgb(
                    "#" + self.note_color_label.styleSheet().split("background-color: #")[-1][2:8]
                ),
                note_alpha=self.note_alpha.value(),
                callback=self.update_preview,
                start_offset=self.start_offset.value(),
                end_offset=self.end_offset.value(),
                fps=self.fps.value(),
                jobs=self.jobs.value(),
                ss=self.from_time.value(),
                t=self.total_time.value(),
                no_device_cache=not self.use_device_cache.isChecked(),
                resize_method="crop" if self.resize_crop.isChecked() else "rescale",
                ffmpeg_arg_ext=extra_ffmpeg_args,
                torch_devices=";".join(
                    f"{self.device_labels[i].text().split()[0]},{self.device_jobs[i].value()}"
                    for i in range(len(self.device_labels))
                ),
            )
            self.exec_in_main(lambda: self.render_button.setToolTip("Please load MuseScore file first"))
            self.exec_in_main(lambda: self.current_mscz_label.setText("No MuseScore file loaded"))
            self.exec_in_main(lambda: self.current_audio_label.setText("No audio file selected"))
            self.exec_in_main(lambda: self.progress_bar.setValue(0))
            self.exec_in_main(lambda: self.progess_text.setText("No job running"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
                torch.xpu.empty_cache()
        finally:
            self.stop = True
            self.exec_in_main(lambda: self.controls_frame.setDisabled(False))
            self.exec_in_main(lambda: self.files_frame.setDisabled(False))
            self.exec_in_main(lambda: self.render_button.setDisabled(True))
            self.exec_in_main(lambda: self.render_button.setToolTip("Please load MuseScore file first"))
            self.exec_in_main(lambda: self.preview_window.setPixmap(QtGui.QPixmap()))
            self.exec_in_main(lambda: self.preview_window.setText("Start rendering to show preview"))
            self.exec_in_main(
                lambda: self.preview_window.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            )
            self.exec_in_main(self.render_button.show)
            self.exec_in_main(self.preview_space_top.show)
            self.exec_in_main(self.preview_space_bottom.show)

    def ask_mscz(self):
        mscz_path = QFileDialog.getOpenFileName(
            self,
            "Select MuseScore file",
            "",
            "MuseScore compatible files(*.mscz *.mscx *.mid *.midi *.musicxml *.mxl);;All files(*)",
        )[0]
        if not mscz_path:
            return
        self.load_mscz(mscz_path)

    @thread_wrapper(daemon=True)
    def load_mscz(self, mscz_path):
        try:
            self.exec_in_main(lambda: self.render_button.setDisabled(True))
            self.exec_in_main(lambda: self.render_button.setToolTip("Please load MuseScore file first"))
            self.exec_in_main(lambda: self.load_mscz_button.setDisabled(True))
            self.exec_in_main(lambda: self.current_mscz_label.setText("Loading..."))
            if not mscz_path:
                return
            self.converter.load_score(pathlib.Path(mscz_path), use_svg=False)
        finally:
            self.exec_in_main(lambda: self.load_mscz_button.setDisabled(False))
            self.exec_in_main(lambda: self.current_mscz_label.setText("No MuseScore file loaded"))
        self.exec_in_main(lambda: self.current_mscz_label.setText(mscz_path))
        self.exec_in_main(lambda: self.render_button.setDisabled(False))
        self.exec_in_main(lambda: self.render_button.setToolTip(""))

    def select_audio(self):
        audio_path = QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            "",
            "Audio files(*.mp3 *.flac *.wav *.ogg *.m4a *.aac *.wma *.opus);;All files(*)",
        )[0]
        if not audio_path:
            return
        self.current_audio_label.setText(audio_path)

    def clear_audio(self):
        self.current_audio_label.setText("No audio file selected")

    @thread_wrapper(daemon=True)
    def search_binaries(self):
        self.musescore_path = self.search_musescore()
        self.ffmpeg_path = self.search_ffmpeg()
        try:
            print("Found musescore at", self.musescore_path, file=sys.stderr)
            print(
                "MuseScore version:",
                subprocess.Popen([self.musescore_path, "--version"], stdout=subprocess.PIPE).communicate()[0].decode(),
                file=sys.stderr,
            )
            print("Found ffmpeg at", self.ffmpeg_path, file=sys.stderr)
            print(
                "ffmpeg version:\n"
                + subprocess.Popen([self.ffmpeg_path, "-version"], stdout=subprocess.PIPE).communicate()[0].decode(),
                file=sys.stderr,
            )
        except Exception:
            self.exec_in_main(
                lambda: QMessageBox.critical(self, "Error", "Failed to run MuseScore or ffmpeg. Please check the path.")
            )
            self.exec_in_main(lambda: self.close())
            raise FileNotFoundError("MuseScore or ffmpeg not found")
        self.converter = convert_core.Converter(
            use_torch=True, ffmpeg_path=self.ffmpeg_path, musescore_path=self.musescore_path
        )

    def search_musescore(self):
        if sys.platform == "win32":
            if pathlib.Path("C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe").exists():
                return "C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe"
            elif pathlib.Path("C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe").exists():
                return "C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe"
        else:
            if sys.platform == "darwin":
                os.environ["PATH"] = "/Applications/MuseScore 4.app/Contents/MacOS" + os.pathsep + os.environ["PATH"]
                os.environ["PATH"] = "/Applications/MuseScore 3.app/Contents/MacOS" + os.pathsep + os.environ["PATH"]
            if p := shutil.which("mscore"):
                return p
            elif p := shutil.which("musescore"):
                return p
            elif p := shutil.which("mscore4portable"):
                return p
            elif p := shutil.which("mscore-portable"):
                return p
        asked_path = self.exec_in_main(
            lambda: QFileDialog.getOpenFileName(
                self,
                "Select MuseScore executable",
                "",
                (
                    "MuseScore executable(MuseScore4.exe MuseScore3.exe);;All files(*)"
                    if sys.platform == "win32"
                    else ("MuseScore executable(mscore musescore mscore-portable mscore4portable);;All files(*)")
                ),
            )
        )[0]
        if not asked_path:
            self.exec_in_main(
                lambda: QMessageBox.critical(self, "Error", "MuseScore executable not found. Please install MuseScore.")
            )
            self.exec_in_main(lambda: self.close())
            raise FileNotFoundError("MuseScore executable not found")
        return asked_path

    def search_ffmpeg(self):
        os.environ["PATH"] = str(pathlib.Path(__file__).parent / "ffmpeg") + os.pathsep + os.environ["PATH"]
        if p := shutil.which("ffmpeg"):
            return p
        asked_path = self.exec_in_main(
            lambda: QFileDialog.getOpenFileName(
                self,
                "Select ffmpeg executable",
                "",
                "ffmpeg executable(ffmpeg);;All files(*)",
            )
        )[0]
        if not asked_path:
            self.exec_in_main(
                lambda: QMessageBox.critical(self, "Error", "ffmpeg executable not found. Please install ffmpeg.")
            )
            self.exec_in_main(lambda: self.close())
            raise FileNotFoundError("ffmpeg executable not found")
        return asked_path


app = QApplication([])
if app.style().name().lower() == "windows11":
    app.setStyle(QStyleFactory.create("windowsvista"))
window = MainWindow()
window.show()
app.exec()
