本文献译自[英语](README.md)版本，仅供参考。

# mscz-to-video

将 MuseScore 文件渲染成视频文件

<video controls playsinline style="width:100%;height:fit-content;padding-bottom:56.25%;overflow-y:hidden" class="video-js" data-setup="{}"><source src="https://mscz-video.carlgao4.workers.dev/FlowerDance/FlowerDance.m3u8">Your browser does not support the video tag.</video>

*这里本应有一支样本视频。请前往 [GitHub Pages](https://carlgao4.github.io/mscz-to-video/) 以观看该视频*

## 特性

- [x] 将 MuseScore 文件导出为视频
- [x] 以不同的高亮颜色展示当前音符和小节
- [x] 手动设置高亮颜色和透明度
- [x] 光标在音符之间匀速移动
- [x] 并行渲染
- [x] 以 PyTorch 加速，包括对于图形处理器和实时编译（在单个 4060m 图形处理器上的速度能够达到大约 4K 52 帧每秒）的支持
- [x] 重设尺寸以刈割或者重新缩放每一页面以使得当前音符和小节将总是处于画面中央
- [x] 支持多种图形处理器
- [x] 图形用户界面（UI）版本
- [x] 自动将音频加诸视频（仅限 UI 版本）
- [x] 即时预览（仅限 UI 版本）
- [ ] 支持自动添加音频

## 要求

**仅支持 Windows 10 与麦金塔 10.15 或者后续版本。仅支持 64 位系统。**

- [MuseScore](https://github.com/musescore/MuseScore)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [Python](https://www.python.org/) 3（见以下 Python 库）
  - `numpy<2`
  - `Pillow`
  - `webcolors`
  - 如果您想要加快渲染，您可以按照 [Pytorch 页面](https://pytorch.org/get-started/locally/)中的指令安装 `torch`

---

UI 版本的额外要求：

- `PySide6`
- `psutil`
- `torch`

如果您想要用英特尔®图形处理器加速，您可以安装带有处理器支持的[英特尔®的 Pytorch 扩展](https://github.com/intel/intel-extension-for-pytorch)。

## UI 用法

**仅限最新版本。对于旧版本，请参阅每一版本的发布说明。**

1. 从[发布页面](https://github.com/CarlGao4/mscz-to-video/releases/latest)根据您的平台下载二进制。
   - Windows
      1. 下载 `mscz2video_<version>.7z` 并且首先提取之。
      2. 由于我已为英伟达®图形处理器加速和英特尔®图形处理器加速构建，您需要根据您的硬件选择 torch 运行时版本。torch 运行时适宜从 [0.2 版本的发布](https://github.com/CarlGao4/mscz-to-video/releases/0.2)下载。后续版本使用完全相同的运行时。文件名称形如 `torch-runtime-<arch>.7z`。
         - 如果您拥有运算能力大于或等于 3.5 的英伟达®显卡，您可以选择 CUDA® 版本 **`cu118`**。
         - 对于英特尔®显卡，构建单一的二进制尺寸太大，所以我已将其分成 4 个版本。如果您拥有带有核显或者 DG1 独显的第 11~14 代核心的中央处理器，您可以选择 **`mkl.gen12lp`**。
         - 对于英特尔®锐炫 Alchemist 图形处理器（Arc Axxx）和 Ponte Vecchio 图形处理器（Datacenter GPU Max），您可以选择 **`mkl.xe-hpg.xe-hpc.xe-hpc-vg`**。
         - 对于 Lunar Lake 中央处理器（英特尔核心 Ultra 2xxV）和 Battlemage 图形处理器（英特尔®锐炫 Bxxx），您可以选择 **`mkl.xe2`**。
         - 对于核心 Ultra 系列但是使用 Raptor Lake Refresh 架构（`100U`、`120U`、`150U`、`220U`、`250U`、`210H`、`220H`、`240H`、`250H`、`270H`），它们使用旧的图形处理器架构，所以您可以选择 **`mkl.gen12lp`**。
         - 对于其他英特尔®核心 Ultra 系列 1 和 2 的图形处理器，请选择 **`mkl.xe-lpg.xe-lpgplus`**。
         - 如果您没有上述图形处理器或者不想使用图形处理器加速，您可以选择中央处理器版本。**两个版本都能够使用图形处理器加速编码，但是只有图形处理器版本能够加速渲染。如果您未安装最新英特尔®图形驱动但是安装了英特尔®图形处理器运行时，程序启动将会失败。请先安装最新英特尔®图形驱动。**
      3. 将 torch 运行时提取到应用内。`mscz2video.exe`应当在与 `torch` 文件夹相同的路径中。
   - 麦金塔
      1. 下载 `mscz2video_xxx_macOS_<ARCH>.dmg` 并且打开它。将应用拖到 `Applications` 文件夹中。`<ARCH>` 是您的麦金塔的架构，可能是 `arm64` 或者 `x86_64`。苹果开发者资格昂贵以致我不能为应用签名，所以如果该应用被麦金塔封禁，您可以参考下文[致麦金塔用户的说明](#CannotOpen)。
2. 创建 MuseScore 文件并且将其排版以适应您的视频，或者使用提供的示例文件 `Flower Dance.mscz`（要求 MuseScore 4.4 或者后续版本）。如果您不想要您的视频屏幕滚动，您需要将您的页面比例设为与您的视频分辨率相同。为此您可以前往将 <kbd> <samp class="menu">格式</samp> → <samp class="menuitem">页面设置</samp> → <samp class="submenu">`页面尺寸`</samp> → <samp class="submenu">自定义</samp> → <samp class="menuitem">宽度</samp> 和 <samp class="menuitem">高度</samp> </kbd> 设为您想要的比例。请勿将它们改得太小，因为默认输出尺寸是 330 dpi（像素每英寸，意味着 1 英寸中有 330 像素）。您还可以改变`谱表间距`并添加新谱行以使得每一页面更好地展示。我个人还推荐设置 <kbd> <samp class="menu">格式</samp> → <samp class="menuitem">样式</samp> → <samp class="submenu">页眉与页脚</samp> </kbd> 以使得奇数/偶数页面具有相同的页眉和页脚。
3. 打开该程序（在 Windows 上，您只需双击 `mscz2video.exe`）。
4. 等待程序启动。过程完成以前，您不能加载 MSCZ 文件。在启动期间，程序将会搜索 MuseScore 和 FFmpeg。我已随附该程序打包 FFmpeg 所以您无需下载它，但是您应当自行下载 MuseScore。该程序将会在 Windows 上搜索 `%ProgramFiles%\MuseScore 4\bin\MuseScore4.exe` 和 `%ProgramFiles%\MuseScore 3\bin\MuseScore3.exe`，和麦金塔上的 `/Applications/MuseScore 4.app/Contents/MacOS/mscore` 和 `/Applications/MuseScore 3.app/Contents/MacOS/mscore`。如果该程序找不到 MuseScore，将会弹出对话框以请您手动设置通往 MuseScore 的路径，只需选择通往可执行的 MuseScore 的正确路径。在麦金塔上，您需要进入该应用包并且在 `Contents/MacOS` 文件夹中找到该可执行程序。欲打开应用包，您需要在 Finder 中按下 <kbd> <kbd>Cmd</kbd> + <kbd>Shift</kbd> + <kbd>G</kbd> </kbd> 并且进入该路径。
5. 加载 MSCZ 文件的方式是点击 <kbd> <samp class="buton">`Load MuseScore file`</samp> </kbd> 或者将文件拖到该按钮处。文件加载以前，您不能开始渲染。在此期间您能够改变其他设置。
    1. **视频设置**：您可以设置视频的分辨率和刷新率。默认分辨率是 1920 × 1080 而默认刷新率是 60 帧每秒。其他常用分辨率有：`1280 × 720`（720p）、`1920 × 1080`（1080p）、`2560 × 1440`（1440p 或者 2K）、`3840 × 2160`（4K）、`7680 × 4320`（8K）。
    2. **Bar and note highlight color**：当前小节和当前音符的背景颜色。您可以设置颜色和透明度，并且预览标签上的效果。
    3. **视频进度**：<kbd> <samp class="button">Start offset</samp> </kbd> 和 <kbd> <samp class="button">End offset</samp> </kbd>是首音之前和尾音起奏之后的时间（*不是*迟于尾音释放的时间）。您还可以设置视频的开始时间和时长。开始时间的计算先于 `Start offset`，所以如果您将开始时间设为 1 并且 `Start offset` 设为 1，视频将会始于第 0 秒。视频将会结束于达到时长和起奏尾音中先到者。开始时间和时长由本程序使用，而不是传到 FFmpeg 的参数。
    4. **parallel jobs**：渲染视频的并行工作的数目。在每个设备上的默认值是 1。如果您已下载中央处理器版本，仅仅中央处理器可用。在麦金塔上，MPS 总是可用的，如果您拥有 M 系列的系统级芯片或者超微半导体®图形处理器。您可以在标签方悬停以见到设备名称而不是设备标识符。
    5. `Cache limit`：缓存限值将会限制储存在内存中的帧数。在渲染期间可能出现重复的帧。当本程序找到已经渲染的帧之时，它将会使用缓存帧而不是再次渲染它。然而，如果达到缓存限值，本程序将会删除缓存中最老旧的帧。除此之外，由于支持并行渲染，帧可能不依次渲染，所以本程序还将会使用缓存来储存已渲染但是尚未使用的帧。默认缓存限值是 60。使用设备缓存将会把原始图像储存在设备内存中，这将会加速渲染过程，方式是避免将图像从中央处理器转移到图形处理器内存中。然而，这将会使用更多设备内存。如果您的设备内存不足，您可以禁用此选项，
    6. `Smooth cursor`：匀速光标移动将会使得光标匀速移动于音符之间。不带匀速光标之时，音符高亮区域将会在起奏下一音符之时跳过去。正如 MuseScore 中的回放光标，MuseScore 3 没有匀速光标，但是 MuseScore 4 有。由于几乎无法缓存相同帧，匀速光标将会使得渲染过程大大放缓。
    7. `Fixed note width`：如果启用此选项，音符高亮矩形的宽度将会固定为您设置的值。如果您将其设为 0，将会自动计算音符高亮矩形的宽度，这是从该 MSCZ 文件读取的一个四分音符的精确宽度（实际上，它是从谱间计算的。如果它不在该 MSCZ 文件中定义或者输入文件不是 MSCZ 文件，将会使用 MuseScore 的默认值）。如果您将其设为大于 0 的值，音符高亮矩形的宽度将会固定为以像素计的该值。附加音符宽度比将会按照您设定的比率扩大音符高亮区域。譬如，如果您将其设为 0.4，音符高亮区域将会在每一侧分别扩大目标音符的 20%。这能够带来更好的视觉效果，就像在音符周围添加了衬垫。
    8. `Resize function`: 重设尺寸功能将会把每一页面的尺寸重设为目标尺寸。Crop 将会把每一页面按照同一比率刈割到最大可能尺寸而保持当前音符处于画面中心。Rescale 将会把每一页面刈割到目标尺寸，而忽略比率。
    9. **编码设置**：请记住，本程序**将不会**识别您的选择能否生产有效的视频！
        - **将音频多路复用到视频中**：本程序**将不会**自动渲染音频。您需要先从 MuseScore 中手动导出音频并且将其载入此处（或者将它拖到该按钮处）。请勿忘记设置 `Audio delay`。通常，您需要将 `Audio delay` 设为与 `Start offset`相同，除非您正在使用其他音频资源或者开始时间不是 0。通过启用链接按钮（🔗，默认启用），此 `Audio delay` 值将会自动调整以匹配您的 `Start offset` 和 `From` 时间设置（`Start offset` - `From`）。
        - **视频编码器设置**：您可以选择视频编解码器和编码器。在 Windows 和 Linux 上，英特尔®、英伟达®，和超微半导体®显卡加速都有所支持（即使您已下载了中央处理器版本）。选择编码器：以 `_nvenc` 结尾的是英伟达®，`_amf` 是超微半导体®，`_qsv` 是英特尔®。 请注意，编码器可能在您的系统上不可用，即使它在此列出。对于编解码器，您可以选择 `h264`（`libx264`、`h264_*`）、`hevc`（`libx265`、`hevc_*`）、`vp9`（`libvpx-vp9`、`vp9_qsv`）、`av1`（`libaom-av1`、`libsvtav1`、`av1_*`），和 `prores`。在麦金塔上，图形处理器加速的编解码器结尾为 `_videotoolbox`。
        - `Video bitrate control`：您可以设置视频码率和质量。您可以为了控制质量而选择 `VBR`（可变码率）和 `CQP`（或者 `CRF`，取决于编解码器）。
        - `Audio encoder`：您可以选择 `aac`（视频中最常用的音频编解码器）、`libopus`、`flac`（无损音频编解码器）、`pcm`（无损和无压缩）、`alac`（[Apple 保真压缩音频编解码器](https://support.apple.com/zh-cn/118295)）、`mp3`（有损音频编解码器），和 `vorbis`。码率控制不可用于无损音频编解码器。
6. 现在您可以开始渲染。预览窗口将会实时展示渲染帧，为了更好的渲染性能而一秒刷新两次。日志窗口的展示也将会减缓渲染过程，所以如无需要可以隐藏。

<details id="CannotOpen">
  <summary>麦金塔用户的注意事项</summary>

> 如果应用由于麦金塔的安全保护特性而无法启动，尝试下列做法：
> 
> 对于版本低于 15.0 的麦金塔：
> 
> 1. 右键点击 `mscz2video` 应用图标并且选中 <kbd> <samp class="menuitem">打开</samp> </kbd>。
> 2. 再次点击随后出现窗口中的 <kbd><samp class="buton">打开</samp> </kbd>。
> 
> 对于版本大于或等于 15.0 的麦金塔：
> 1. 在您的麦金塔上，前往 <kbd> <samp class="menu">系统设置</samp> > <samp class="submenu">隐私与安全性</samp> > 滚动到 <samp class="submenu">安全性</samp> 分区</kbd>。
> 2. 如果您见到声明 `已阻止“mscz2video.app”以保护 Mac。` 的消息——在此消息右方，点击 <kbd> <samp class="button">仍要打开</samp> </kbd>。
> 3. 输入您的登录密码，然后点击 <kbd> <samp class="button">好</samp> </kbd>。这将会在[门禁](https://support.apple.com/zh-cn/guide/security/sec5599b66df/web)中创建例外项目，允许 `mscz2video` 运行。
> 
> 带有截图的类似步骤见于我的另一项目 [Demucs-GUI](https://github.com/CarlGao4/Demucs-GUI#CannotOpen)。

</details>

## 命令行用法

1. 克隆本仓库。有两个必需文件：`mscz2video.py`（命令行界面文字转换器）和 `convert_core.py`（代码转换器），所以您还可以只直接下载这两个文件并且将它们放在同一路径中。
2. 创建 MuseScore 文件。您可以使用提供的示例文件 `Flower Dance.mscz`（要求 MuseScore 4.4 或者后续版本）。
3. 如果您不想要您的视频滚动屏幕，您需要将页面比例设为与视频分辨率相同。为此您可以前往将 <kbd> <samp class="menu">格式</samp> → <samp class="menuitem">页面设置</samp> → <samp class="submenu">页面尺寸</samp> → <samp class="submenu">自定义</samp> → <code>宽度</code> 和 <code>高度</code> </kbd> 设为您想要的比率。请勿将它们改得太小，因为默认输出尺寸是 360 像素每英寸。您还可以更改 `谱表间距` 和添加新谱行以使得每一页面更好地展示。我个人还推荐设置 <kbd> <samp class="menu">格式</samp> → <samp class="menuitem">样式</samp> → <samp class="submenu">页眉与页脚</samp> </kbd> 以使得奇数/偶数页面具有相同的页眉和页脚。
4. 准备 FFmpeg 与 MuseScore。您可以使用您的包管理器来安装它们，或者下载 [MuseScore](https://musescore.org) 和 [Gyan.dev 构建的 FFmpeg Windows](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip). 找到通往可执行的 `ffmpeg` 和 `MuseScore` 的路径。后续您需要用 `--ffmpeg-path` 和 `--musescore-path` 来设置通往 `ffmpeg` 和 `MuseScore` 的路径。
5. 转换文件之前，您可能需要学习一些 FFMpeg 的基础用法，因为此文字仅仅将帧和输出文件名称传到 FFMpeg 并且帧为 `RGB24` 格式但是通常视频被编码为 `YUV420p` 格式。您可能还需要此文字的帮助，方式是运行 `python3 mscz2video.py --help`，因为它有许多自定义输出视频的选项。详细信息参考[命令行参数](#command-line-arguments)。
6. 现在您可以转换该文件，方式是运行 `python3 mscz2video.py "Flower Dance.mscz" "Flower Dance.mp4" --ffmpeg-path "path/to/ffmpeg" --musescore-path "path/to/MuseScore" --start-offset 1 --end-offset 5 -r 30 -s 1920x1080 -j 4 --smooth-cursor "--" -i "Flower Dance.flac" -c:v libx265 -b:v 768k -c:a aac -b:a 128k -pix_fmt yuv420p -tag:v hvc1` 来创建 30 帧每秒、分辨率为 1920 × 1080，首音之前有 1 秒钟的等待时间、末音之后有 5 秒钟的等待时间、并行 4 个工作、匀速光标移动，和以 `libx265` 和 `aac` 编解码器编码的视频，正如上述视频。此文字不自动将音频加诸该视频，所以我添加了音频文件，方式是将附加参数传给 FFmpeg（`"--"` 之后的所有参数将不会解析而是直接传给 FFmpeg）。记得先从 MuseScore 手动导出音频文件。
7. 您还可以使用 PyTorch（支持图形处理器）以加快渲染。对于用法，您可以阅读文字的帮助。

## 命令行参数

```
用法: mscz2video.py [-h] [-r FPS] [-s SIZE] [--bar-color COLOR] [--bar-alpha UINT8] [--note-color COLOR] [--note-alpha UINT8] [--ffmpeg-path PATH] [--musescore-path PATH] [--start-offset FLOAT] [--end-offset FLOAT] [-ss FLOAT] [-t FLOAT] [--ffmpeg-help] [-j UINT] [--cache-limit UINT] [--use-torch] [--torch-devices STR] [--no-device-cache] [--resize-function {crop,rescale}] [--smooth-cursor] [--fixed-note-width [FLOAT]] [--extra-note-width-ratio FLOAT] [--version] input_mscz output_video
将 MuseScore 文件转换为视频

可用参数:
  input_mscz            输入 MuseScore 文件
  output_video          输出视频文件

选项:
  -h, --help                       展示此帮助消息并退出
  -r FPS, --fps FPS                帧率，默认为 60
  -s SIZE                          以宽x高 (比如 1920x1080) 计的分辨率，默认为首页尺寸
  --bar-color COLOR                当前小节的颜色，默认为红色，支持 3/6 位红绿蓝 (以 # 开头) 和超文本置标语言格式的颜色名称
  --bar-alpha UINT8                当前小节的透明度，默认为 85/255
  --note-color COLOR               当前音符的颜色，默认为青色，支持 3/6 位红绿蓝 (以 # 开头) 和超文本置标语言格式的颜色名称
  --note-alpha UINT8               当前音符的透明度，默认为 85/255
  --ffmpeg-path PATH               ffmpeg 路径，默认为 ffmpeg
  --musescore-path PATH            MuseScore 路径，默认为 musescore
  --start-offset FLOAT             首音之前的等待时间，默认为 0.0
  --end-offset FLOAT               末音之后的等待时间，默认为 0.0
  -ss FLOAT                        以秒计的开始时间偏移，默认为 0.0, 包括开端偏移 (start_offset=1 与 ss=1 将会导致无等待时间)
  -t FLOAT                         以秒计的时长，默认到歌曲结束
  --ffmpeg-help                    打印 FFmpeg 参数的帮助
  -j UINT, --jobs UINT             并行工作数目，默认为 1
  --cache-limit UINT               内存中同帧缓存限值，默认为 100
  --use-torch                      为图像处理使用 PyTorch, 更快且具有图形处理器支持
  --torch-devices STR              PyTorch 设备，与 colon 分离, 默认仅限中央处理器。您可以使用一个半角逗号来设置每一设备上的最大并行工作数目，比如 cuda:0,1;cpu,4 并且最大工作数目的总和必须大于或等于并行工作数目
  --no-device-cache                勿将原始图像缓存到每一设备。每次从内存加载。可能较慢但是所用设备内存较少。
  --resize-function {crop,rescale} 用来重设尺寸的函数，会将每一页面刈割为最大可能尺寸以相同比率，重新缩放会将每一页尺寸面重设为目标尺寸，默认为刈割
  --fixed-note-width [FLOAT]       不带此参数，音符高亮矩形的宽度将会调整为音符宽度。如果使用此参数而不带值或者设为 0，音符高亮矩形的宽度将会自动计算，或者是一个四分音符的宽度
  --extra-note-width-ratio FLOAT   额外音符高亮区域宽度比率，默认为 0.4，意味着将会在每一侧扩大目标音符的 20%
  --smooth-cursor                  匀速光标移动
```
