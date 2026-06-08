# ComfyUI FunASR

这是一个 ComfyUI 自定义节点插件，用于调用 FunASR 做语音识别，并在模型返回真实时间戳时输出 SRT 字幕。

## 节点

- `FunASR Transcribe Audio`
  - 输入：ComfyUI `AUDIO`
  - 输出：识别文本、SRT 字幕
- `FunASR Transcribe File`
  - 输入：音频文件路径
  - 输出：识别文本、SRT 字幕

## 模型选择

节点的 `model` 下拉框包含：

```text
Paraformer-Large
SenseVoiceSmall
```

`Paraformer-Large` 适合中文识别和时间戳/SRT 输出。

`SenseVoiceSmall` 适合语音识别、情感标签和事件标签。

## 本地模型目录

模型固定放在 ComfyUI 的 `models` 目录下：

```text
F:\code\comfyui\models\Paraformer-Large
F:\code\comfyui\models\SenseVoiceSmall
```

如果目录不存在，节点会把选中的模型下载到对应固定目录。

`Paraformer-Large` 会自动搭配 `fsmn-vad` 进行长音频分段，VAD 模型会放在：

```text
F:\code\comfyui\models\Paraformer-Large\fsmn-vad
F:\code\comfyui\models\SenseVoiceSmall\fsmn-vad
```

离线部署时，请提前把上述模型目录放好。

## 依赖

在 ComfyUI 的 Python 环境中安装依赖：

```powershell
F:\code\comfyui\.ext\python.exe -m pip install funasr huggingface_hub modelscope transformers torchaudio
```

建议使用较新的 FunASR 版本。当前节点按 FunASR 1.3.x 的返回结构适配。

节点运行时会跳过 FunASR 模型目录里的 `requirements.txt` 自动安装步骤，避免 ComfyUI 运行过程中额外拉起 pip。

## 输入参数

- `model`：选择 `Paraformer-Large` 或 `SenseVoiceSmall`
- `device`：推理设备，支持 `auto`、`cuda:0`、`cpu`
- `batch_size_s`：推理批处理时长，单位秒
- `unload_model`：任务结束后是否释放已缓存模型

## 输出

- `text`：合并后的识别文本
- `srt`：SRT 字幕文本

SRT 只会在 FunASR 返回真实时间戳时生成。节点会读取以下字段：

```text
sentence_info
timestamp
timestamps
ctc_timestamps
```

如果模型没有返回真实时间戳，`srt` 会为空，不会用音频总时长硬切假字幕。

当前 SRT 生成逻辑参考 `ComfyUI-AV-FunASR`：优先使用 FunASR 返回的 `timestamp`，再把识别文本按真实时间戳聚合成字幕段；不会平均分配音频时长。

## 使用建议

中文字幕优先使用：

```text
model=Paraformer-Large
batch_size_s=300
```

如果只想要 SenseVoice 的情感/事件标签，可以使用：

```text
model=SenseVoiceSmall
```
