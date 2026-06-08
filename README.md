# ComfyUI FunASR

ComfyUI custom nodes for testing FunASR speech recognition models with SRT output.

## Nodes

- `FunASR Transcribe Audio`
  - Input: ComfyUI `AUDIO`
  - Output: recognized text and SRT
- `FunASR Transcribe File`
  - Input: audio file path
  - Output: recognized text and SRT

## Model Choices

The node exposes a `model` dropdown:

```text
Fun-ASR-Nano-2512
SenseVoiceSmall
```

Use `Fun-ASR-Nano-2512` for recognition with timestamps.

Use `SenseVoiceSmall` for recognition with emotion and event tags.

SenseVoiceSmall can also produce timestamped sentence output through the FunASR pipeline when `vad_model=fsmn-vad` and `punc_model=ct-punc` are enabled. Enable `spk_model=cam++` as well to add speaker labels.

`Fun-ASR-Nano-2512` is not registered by FunASR 1.1.x by default. The node automatically downloads the small runtime files from `FunAudioLLM/Fun-ASR` into:

```text
F:\code\comfyui\models\Fun-ASR-Nano-2512\runtime
```

The model weights stay in `F:\code\comfyui\models\Fun-ASR-Nano-2512`.

## Local Model Paths

Models are stored under ComfyUI's `models` folder:

```text
F:\code\comfyui\models\Fun-ASR-Nano-2512
F:\code\comfyui\models\SenseVoiceSmall
```

If a folder does not exist, the plugin downloads the selected model to its fixed local folder.

When `vad_model` is `fsmn-vad`, the VAD model is kept under the selected model folder:

```text
F:\code\comfyui\models\Fun-ASR-Nano-2512\fsmn-vad
F:\code\comfyui\models\SenseVoiceSmall\fsmn-vad
```

When `punc_model` or `spk_model` is enabled for `SenseVoiceSmall`, helper models are stored here:

```text
F:\code\comfyui\models\SenseVoiceSmall\ct-punc
F:\code\comfyui\models\SenseVoiceSmall\cam++
```

For offline deployment, pre-place the `Fun-ASR-Nano-2512` model folder and the `runtime` folder above.

## Dependencies

Install dependencies in ComfyUI's Python environment:

```powershell
F:\code\comfyui\.ext\python.exe -m pip install funasr huggingface_hub modelscope transformers torchaudio
```

The plugin skips FunASR's per-model `requirements.txt` auto-install step at runtime so it does not spawn `pip` while ComfyUI is running.

## Inputs

- `model`: `Fun-ASR-Nano-2512` or `SenseVoiceSmall`
- `language`: `auto`, `zh`, `en`, `yue`, `ja`, `ko`, `nospeech`
- `device`: `auto`, `cuda:0`, `cpu`
- `use_itn`: inverse text normalization
- `batch_size_s`: inference batch size in seconds
- `vad_model`: `fsmn-vad` or `none`
- `punc_model`: `ct-punc` or `none`
- `spk_model`: `cam++` or `none`

## Outputs

- `text`: merged recognized text
- `srt`: subtitle text in SRT format. The node reads FunASR `sentence_info`, `timestamp`, `timestamps`, and `ctc_timestamps` fields when available; if no timestamps are returned, it emits one full-duration subtitle block.
