import json
import os
import re
import sys
import urllib.request
import uuid

import folder_paths


MODEL_CACHE = {}
NANO_RUNTIME_IMPORTED = False
MODEL_CHOICES = ["Fun-ASR-Nano-2512", "SenseVoiceSmall"]
MODEL_CONFIGS = {
    "Fun-ASR-Nano-2512": {
        "id": "FunAudioLLM/Fun-ASR-Nano-2512",
        "dir": os.path.join(folder_paths.models_dir, "Fun-ASR-Nano-2512"),
        "runtime_url": "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main",
        "hub": "hf",
        "trust_remote_code": True,
        "itn_arg": "itn",
        "sentence_timestamp": True,
    },
    "SenseVoiceSmall": {
        "id": "iic/SenseVoiceSmall",
        "dir": os.path.join(folder_paths.models_dir, "SenseVoiceSmall"),
        "hub": "ms",
        "trust_remote_code": False,
        "itn_arg": "use_itn",
        "sentence_timestamp": True,
    },
}
VAD_MODEL_IDS = {
    "hf": "funasr/fsmn-vad",
    "ms": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
}
PUNC_MODEL_IDS = {
    "hf": "funasr/ct-punc",
    "ms": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
}
SPK_MODEL_IDS = {
    "hf": "funasr/campplus",
    "ms": "iic/speech_campplus_sv_zh-cn_16k-common",
}
NANO_RUNTIME_FILES = ("model.py", "ctc.py", "tools/utils.py")


def _import_error_message(error):
    return (
        "FunASR dependencies are not installed. Install them in ComfyUI's Python environment:\n"
        "F:\\code\\comfyui\\.ext\\python.exe -m pip install funasr huggingface_hub transformers torchaudio\n"
        f"Original error: {error}"
    )


def _get_device(device):
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _snapshot_download(model_id, local_dir, hub, label):
    if hub == "hf":
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise RuntimeError(_import_error_message(e))
        print(f"[FunASR] Downloading {label} to {local_dir}")
        snapshot_download(repo_id=model_id, local_dir=local_dir)
        return

    try:
        from modelscope import snapshot_download
    except Exception as e:
        raise RuntimeError(_import_error_message(e))
    print(f"[FunASR] Downloading {label} to {local_dir}")
    snapshot_download(model_id, local_dir=local_dir)


def _model_config(model_choice):
    if model_choice not in MODEL_CONFIGS:
        raise RuntimeError(f"Unsupported FunASR model: {model_choice}")
    return MODEL_CONFIGS[model_choice]


def _resolve_local_model(model_choice):
    config = _model_config(model_choice)
    local_model = os.path.abspath(config["dir"])
    if os.path.isdir(local_model):
        return local_model

    os.makedirs(local_model, exist_ok=True)
    _snapshot_download(config["id"], local_model, config["hub"], model_choice)
    if os.path.isdir(local_model):
        return local_model

    raise RuntimeError(
        f"{model_choice} download failed or model folder is not available:\n"
        f"{local_model}"
    )


def _ensure_model(model_id, local_dir, hub, label):
    local_dir = os.path.abspath(local_dir)
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        return local_dir

    os.makedirs(local_dir, exist_ok=True)
    _snapshot_download(model_id, local_dir, hub, label)
    return local_dir


def _download_text_file(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[FunASR] Downloading runtime file: {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    with open(path, "wb") as file:
        file.write(data)


def _ensure_nano_runtime(config, local_model):
    runtime_dir = os.path.join(local_model, "runtime")
    runtime_url = config["runtime_url"].rstrip("/")
    missing = []
    for name in NANO_RUNTIME_FILES:
        target = os.path.join(runtime_dir, *name.split("/"))
        if not os.path.exists(target):
            missing.append((name, target))

    for name, target in missing:
        _download_text_file(f"{runtime_url}/{name}", target)

    tools_dir = os.path.join(runtime_dir, "tools")
    os.makedirs(tools_dir, exist_ok=True)
    init_file = os.path.join(tools_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w", encoding="utf-8") as file:
            file.write("")

    model_code = os.path.join(runtime_dir, "model.py")
    if not os.path.exists(model_code):
        raise RuntimeError(
            "Fun-ASR-Nano-2512 runtime code is missing. Expected:\n"
            f"{model_code}"
        )
    return model_code.replace("\\", "/")


def _import_nano_runtime(model_code):
    global NANO_RUNTIME_IMPORTED
    if NANO_RUNTIME_IMPORTED:
        return

    runtime_dir = os.path.dirname(model_code)
    if runtime_dir not in sys.path:
        sys.path.insert(0, runtime_dir)

    try:
        from funasr.utils.dynamic_import import import_module_from_path
        from funasr.register import tables
    except Exception as e:
        raise RuntimeError(_import_error_message(e))

    import_module_from_path(model_code)
    if tables.model_classes.get("FunASRNano") is None:
        raise RuntimeError(
            "Fun-ASR-Nano-2512 runtime loaded, but FunASRNano was not registered.\n"
            f"runtime: {model_code}"
        )
    NANO_RUNTIME_IMPORTED = True


def _optional_pipeline_model(config, choice, local_dir_name, ids, label):
    if not choice or choice == "none":
        return None
    model_id = ids[config["hub"]]
    model_dir = os.path.join(config["dir"], local_dir_name)
    return _ensure_model(model_id, model_dir, config["hub"], label)


def _get_model(model_choice, vad_model, punc_model, spk_model, device):
    config = _model_config(model_choice)
    local_model = _resolve_local_model(model_choice)
    remote_code = None
    if model_choice == "Fun-ASR-Nano-2512":
        remote_code = _ensure_nano_runtime(config, local_model)
        _import_nano_runtime(remote_code)

    local_vad_model = None
    local_punc_model = None
    local_spk_model = None
    if vad_model and vad_model != "none":
        vad_model_id = VAD_MODEL_IDS[config["hub"]]
        vad_model_dir = os.path.join(config["dir"], "fsmn-vad")
        if model_choice == "Fun-ASR-Nano-2512" and not (
            os.path.isdir(vad_model_dir) and os.listdir(vad_model_dir)
        ):
            print(f"[FunASR] Skip fsmn-vad for {model_choice}; local VAD model not found: {vad_model_dir}")
        else:
            local_vad_model = _ensure_model(vad_model_id, vad_model_dir, config["hub"], "fsmn-vad")
    if model_choice == "SenseVoiceSmall":
        local_punc_model = _optional_pipeline_model(config, punc_model, "ct-punc", PUNC_MODEL_IDS, "ct-punc")
        local_spk_model = _optional_pipeline_model(config, spk_model, "cam++", SPK_MODEL_IDS, "cam++")
        if local_spk_model and not local_punc_model:
            raise RuntimeError("spk_model=cam++ requires punc_model=ct-punc.")

    cache_key = (
        model_choice,
        local_model,
        local_vad_model or "none",
        local_punc_model or "none",
        local_spk_model or "none",
        device,
    )
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    try:
        from funasr import AutoModel
        import funasr.utils.install_model_requirements as install_model_requirements
    except Exception as e:
        raise RuntimeError(_import_error_message(e))

    def skip_model_requirements(requirements_path):
        print(f"[FunASR] Skip model requirements install: {requirements_path}")
        return True

    install_model_requirements.install_requirements = skip_model_requirements

    kwargs = {
        "model": local_model,
        "hub": config["hub"],
        "device": device,
        "disable_update": True,
    }
    if config["trust_remote_code"]:
        kwargs["trust_remote_code"] = True
    if remote_code:
        kwargs["remote_code"] = remote_code
    if local_vad_model:
        kwargs["vad_model"] = local_vad_model
        kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
    if local_punc_model:
        kwargs["punc_model"] = local_punc_model
    if local_spk_model:
        kwargs["spk_model"] = local_spk_model
        kwargs["spk_mode"] = "punc_segment"

    model = AutoModel(**kwargs)
    MODEL_CACHE[cache_key] = model
    return model


def _audio_duration(audio):
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        return None
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        return None
    return float(waveform.shape[-1]) / float(sample_rate)


def _audio_file_duration(audio_path):
    try:
        import torchaudio

        info = torchaudio.info(audio_path)
        if info.sample_rate > 0 and info.num_frames > 0:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass
    return None


def _save_audio_to_temp(audio):
    try:
        import torchaudio
    except Exception as e:
        raise RuntimeError(_import_error_message(e))

    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise RuntimeError("Invalid ComfyUI AUDIO input.")

    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    audio_path = os.path.join(temp_dir, f"sensevoice_{uuid.uuid4().hex}.wav")
    waveform = audio["waveform"]
    if waveform.ndim == 3:
        waveform = waveform.squeeze(0)
    torchaudio.save(audio_path, waveform.cpu(), int(audio["sample_rate"]))
    return audio_path


def _seconds_from_any(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value >= 1000:
        value = value / 1000.0
    return max(0.0, value)


def _srt_timestamp(seconds):
    seconds = max(0.0, float(seconds or 0.0))
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _seconds_from_milliseconds(value):
    if value is None:
        return None
    try:
        return max(0.0, float(value) / 1000.0)
    except (TypeError, ValueError):
        return None


def _append_srt_entry(entries, text, start, end, milliseconds=False):
    text = str(text or "").strip()
    if milliseconds:
        start = _seconds_from_milliseconds(start)
        end = _seconds_from_milliseconds(end)
    else:
        start = _seconds_from_any(start)
        end = _seconds_from_any(end)
    if not text or start is None or end is None:
        return
    if end <= start:
        end = start + 0.5
    entries.append((start, end, text))


def _first_present(item, *keys):
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _is_sentence_break(token):
    return token in {".", "!", "?", "\u3002", "\uff01", "\uff1f"}


def _collect_srt_entries_from_item(item, entries):
    if not isinstance(item, dict):
        return

    sentence_info = item.get("sentence_info")
    if isinstance(sentence_info, list):
        for sentence in sentence_info:
            if not isinstance(sentence, dict):
                continue
            text = sentence.get("text", sentence.get("sentence", ""))
            speaker = _first_present(sentence, "speaker", "spk", "spk_id")
            if speaker is not None:
                text = f"Speaker {speaker}: {text}"
            _append_srt_entry(
                entries,
                text,
                _first_present(sentence, "start", "start_time"),
                _first_present(sentence, "end", "end_time"),
                milliseconds=True,
            )

    timestamps = item.get("timestamp")
    words = item.get("words")
    if isinstance(timestamps, list):
        for index, chunk in enumerate(timestamps):
            if not isinstance(chunk, (list, tuple)) or len(chunk) < 2:
                continue
            if len(chunk) > 2:
                text = chunk[2]
            elif isinstance(words, list) and index < len(words):
                text = words[index]
            else:
                continue
            _append_srt_entry(entries, text, chunk[0], chunk[1], milliseconds=True)

    for timestamp_key, text_key in (("timestamps", "text"), ("ctc_timestamps", "ctc_text")):
        token_timestamps = item.get(timestamp_key)
        if not isinstance(token_timestamps, list):
            continue
        token_entries = []
        for timestamp in token_timestamps:
            if not isinstance(timestamp, dict):
                continue
            token = timestamp.get("token", "")
            start = _seconds_from_any(_first_present(timestamp, "start", "start_time"))
            end = _seconds_from_any(_first_present(timestamp, "end", "end_time"))
            if token and start is not None and end is not None:
                token_entries.append((start, end, str(token)))
        if not token_entries:
            continue

        group_start = None
        group_end = None
        group_tokens = []
        for start, end, token in token_entries:
            if group_start is None:
                group_start = start
            group_end = end
            group_tokens.append(token)
            text = "".join(group_tokens)
            if _is_sentence_break(token) or (group_end - group_start) >= 6.0 or len(group_tokens) >= 24:
                _append_srt_entry(entries, re.sub(r"\s+", " ", text).strip(), group_start, group_end)
                group_start = None
                group_end = None
                group_tokens = []
        if group_tokens:
            text = "".join(group_tokens)
            _append_srt_entry(entries, re.sub(r"\s+", " ", text).strip(), group_start, group_end)
        return


def _build_srt(result, fallback_text="", fallback_duration=None):
    items = result if isinstance(result, list) else [result]
    entries = []
    for item in items:
        _collect_srt_entries_from_item(item, entries)

    entries.sort(key=lambda entry: (entry[0], entry[1]))
    blocks = []
    for index, (start, end, text) in enumerate(entries, start=1):
        blocks.append(
            f"{index}\n"
            f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n"
            f"{text}"
        )
    if blocks:
        return "\n\n".join(blocks)

    fallback_text = str(fallback_text or "").strip()
    fallback_duration = _seconds_from_any(fallback_duration)
    if fallback_text and fallback_duration and fallback_duration > 0:
        return (
            "1\n"
            f"00:00:00,000 --> {_srt_timestamp(fallback_duration)}\n"
            f"{fallback_text}"
        )
    return ""


def _normalize_result(result, fallback_duration=None):
    if isinstance(result, list):
        items = result
    else:
        items = [result]

    texts = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("text")
            if text is None and isinstance(item.get("sentence_info"), list):
                text = "".join(
                    str(sentence.get("text", sentence.get("sentence", "")))
                    for sentence in item["sentence_info"]
                )
            if text is not None:
                texts.append(str(text).strip())
        elif item is not None:
            texts.append(str(item).strip())

    text = "\n".join(part for part in texts if part)
    srt = _build_srt(result, text, fallback_duration)
    return text, srt


class SenseVoiceTranscribeAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (MODEL_CHOICES, {"default": "Fun-ASR-Nano-2512"}),
                "language": (["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],),
                "device": (["auto", "cuda:0", "cpu"],),
                "use_itn": ("BOOLEAN", {"default": True}),
                "batch_size_s": ("INT", {"default": 60, "min": 1, "max": 600, "step": 1}),
            },
            "optional": {
                "vad_model": (["none", "fsmn-vad"],),
                "punc_model": (["none", "ct-punc"],),
                "spk_model": (["none", "cam++"],),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "srt")
    FUNCTION = "transcribe"
    CATEGORY = "FunASR"

    def transcribe(
        self,
        audio,
        model,
        language,
        device,
        use_itn,
        batch_size_s,
        vad_model="none",
        punc_model="none",
        spk_model="none",
    ):
        duration = _audio_duration(audio)
        audio_path = _save_audio_to_temp(audio)
        try:
            return SenseVoiceTranscribeFile().transcribe(
                audio_path,
                model,
                language,
                device,
                use_itn,
                batch_size_s,
                vad_model,
                punc_model,
                spk_model,
                duration,
            )
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass


class SenseVoiceTranscribeFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {"default": ""}),
                "model": (MODEL_CHOICES, {"default": "Fun-ASR-Nano-2512"}),
                "language": (["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],),
                "device": (["auto", "cuda:0", "cpu"],),
                "use_itn": ("BOOLEAN", {"default": True}),
                "batch_size_s": ("INT", {"default": 60, "min": 1, "max": 600, "step": 1}),
            },
            "optional": {
                "vad_model": (["none", "fsmn-vad"],),
                "punc_model": (["none", "ct-punc"],),
                "spk_model": (["none", "cam++"],),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "srt")
    FUNCTION = "transcribe"
    CATEGORY = "FunASR"

    def transcribe(
        self,
        audio_path,
        model,
        language,
        device,
        use_itn,
        batch_size_s,
        vad_model="none",
        punc_model="none",
        spk_model="none",
        audio_duration=None,
    ):
        if not audio_path:
            raise RuntimeError("audio_path is empty.")

        audio_path = folder_paths.get_annotated_filepath(audio_path)
        if not os.path.exists(audio_path):
            raise RuntimeError(f"Audio file not found: {audio_path}")
        if audio_duration is None:
            audio_duration = _audio_file_duration(audio_path)

        infer_device = _get_device(device)
        config = _model_config(model)
        recognizer = _get_model(model, vad_model, punc_model, spk_model, infer_device)
        language_arg = "auto" if language == "auto" else language

        generate_kwargs = {
            "input": audio_path,
            "cache": {},
            "language": language_arg,
            config["itn_arg"]: bool(use_itn),
            "batch_size_s": int(batch_size_s),
            "merge_vad": True,
            "merge_length_s": 15,
        }
        if model == "Fun-ASR-Nano-2512":
            generate_kwargs["batch_size"] = 1
            generate_kwargs["batch_size_s"] = 1
        if (
            model == "Fun-ASR-Nano-2512"
            and config["sentence_timestamp"]
            and getattr(recognizer, "punc_model", None) is not None
        ):
            generate_kwargs["sentence_timestamp"] = True

        result = recognizer.generate(**generate_kwargs)
        return _normalize_result(result, audio_duration)


NODE_CLASS_MAPPINGS = {
    "SenseVoiceTranscribeAudio": SenseVoiceTranscribeAudio,
    "SenseVoiceTranscribeFile": SenseVoiceTranscribeFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SenseVoiceTranscribeAudio": "FunASR Transcribe Audio",
    "SenseVoiceTranscribeFile": "FunASR Transcribe File",
}
