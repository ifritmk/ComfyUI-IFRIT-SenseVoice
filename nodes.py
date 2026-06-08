import json
import os
import uuid

import folder_paths


MODEL_CACHE = {}
DEFAULT_MODEL_DIR = os.path.join(folder_paths.models_dir, "SenseVoiceSmall")
MODEL_ID = "iic/SenseVoiceSmall"


def _import_error_message(error):
    return (
        "SenseVoice dependencies are not installed. Install them in ComfyUI's Python environment:\n"
        "F:\\code\\comfyui\\.ext\\python.exe -m pip install funasr modelscope torchaudio\n"
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


def _resolve_local_model():
    local_model = os.path.abspath(DEFAULT_MODEL_DIR)
    if os.path.isdir(local_model):
        return local_model

    try:
        from modelscope import snapshot_download
    except Exception as e:
        raise RuntimeError(_import_error_message(e))

    os.makedirs(local_model, exist_ok=True)
    print(f"[SenseVoice] Downloading {MODEL_ID} to {local_model}")
    snapshot_download(MODEL_ID, local_dir=local_model)
    if os.path.isdir(local_model):
        return local_model

    raise RuntimeError(
        "SenseVoiceSmall download failed or model folder is not available:\n"
        f"{local_model}"
    )


def _get_model(vad_model, device):
    local_model = _resolve_local_model()
    cache_key = (local_model, vad_model, device)
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    try:
        from funasr import AutoModel
    except Exception as e:
        raise RuntimeError(_import_error_message(e))

    kwargs = {
        "model": local_model,
        "device": device,
        "disable_update": True,
    }
    if vad_model and vad_model != "none":
        kwargs["vad_model"] = vad_model
        kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}

    model = AutoModel(**kwargs)
    MODEL_CACHE[cache_key] = model
    return model


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


def _normalize_result(result):
    if isinstance(result, list):
        items = result
    else:
        items = [result]

    texts = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("text")
            if text is None and isinstance(item.get("sentence_info"), list):
                text = "".join(str(sentence.get("text", "")) for sentence in item["sentence_info"])
            if text is not None:
                texts.append(str(text).strip())
        elif item is not None:
            texts.append(str(item).strip())

    text = "\n".join(part for part in texts if part)
    raw_json = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    return text, raw_json


class SenseVoiceTranscribeAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "language": (["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],),
                "device": (["auto", "cuda:0", "cpu"],),
                "use_itn": ("BOOLEAN", {"default": True}),
                "batch_size_s": ("INT", {"default": 60, "min": 1, "max": 600, "step": 1}),
            },
            "optional": {
                "vad_model": (["fsmn-vad", "none"],),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "raw_json")
    FUNCTION = "transcribe"
    CATEGORY = "SenseVoice"

    def transcribe(self, audio, language, device, use_itn, batch_size_s, vad_model="fsmn-vad"):
        audio_path = _save_audio_to_temp(audio)
        try:
            return SenseVoiceTranscribeFile().transcribe(
                audio_path,
                language,
                device,
                use_itn,
                batch_size_s,
                vad_model,
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
                "language": (["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],),
                "device": (["auto", "cuda:0", "cpu"],),
                "use_itn": ("BOOLEAN", {"default": True}),
                "batch_size_s": ("INT", {"default": 60, "min": 1, "max": 600, "step": 1}),
            },
            "optional": {
                "vad_model": (["fsmn-vad", "none"],),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "raw_json")
    FUNCTION = "transcribe"
    CATEGORY = "SenseVoice"

    def transcribe(self, audio_path, language, device, use_itn, batch_size_s, vad_model="fsmn-vad"):
        if not audio_path:
            raise RuntimeError("audio_path is empty.")

        audio_path = folder_paths.get_annotated_filepath(audio_path)
        if not os.path.exists(audio_path):
            raise RuntimeError(f"Audio file not found: {audio_path}")

        infer_device = _get_device(device)
        recognizer = _get_model(vad_model, infer_device)
        language_arg = "auto" if language == "auto" else language

        result = recognizer.generate(
            input=audio_path,
            cache={},
            language=language_arg,
            use_itn=bool(use_itn),
            batch_size_s=int(batch_size_s),
            merge_vad=True,
            merge_length_s=15,
        )
        return _normalize_result(result)


NODE_CLASS_MAPPINGS = {
    "SenseVoiceTranscribeAudio": SenseVoiceTranscribeAudio,
    "SenseVoiceTranscribeFile": SenseVoiceTranscribeFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SenseVoiceTranscribeAudio": "SenseVoice Transcribe Audio",
    "SenseVoiceTranscribeFile": "SenseVoice Transcribe File",
}
