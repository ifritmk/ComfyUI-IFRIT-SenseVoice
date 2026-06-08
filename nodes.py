import difflib
import os
import re
import uuid

import folder_paths


MODEL_CACHE = {}
MODEL_CONFIGS = {
    "Paraformer-Large": {
        "id": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "dir": os.path.join(folder_paths.models_dir, "Paraformer-Large"),
        "hub": "ms",
        "trust_remote_code": False,
        "itn_arg": "use_itn",
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


def _get_model(model_choice, device):
    config = _model_config(model_choice)
    local_model = _resolve_local_model(model_choice)

    local_vad_model = None
    if model_choice == "Paraformer-Large":
        vad_model_id = VAD_MODEL_IDS[config["hub"]]
        vad_model_dir = os.path.join(config["dir"], "fsmn-vad")
        local_vad_model = _ensure_model(vad_model_id, vad_model_dir, config["hub"], "fsmn-vad")

    cache_key = (
        model_choice,
        local_model,
        local_vad_model or "none",
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
    if local_vad_model:
        kwargs["vad_model"] = local_vad_model
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
    audio_path = os.path.join(temp_dir, f"funasr_{uuid.uuid4().hex}.wav")
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


def _append_srt_entry_auto_time(entries, text, start, end):
    try:
        start_value = float(start)
        end_value = float(end)
    except (TypeError, ValueError):
        return
    _append_srt_entry(entries, text, start, end, milliseconds=max(start_value, end_value) > 100)


def _first_present(item, *keys):
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _is_sentence_break(token):
    return token in {".", "!", "?", "\u3002", "\uff01", "\uff1f", "\uff1b", ";"}


def _clean_asr_text(text):
    text = str(text or "")
    text = re.sub(r"<\|[^|]+?\|>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_text_for_srt(text, max_chars=28):
    text = _clean_asr_text(text)
    if not text:
        return []

    parts = []
    current = ""
    soft_breaks = {"\uff0c", "\u3001", ","}
    for char in text:
        current += char
        if _is_sentence_break(char) or char in soft_breaks or len(current) >= max_chars:
            part = current.strip()
            if part:
                parts.append(part)
            current = ""
    current = current.strip()
    if current:
        parts.append(current)
    return parts


def _chars_no_space(text):
    return [char for char in _clean_asr_text(text) if not char.isspace()]


def _split_text_by_weights(text, weights):
    chars = _chars_no_space(text)
    if not chars or not weights:
        return []

    total_weight = max(1, sum(max(1, int(weight)) for weight in weights))
    cursor = 0
    parts = []
    for index, weight in enumerate(weights):
        if index == len(weights) - 1:
            end = len(chars)
        else:
            end = round(len(chars) * sum(max(1, int(value)) for value in weights[: index + 1]) / total_weight)
            end = max(cursor + 1, min(end, len(chars) - (len(weights) - index - 1)))
        parts.append("".join(chars[cursor:end]).strip())
        cursor = end
    return parts


def _append_weighted_srt_entries(entries, text, time_ranges):
    parts = _split_text_for_srt(text)
    if not parts or not time_ranges:
        return False

    clean_ranges = []
    for start, end in time_ranges:
        start = _seconds_from_any(start)
        end = _seconds_from_any(end)
        if start is not None and end is not None and end > start:
            clean_ranges.append((start, end))
    if not clean_ranges:
        return False

    clean_ranges.sort(key=lambda value: (value[0], value[1]))
    total_chars = max(1, sum(len(part) for part in parts))
    range_index = 0
    for index, part in enumerate(parts):
        remaining_parts = len(parts) - index
        remaining_ranges = len(clean_ranges) - range_index
        if remaining_ranges <= 0:
            break
        if remaining_ranges >= remaining_parts:
            wanted = max(1, round(len(part) / total_chars * len(clean_ranges)))
            count = min(wanted, remaining_ranges - remaining_parts + 1)
            chosen = clean_ranges[range_index : range_index + count]
            range_index += count
            _append_srt_entry(entries, part, chosen[0][0], chosen[-1][1])
            continue

        timeline_start = clean_ranges[range_index][0]
        timeline_end = clean_ranges[-1][1]
        remaining_text_chars = max(1, sum(len(value) for value in parts[index:]))
        cursor = timeline_start
        for tail_index, tail_part in enumerate(parts[index:], start=index):
            duration = (timeline_end - timeline_start) * len(tail_part) / remaining_text_chars
            end = timeline_end if tail_index == len(parts) - 1 else cursor + max(0.6, duration)
            _append_srt_entry(entries, tail_part, cursor, min(end, timeline_end))
            cursor = min(end, timeline_end)
        return True

    return True


def _split_text_by_reference(reference_text, target_text, max_chars=28):
    target_chars = _chars_no_space(target_text)
    if not target_chars:
        return []

    reference_parts = _split_text_for_srt(reference_text, max_chars=max_chars)
    if not reference_parts:
        return _split_text_for_srt(target_text, max_chars=max_chars)

    reference_chars = _chars_no_space(reference_text)
    if not reference_chars:
        weights = [len(_chars_no_space(part)) for part in reference_parts]
        return _split_text_by_weights(target_text, weights)

    matcher = difflib.SequenceMatcher(None, reference_chars, target_chars, autojunk=False)
    mapped_positions = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"equal", "replace"}:
            length = min(i2 - i1, j2 - j1)
            for offset in range(length):
                mapped_positions[i1 + offset] = j1 + offset

    parts = []
    ref_cursor = 0
    target_cursor = 0
    for index, reference_part in enumerate(reference_parts):
        ref_len = len(_chars_no_space(reference_part))
        ref_end = ref_cursor + ref_len
        if index == len(reference_parts) - 1:
            target_end = len(target_chars)
        else:
            candidates = [
                mapped_positions[pos]
                for pos in range(ref_cursor, ref_end)
                if pos in mapped_positions
            ]
            if candidates:
                target_end = max(candidates) + 1
            else:
                target_end = round(len(target_chars) * ref_end / max(1, len(reference_chars)))
            target_end = max(target_cursor + 1, min(target_end, len(target_chars) - (len(reference_parts) - index - 1)))
        parts.append("".join(target_chars[target_cursor:target_end]).strip())
        target_cursor = target_end
        ref_cursor = ref_end

    return [part for part in parts if part]


def _append_timestamp_text_entries(entries, text, timestamps):
    text = _clean_asr_text(text)
    if not text:
        return False

    clean_timestamps = []
    for chunk in timestamps:
        if isinstance(chunk, (list, tuple)) and len(chunk) >= 2:
            start = _seconds_from_milliseconds(chunk[0])
            end = _seconds_from_milliseconds(chunk[1])
            if start is not None and end is not None:
                clean_timestamps.append((start, end))
    if not clean_timestamps:
        return False

    tokens = text.split()
    if not tokens:
        tokens = [char for char in text if not char.isspace()]
    if not tokens:
        return False

    group_start = None
    group_end = None
    group_tokens = []
    usable = min(len(tokens), len(clean_timestamps))
    for index in range(usable):
        token = tokens[index]
        start, end = clean_timestamps[index]
        if group_start is None:
            group_start = start
        group_end = end
        group_tokens.append(token)
        text_part = "".join(group_tokens)
        if _is_sentence_break(token[-1]) or (group_end - group_start) >= 3.0 or len(text_part) >= 22:
            _append_srt_entry(entries, text_part, group_start, group_end)
            group_start = None
            group_end = None
            group_tokens = []
    if group_tokens:
        _append_srt_entry(entries, "".join(group_tokens), group_start, group_end)
    return True


def _extract_timed_text_segments_from_item(item):
    if not isinstance(item, dict):
        return []

    segments = []
    sentence_info = item.get("sentence_info")
    if isinstance(sentence_info, list):
        for sentence in sentence_info:
            if not isinstance(sentence, dict):
                continue
            text = _clean_asr_text(sentence.get("text", sentence.get("sentence", "")))
            start = _first_present(sentence, "start", "start_time")
            end = _first_present(sentence, "end", "end_time")
            try:
                start_value = float(start)
                end_value = float(end)
            except (TypeError, ValueError):
                continue
            if max(start_value, end_value) > 100:
                start_value = _seconds_from_milliseconds(start_value)
                end_value = _seconds_from_milliseconds(end_value)
            else:
                start_value = _seconds_from_any(start_value)
                end_value = _seconds_from_any(end_value)
            if text and start_value is not None and end_value is not None and end_value > start_value:
                segments.append((start_value, end_value, text))

    timestamps = item.get("timestamp")
    words = item.get("words")
    item_text = _clean_asr_text(item.get("text", ""))
    if isinstance(timestamps, list):
        timestamp_ranges = []
        timestamp_texts = []
        for index, chunk in enumerate(timestamps):
            if not isinstance(chunk, (list, tuple)) or len(chunk) < 2:
                continue
            start = _seconds_from_milliseconds(chunk[0])
            end = _seconds_from_milliseconds(chunk[1])
            if start is None or end is None or end <= start:
                continue
            if len(chunk) > 2:
                text = _clean_asr_text(chunk[2])
            elif isinstance(words, list) and index < len(words):
                text = _clean_asr_text(words[index])
            else:
                text = ""
            timestamp_ranges.append((start, end))
            timestamp_texts.append(text)
        if any(timestamp_texts):
            group_start = None
            group_end = None
            group_text = ""
            for (start, end), token in zip(timestamp_ranges, timestamp_texts):
                if group_start is None:
                    group_start = start
                group_end = end
                group_text += token
                if _is_sentence_break(token[-1:]) or (group_end - group_start) >= 3.0 or len(group_text) >= 28:
                    segments.append((group_start, group_end, group_text))
                    group_start = None
                    group_end = None
                    group_text = ""
            if group_text:
                segments.append((group_start, group_end, group_text))
        elif item_text and timestamp_ranges:
            temp_entries = []
            if _append_timestamp_text_entries(temp_entries, item_text, timestamps):
                segments.extend(temp_entries)
            else:
                weights = [max(1, len(part)) for part in _split_text_for_srt(item_text)]
                parts = _split_text_by_weights(item_text, weights)
                for part, (start, end) in zip(parts, timestamp_ranges):
                    segments.append((start, end, part))

    segments.sort(key=lambda value: (value[0], value[1]))
    return segments


def _append_aligned_srt_entries(entries, timestamp_result, text):
    items = timestamp_result if isinstance(timestamp_result, list) else [timestamp_result]
    timed_segments = []
    for item in items:
        timed_segments.extend(_extract_timed_text_segments_from_item(item))
    timed_segments = [
        (start, end, segment_text)
        for start, end, segment_text in timed_segments
        if segment_text and end > start
    ]
    if len(timed_segments) < 2:
        return False

    reference_text = "".join(segment_text for _, _, segment_text in timed_segments)
    aligned_parts = _split_text_by_reference(reference_text, text)
    if not aligned_parts:
        weights = [len(_chars_no_space(segment_text)) for _, _, segment_text in timed_segments]
        aligned_parts = _split_text_by_weights(text, weights)
    if not aligned_parts:
        return False

    if len(aligned_parts) != len(timed_segments):
        weights = [len(_chars_no_space(segment_text)) for _, _, segment_text in timed_segments]
        aligned_parts = _split_text_by_weights(text, weights)

    for (start, end, _), part in zip(timed_segments, aligned_parts):
        _append_srt_entry(entries, part, start, end)
    return bool(entries)


def _extract_time_ranges_from_item(item):
    if not isinstance(item, dict):
        return []

    ranges = []
    sentence_info = item.get("sentence_info")
    if isinstance(sentence_info, list):
        for sentence in sentence_info:
            if not isinstance(sentence, dict):
                continue
            start = _first_present(sentence, "start", "start_time")
            end = _first_present(sentence, "end", "end_time")
            try:
                start_value = float(start)
                end_value = float(end)
            except (TypeError, ValueError):
                continue
            if max(start_value, end_value) > 100:
                start_value = _seconds_from_milliseconds(start_value)
                end_value = _seconds_from_milliseconds(end_value)
            else:
                start_value = _seconds_from_any(start_value)
                end_value = _seconds_from_any(end_value)
            if start_value is not None and end_value is not None and end_value > start_value:
                ranges.append((start_value, end_value))

    for key in ("timestamp", "timestamps", "ctc_timestamps"):
        timestamps = item.get(key)
        if not isinstance(timestamps, list):
            continue
        for timestamp in timestamps:
            if isinstance(timestamp, dict):
                start = _seconds_from_any(_first_present(timestamp, "start", "start_time"))
                end = _seconds_from_any(_first_present(timestamp, "end", "end_time"))
            elif isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
                start = _seconds_from_milliseconds(timestamp[0])
                end = _seconds_from_milliseconds(timestamp[1])
            else:
                continue
            if start is not None and end is not None and end > start:
                ranges.append((start, end))

    ranges.sort(key=lambda value: (value[0], value[1]))
    return ranges


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
            _append_srt_entry_auto_time(
                entries,
                text,
                _first_present(sentence, "start", "start_time"),
                _first_present(sentence, "end", "end_time"),
            )

    timestamps = item.get("timestamp")
    words = item.get("words")
    if isinstance(timestamps, list):
        timestamp_entries_before = len(entries)
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
        if len(entries) == timestamp_entries_before:
            _append_timestamp_text_entries(entries, item.get("text", ""), timestamps)

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


def _build_srt(result):
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

    return ""


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
                text = "".join(
                    str(sentence.get("text", sentence.get("sentence", "")))
                    for sentence in item["sentence_info"]
                )
            if text is not None:
                texts.append(_clean_asr_text(text))
        elif item is not None:
            texts.append(_clean_asr_text(item))

    text = "\n".join(part for part in texts if part)
    srt = _build_srt(result)
    return text, srt


def _build_srt_with_text(timestamp_result, text):
    text = _clean_asr_text(text)
    if not text:
        return _build_srt(timestamp_result)

    items = timestamp_result if isinstance(timestamp_result, list) else [timestamp_result]
    entries = []
    if _append_aligned_srt_entries(entries, timestamp_result, text):
        entries.sort(key=lambda entry: (entry[0], entry[1]))
        blocks = []
        for index, (start, end, entry_text) in enumerate(entries, start=1):
            blocks.append(
                f"{index}\n"
                f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n"
                f"{entry_text}"
            )
        return "\n\n".join(blocks)

    time_ranges = []
    for item in items:
        time_ranges.extend(_extract_time_ranges_from_item(item))
    if _append_weighted_srt_entries(entries, text, time_ranges):
        entries.sort(key=lambda entry: (entry[0], entry[1]))
        blocks = []
        for index, (start, end, entry_text) in enumerate(entries, start=1):
            blocks.append(
                f"{index}\n"
                f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n"
                f"{entry_text}"
            )
        return "\n\n".join(blocks)

    patched_items = []
    used_text = False
    for item in items:
        if isinstance(item, dict):
            patched_item = dict(item)
            if not used_text:
                patched_item["text"] = text
                patched_item.pop("sentence_info", None)
                patched_item.pop("words", None)
                timestamps = patched_item.get("timestamp")
                if isinstance(timestamps, list):
                    patched_item["timestamp"] = [
                        [chunk[0], chunk[1]]
                        if isinstance(chunk, (list, tuple)) and len(chunk) >= 2
                        else chunk
                        for chunk in timestamps
                    ]
                used_text = True
            patched_items.append(patched_item)
        else:
            patched_items.append(item)
    return _build_srt(patched_items)


def _infer_audio(audio, model, device, batch_size_s, unload_model):
    audio_path = _save_audio_to_temp(audio)
    try:
        infer_device = _get_device(device)
        config = _model_config(model)
        recognizer = _get_model(model, infer_device)
        generate_kwargs = {
            "input": audio_path,
            "cache": {},
            "batch_size_s": int(batch_size_s),
            "merge_vad": True,
            "merge_length_s": 15,
        }
        itn_arg = config.get("itn_arg")
        if itn_arg:
            generate_kwargs[itn_arg] = True
        if model == "SenseVoiceSmall":
            generate_kwargs["language"] = "auto"
            generate_kwargs["ban_emo_unk"] = False
        if model == "Paraformer-Large":
            generate_kwargs["output_timestamp"] = True
            generate_kwargs["return_time_stamps"] = True

        result = recognizer.generate(**generate_kwargs)
        if unload_model:
            for cache_key in list(MODEL_CACHE):
                if cache_key[0] == model and cache_key[-1] == infer_device:
                    MODEL_CACHE.pop(cache_key, None)
            try:
                import gc
                import torch

                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
        return result
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


class FunASRTranscribeText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "device": (["auto", "cuda:0", "cpu"],),
                "batch_size_s": ("INT", {"default": 60, "min": 1, "max": 600, "step": 1}),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    CATEGORY = "FunASR"

    def transcribe(
        self,
        audio,
        device,
        batch_size_s,
        unload_model,
    ):
        result = _infer_audio(audio, "SenseVoiceSmall", device, batch_size_s, unload_model)
        text, _ = _normalize_result(result)
        return (text,)


class FunASRTranscribeSRT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "device": (["auto", "cuda:0", "cpu"],),
                "batch_size_s": ("INT", {"default": 300, "min": 1, "max": 600, "step": 1}),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("srt",)
    FUNCTION = "transcribe"
    CATEGORY = "FunASR"

    def transcribe(
        self,
        audio,
        device,
        batch_size_s,
        unload_model,
    ):
        text_result = _infer_audio(audio, "SenseVoiceSmall", device, batch_size_s, unload_model)
        text, _ = _normalize_result(text_result)
        timestamp_result = _infer_audio(audio, "Paraformer-Large", device, batch_size_s, unload_model)
        srt = _build_srt_with_text(timestamp_result, text)
        return (srt,)


NODE_CLASS_MAPPINGS = {
    "FunASRTranscribeText": FunASRTranscribeText,
    "FunASRTranscribeSRT": FunASRTranscribeSRT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FunASRTranscribeText": "FunASR Text",
    "FunASRTranscribeSRT": "FunASR SRT",
}
