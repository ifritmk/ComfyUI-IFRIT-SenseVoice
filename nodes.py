import difflib
import os
import re
import unicodedata
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
    if model_choice in {"Paraformer-Large", "SenseVoiceSmall"}:
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


def _release_model(model_choice=None, device=None):
    for cache_key in list(MODEL_CACHE):
        if model_choice is not None and cache_key[0] != model_choice:
            continue
        if device is not None and cache_key[-1] != device:
            continue
        MODEL_CACHE.pop(cache_key, None)
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


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


def _timestamp_values_are_milliseconds(values):
    clean_values = []
    for value in values:
        try:
            clean_values.append(abs(float(value)))
        except (TypeError, ValueError):
            pass
    if not clean_values:
        return False
    return max(clean_values) > 1000


def _timestamp_pair_seconds(start, end, milliseconds=False):
    if milliseconds:
        return _seconds_from_milliseconds(start), _seconds_from_milliseconds(end)
    return _seconds_from_any(start), _seconds_from_any(end)


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
    split_marks = {",", "\uff0c", "\u3001", ".", "!", "?", "\u3002", "\uff01", "\uff1f", "\uff1b", ";"}
    for char in text:
        current += char
        if char in split_marks or len(current) >= max_chars:
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


def _is_alignment_char(char):
    if not char or char.isspace():
        return False
    return not unicodedata.category(char).startswith(("P", "S"))


def _alignment_chars_with_positions(text):
    text_chars = _chars_no_space(text)
    alignment_chars = []
    positions = []
    for index, char in enumerate(text_chars):
        if _is_alignment_char(char):
            alignment_chars.append(char)
            positions.append(index)
    return alignment_chars, positions, text_chars


def _alignment_len(text):
    alignment_chars, _, _ = _alignment_chars_with_positions(text)
    return len(alignment_chars)


def _alignment_index_ranges(parts):
    ranges = []
    cursor = 0
    for part in parts:
        length = _alignment_len(part)
        if length <= 0:
            ranges.append((cursor, cursor))
            continue
        ranges.append((cursor, cursor + length - 1))
        cursor += length
    return ranges


def _build_alignment_map(source_chars, target_chars):
    matcher = difflib.SequenceMatcher(None, source_chars, target_chars, autojunk=False)
    mapping = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                mapping[i1 + offset] = j1 + offset
            continue
        if tag == "replace":
            source_len = i2 - i1
            target_len = j2 - j1
            if source_len <= 0 or target_len <= 0:
                continue
            for offset in range(source_len):
                if source_len == 1:
                    target_offset = 0
                else:
                    target_offset = round(offset * (target_len - 1) / (source_len - 1))
                mapping[i1 + offset] = j1 + target_offset
    return mapping


def _nearest_mapped_index(mapping, source_index, source_len, direction):
    if source_index in mapping:
        return mapping[source_index]
    if direction < 0:
        search_range = range(source_index - 1, -1, -1)
    else:
        search_range = range(source_index + 1, source_len)
    for candidate in search_range:
        if candidate in mapping:
            return mapping[candidate]
    return None


def _clean_srt_text(text):
    chars = []
    previous_space = False
    for char in _clean_asr_text(text):
        if char.isspace():
            if chars and not previous_space:
                chars.append(" ")
                previous_space = True
            continue
        if _is_alignment_char(char):
            chars.append(char)
            previous_space = False
    return "".join(chars).strip()


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
        parts.append(_clean_srt_text("".join(chars[cursor:end])))
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
            _append_srt_entry(entries, _clean_srt_text(part), chosen[0][0], chosen[-1][1])
            continue

        timeline_start = clean_ranges[range_index][0]
        timeline_end = clean_ranges[-1][1]
        remaining_text_chars = max(1, sum(len(value) for value in parts[index:]))
        cursor = timeline_start
        for tail_index, tail_part in enumerate(parts[index:], start=index):
            duration = (timeline_end - timeline_start) * len(tail_part) / remaining_text_chars
            end = timeline_end if tail_index == len(parts) - 1 else cursor + max(0.6, duration)
            _append_srt_entry(entries, _clean_srt_text(tail_part), cursor, min(end, timeline_end))
            cursor = min(end, timeline_end)
        return True

    return True


def _split_text_by_reference_parts(reference_text, target_text, reference_parts):
    target_alignment_chars, target_positions, target_chars = _alignment_chars_with_positions(target_text)
    if not target_chars:
        return []

    if not reference_parts:
        return []

    reference_alignment_chars, _, _ = _alignment_chars_with_positions(reference_text)
    if not reference_alignment_chars or not target_alignment_chars:
        weights = [_alignment_len(part) for part in reference_parts]
        return _split_text_by_weights(target_text, weights)

    matcher = difflib.SequenceMatcher(None, reference_alignment_chars, target_alignment_chars, autojunk=False)
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
        ref_len = _alignment_len(reference_part)
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
                target_alignment_end = max(candidates)
                target_end = target_positions[target_alignment_end] + 1
            else:
                target_alignment_end = round(len(target_alignment_chars) * ref_end / max(1, len(reference_alignment_chars)))
                target_alignment_end = max(0, min(target_alignment_end, len(target_positions) - 1))
                target_end = target_positions[target_alignment_end] + 1
            target_end = max(target_cursor + 1, min(target_end, len(target_chars) - (len(reference_parts) - index - 1)))
        parts.append(_clean_srt_text("".join(target_chars[target_cursor:target_end])))
        target_cursor = target_end
        ref_cursor = ref_end

    return [part for part in parts if part]


def _split_text_by_reference(reference_text, target_text, max_chars=28):
    reference_parts = _split_text_for_srt(reference_text, max_chars=max_chars)
    if not reference_parts:
        return _split_text_for_srt(target_text, max_chars=max_chars)
    return _split_text_by_reference_parts(reference_text, target_text, reference_parts)


def _append_timestamp_text_entries(entries, text, timestamps):
    text = _clean_asr_text(text)
    if not text:
        return False

    clean_timestamps = []
    timestamp_values = []
    for chunk in timestamps:
        if isinstance(chunk, (list, tuple)) and len(chunk) >= 2:
            timestamp_values.extend([chunk[0], chunk[1]])
    timestamps_are_ms = _timestamp_values_are_milliseconds(timestamp_values)
    for chunk in timestamps:
        if isinstance(chunk, (list, tuple)) and len(chunk) >= 2:
            start, end = _timestamp_pair_seconds(chunk[0], chunk[1], timestamps_are_ms)
            if start is not None and end is not None:
                clean_timestamps.append((start, end))
    if not clean_timestamps:
        return False

    tokens = text.split()
    if not tokens:
        tokens = [char for char in text if not char.isspace()]
    if not tokens:
        return False
    if len(clean_timestamps) == 1:
        start, end = clean_timestamps[0]
        _append_srt_entry(entries, text, start, end)
        return True

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

    if segments:
        segments.sort(key=lambda value: (value[0], value[1]))
        return segments

    timestamps = item.get("timestamp")
    words = item.get("words")
    item_text = _clean_asr_text(item.get("text", ""))
    if isinstance(timestamps, list):
        timestamp_values = [
            value
            for timestamp in timestamps
            if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2
            for value in (timestamp[0], timestamp[1])
        ]
        timestamps_are_ms = _timestamp_values_are_milliseconds(timestamp_values)
        timestamp_ranges = []
        timestamp_texts = []
        for index, chunk in enumerate(timestamps):
            if not isinstance(chunk, (list, tuple)) or len(chunk) < 2:
                continue
            start, end = _timestamp_pair_seconds(chunk[0], chunk[1], timestamps_are_ms)
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
            for (start, end), token in zip(timestamp_ranges, timestamp_texts):
                if token:
                    segments.append((start, end, token))
        elif item_text and timestamp_ranges:
            item_chars = _alignment_chars_with_positions(item_text)[0]
            usable = min(len(item_chars), len(timestamp_ranges))
            for index in range(usable):
                start, end = timestamp_ranges[index]
                segments.append((start, end, item_chars[index]))

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

    raw_sense_parts = _split_text_for_srt(text)
    sense_parts = [_clean_srt_text(part) for part in raw_sense_parts]
    sense_parts = [part for part in sense_parts if part]
    if len(sense_parts) < 2:
        return False

    sense_chars = []
    for part in raw_sense_parts:
        sense_chars.extend(_alignment_chars_with_positions(part)[0])
    para_chars = []
    para_token_indexes = []
    for index, (_, _, segment_text) in enumerate(timed_segments):
        chars = _alignment_chars_with_positions(segment_text)[0]
        para_chars.extend(chars)
        para_token_indexes.extend([index] * len(chars))

    if not sense_chars or not para_chars or not para_token_indexes:
        return False

    mapping = _build_alignment_map(sense_chars, para_chars)
    if not mapping:
        return False

    last_token_end = -1
    previous_para_end_char = -1
    sense_ranges = _alignment_index_ranges(raw_sense_parts)
    total_sense_chars = max(1, len(sense_chars))
    total_para_chars = max(1, len(para_chars))
    for sense_part, (sense_start, sense_end) in zip(sense_parts, sense_ranges):
        if sense_end < sense_start:
            continue
        para_start_char = _nearest_mapped_index(mapping, sense_start, len(sense_chars), 1)
        para_end_char = _nearest_mapped_index(mapping, sense_end, len(sense_chars), -1)
        if para_start_char is None:
            para_start_char = round(sense_start * (total_para_chars - 1) / max(1, total_sense_chars - 1))
        if para_end_char is None:
            para_end_char = round(sense_end * (total_para_chars - 1) / max(1, total_sense_chars - 1))
        if para_end_char < para_start_char:
            para_start_char, para_end_char = para_end_char, para_start_char
        para_start_char = max(para_start_char, previous_para_end_char + 1)
        para_end_char = max(para_end_char, para_start_char)
        para_start_char = max(0, min(para_start_char, len(para_token_indexes) - 1))
        para_end_char = max(0, min(para_end_char, len(para_token_indexes) - 1))
        token_start = para_token_indexes[para_start_char]
        token_end = para_token_indexes[para_end_char]
        if token_end < token_start:
            token_start, token_end = token_end, token_start
        token_start = max(token_start, last_token_end + 1)
        token_end = max(token_end, token_start)
        if token_start >= len(timed_segments):
            break
        token_end = min(token_end, len(timed_segments) - 1)
        start = timed_segments[token_start][0]
        end = timed_segments[token_end][1]
        _append_srt_entry(entries, sense_part, start, end)
        last_token_end = token_end
        previous_para_end_char = para_end_char
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
        timestamp_values = []
        for timestamp in timestamps:
            if isinstance(timestamp, dict):
                timestamp_values.extend([
                    _first_present(timestamp, "start", "start_time"),
                    _first_present(timestamp, "end", "end_time"),
                ])
            elif isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
                timestamp_values.extend([timestamp[0], timestamp[1]])
        timestamps_are_ms = _timestamp_values_are_milliseconds(timestamp_values)
        for timestamp in timestamps:
            if isinstance(timestamp, dict):
                start_raw = _first_present(timestamp, "start", "start_time")
                end_raw = _first_present(timestamp, "end", "end_time")
                start, end = _timestamp_pair_seconds(start_raw, end_raw, timestamps_are_ms)
            elif isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
                start, end = _timestamp_pair_seconds(timestamp[0], timestamp[1], timestamps_are_ms)
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
        sentence_entries_before = len(entries)
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
        if len(entries) > sentence_entries_before:
            return

    timestamps = item.get("timestamp")
    words = item.get("words")
    if isinstance(timestamps, list):
        timestamp_values = [
            value
            for timestamp in timestamps
            if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2
            for value in (timestamp[0], timestamp[1])
        ]
        timestamps_are_ms = _timestamp_values_are_milliseconds(timestamp_values)
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
            _append_srt_entry(entries, text, chunk[0], chunk[1], milliseconds=timestamps_are_ms)
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
        text = _clean_srt_text(text)
        if not text:
            continue
        blocks.append(
            f"{len(blocks) + 1}\n"
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
            _release_model(model, infer_device)
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
                "batch_size_s": ("INT", {"default": 30, "min": 1, "max": 600, "step": 1}),
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
                "unload_model": ("BOOLEAN", {"default": True}),
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
        infer_device = _get_device(device)
        text_result = _infer_audio(audio, "SenseVoiceSmall", device, batch_size_s, True)
        text, _ = _normalize_result(text_result)
        _release_model("SenseVoiceSmall", infer_device)
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
