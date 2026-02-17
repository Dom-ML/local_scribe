import os
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Set local model cache before importing HF-dependent libraries
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
os.environ["HF_HUB_CACHE"] = str(MODELS_DIR)

import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml
from mlx_audio.stt import load

SETTINGS_FILE = Path(__file__).parent / "settings.yaml"
TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"
MODEL_NAME = "mlx-community/Qwen3-ASR-0.6B-8bit"
SUMMARIZATION_MODEL = "mlx-community/LFM2-2.6B-Transcript-4bit"
DIARIZATION_MODEL = "mlx-community/diar_sortformer_4spk-v1-fp32"
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


def load_settings() -> dict:
    """Load settings from YAML file."""
    summarization_defaults = {
        "enabled": True,
        "model": SUMMARIZATION_MODEL,
        "temperature": 0.3,
        "max_tokens": 512,
        "summary_type": "executive",
        "system_prompt": "You are an expert meeting analyst. Analyze the transcript carefully and provide clear, accurate information based on the content.",
        "prompts": {
            "executive": "Provide a brief executive summary (2-3 sentences) of the key outcomes and decisions.",
            "detailed": "Provide a detailed summary covering all major topics, discussions, and outcomes.",
            "action_items": "List the specific action items assigned during this meeting.",
            "key_decisions": "List the key decisions that were made during this meeting.",
            "participants": "List the participants mentioned in this transcript.",
            "topics": "List the main topics and subjects that were discussed.",
        },
    }
    diarization_defaults = {"enabled": True, "backend": "sortformer", "model": DIARIZATION_MODEL, "threshold": 0.5, "min_duration": 0.0, "merge_gap": 0.0, "pyannote_model": PYANNOTE_MODEL, "min_speakers": None, "max_speakers": None}
    system_audio_defaults = {"enabled": False, "device": None, "output": "recording_system.wav", "gain": 1.0}
    defaults = {"output": "recording.wav", "language": "English", "sample_rate": 16000, "level_bar_width": 30, "transcribe_only": None, "summarization": summarization_defaults, "diarization": diarization_defaults, "system_audio": system_audio_defaults}
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            user_settings = yaml.safe_load(f) or {}
            merged = {**defaults, **user_settings}
            if "summarization" in user_settings:
                merged["summarization"] = {**summarization_defaults, **user_settings["summarization"]}
                if "prompts" in user_settings["summarization"]:
                    merged["summarization"]["prompts"] = {**summarization_defaults["prompts"], **user_settings["summarization"]["prompts"]}
            if "diarization" in user_settings:
                merged["diarization"] = {**diarization_defaults, **user_settings["diarization"]}
            if "system_audio" in user_settings:
                merged["system_audio"] = {**system_audio_defaults, **user_settings["system_audio"]}
            return merged
    return defaults


def select_input_device() -> tuple[int, str]:
    """Display available input devices and let user select one. Returns (index, name)."""
    devices = [(i, dev["name"]) for i, dev in enumerate(sd.query_devices()) if dev["max_input_channels"] > 0]
    print("Select audio input device:")
    for i, (idx, name) in enumerate(devices):
        print(f"  [{i + 1}] {name}")
    while True:
        try:
            choice = int(input("\nEnter number: ")) - 1
            if 0 <= choice < len(devices):
                return devices[choice]
            print("Invalid selection, try again.")
        except ValueError:
            print("Enter a number.")


def select_system_device(settings: dict) -> tuple[int, str] | None:
    """Auto-detect or match a system audio input device (e.g. BlackHole). Returns (index, name) or None."""
    sys_settings = settings["system_audio"]
    search_term = sys_settings["device"].lower() if sys_settings["device"] else "blackhole"
    matches = [(i, dev["name"]) for i, dev in enumerate(sd.query_devices()) if dev["max_input_channels"] > 0 and search_term in dev["name"].lower()]
    if not matches:
        print(f"Warning: No system audio device matching '{search_term}' found. Falling back to mic-only.")
        return None
    if len(matches) == 1:
        print(f"System audio device: {matches[0][1]}")
        return matches[0]
    print("Multiple system audio devices found:")
    for i, (idx, name) in enumerate(matches):
        print(f"  [{i + 1}] {name}")
    while True:
        try:
            choice = int(input("\nSelect system audio device: ")) - 1
            if 0 <= choice < len(matches):
                return matches[choice]
            print("Invalid selection, try again.")
        except ValueError:
            print("Enter a number.")


def record_audio(device: int, sample_rate: int, level_bar_width: int) -> np.ndarray:
    """Record audio from the specified device until Enter is pressed."""
    print("\nRecording... (press Enter to stop)")
    audio_chunks: list[np.ndarray] = []

    def callback(indata: np.ndarray, frames: int, time, status) -> None:
        audio_chunks.append(indata.copy())
        rms = np.sqrt(np.mean(indata**2))
        level = min(int(rms * 200), level_bar_width)
        bar = "\u2588" * level + "\u2591" * (level_bar_width - level)
        print(f"\r  [{bar}] ", end="", flush=True)

    with sd.InputStream(samplerate=sample_rate, channels=1, device=device, dtype="float32", callback=callback):
        input()

    print()
    return np.concatenate(audio_chunks)


def record_dual_audio(mic_device: int, sys_device: int, sample_rate: int, level_bar_width: int, gain: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Record from mic and system audio devices simultaneously until Enter is pressed."""
    print("\nRecording from mic + system audio... (press Enter to stop)")
    mic_chunks: list[np.ndarray] = []
    sys_chunks: list[np.ndarray] = []
    sys_sample_rate = int(sd.query_devices(sys_device)["default_samplerate"])

    def mic_callback(indata: np.ndarray, frames: int, time, status) -> None:
        mic_chunks.append(indata.copy())
        rms = np.sqrt(np.mean(indata**2))
        level = min(int(rms * 200), level_bar_width)
        bar = "\u2588" * level + "\u2591" * (level_bar_width - level)
        print(f"\r  [{bar}] ", end="", flush=True)

    def sys_callback(indata: np.ndarray, frames: int, time, status) -> None:
        sys_chunks.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, device=mic_device, dtype="float32", callback=mic_callback), sd.InputStream(samplerate=sys_sample_rate, channels=1, device=sys_device, dtype="float32", callback=sys_callback):
        input()

    print()
    mic_audio = np.concatenate(mic_chunks)
    sys_audio = np.concatenate(sys_chunks)

    if sys_sample_rate != sample_rate:
        import librosa
        print(f"Resampling system audio from {sys_sample_rate}Hz to {sample_rate}Hz...")
        sys_audio = librosa.resample(sys_audio.flatten(), orig_sr=sys_sample_rate, target_sr=sample_rate)
        sys_audio = sys_audio.reshape(-1, 1)

    if gain != 1.0:
        sys_audio = sys_audio * gain
        sys_audio = np.clip(sys_audio, -1.0, 1.0)

    return mic_audio, sys_audio


def save_audio(audio: np.ndarray, output_path: str, sample_rate: int) -> None:
    """Save audio data to a WAV file."""
    sf.write(output_path, audio, sample_rate)


def diarize(audio_path: str, settings: dict) -> list[dict]:
    """Run speaker diarization on audio file. Returns list of segments with start, end, speaker."""
    backend = settings["diarization"].get("backend", "sortformer")
    if backend == "pyannote":
        return _diarize_pyannote(audio_path, settings)
    return _diarize_sortformer(audio_path, settings)


def _diarize_sortformer(audio_path: str, settings: dict) -> list[dict]:
    """Run diarization using Sortformer (MLX-native). Supports up to 4 speakers."""
    import mlx.core as mx
    from mlx_audio.vad import load as load_vad

    diar_settings = settings["diarization"]
    print("Running speaker diarization (Sortformer)...")
    model = load_vad(diar_settings["model"])
    result = model.generate(audio_path, threshold=diar_settings["threshold"], min_duration=diar_settings["min_duration"], merge_gap=diar_settings["merge_gap"])
    segments = [{"start": seg.start, "end": seg.end, "speaker": f"SPEAKER_{seg.speaker}"} for seg in result.segments]

    del model, result
    mx.clear_cache()
    return segments


def _diarize_pyannote(audio_path: str, settings: dict) -> list[dict]:
    """Run diarization using pyannote (PyTorch-based). Requires HF_TOKEN and pyannote.audio installed."""
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        raise ImportError("pyannote backend requires pyannote.audio. Install with: uv sync --extra pyannote")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable required for pyannote diarization. Create a .env file with your token.")

    diar_settings = settings["diarization"]
    print("Running speaker diarization (pyannote)...")
    pipeline = Pipeline.from_pretrained(diar_settings["pyannote_model"], token=hf_token)

    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))

    # Load audio as in-memory waveform (avoids torchcodec/FFmpeg dependency)
    audio_data, sample_rate = sf.read(audio_path)
    if audio_data.ndim == 1:
        audio_data = audio_data[np.newaxis, :]  # Add channel dimension
    else:
        audio_data = audio_data.T  # soundfile returns (samples, channels), pyannote expects (channels, samples)
    waveform = torch.from_numpy(audio_data).float()
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    diarization_args = {}
    if diar_settings.get("min_speakers"):
        diarization_args["min_speakers"] = diar_settings["min_speakers"]
    if diar_settings.get("max_speakers"):
        diarization_args["max_speakers"] = diar_settings["max_speakers"]

    result = pipeline(audio_input, **diarization_args)

    segments = []
    for turn, _, speaker in result.speaker_diarization.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    # Release PyTorch diarization model before MLX ASR loads
    del pipeline, result, waveform, audio_input
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return segments


def transcribe_segments(audio_path: str, segments: list[dict], language: str) -> list[dict]:
    """Transcribe each diarization segment. Adds 'text' field to each segment."""
    print(f"Transcribing {len(segments)} segments...")
    audio_data, sample_rate = sf.read(audio_path)
    model = load(MODEL_NAME)

    for i, segment in enumerate(segments):
        start_sample = int(segment["start"] * sample_rate)
        end_sample = int(segment["end"] * sample_rate)
        segment_audio = audio_data[start_sample:end_sample]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, segment_audio, sample_rate)
            result = model.generate(tmp.name, language=language)
            segment["text"] = result.text.strip()
            os.unlink(tmp.name)

        print(f"  Segment {i + 1}/{len(segments)} done")

    return segments


def format_transcript_for_llm(text: str, timestamp: datetime, duration_seconds: float | None = None, segments: list[dict] | None = None) -> str:
    """Format transcript for LFM2 model input."""
    duration_str = f"{int(duration_seconds // 60)}:{int(duration_seconds % 60):02d}" if duration_seconds else "Unknown"

    if segments:
        speaker_map = {}
        speaker_num = 1
        for seg in segments:
            if seg["speaker"] not in speaker_map:
                speaker_map[seg["speaker"]] = f"Speaker {speaker_num}"
                speaker_num += 1

        participants = ", ".join(speaker_map.values())
        body_lines = []
        for seg in segments:
            speaker_label = speaker_map[seg["speaker"]]
            if "source" in seg:
                speaker_label += f" [{seg['source']}]"
            body_lines.append(f"**{speaker_label}**: {seg['text']}")
        body = "\n\n".join(body_lines)
    else:
        participants = "Speaker 1"
        body = f"**Speaker 1**: {text}"

    header = f"""Title: Audio Transcript
Date: {timestamp.strftime("%Y-%m-%d")}
Time: {timestamp.strftime("%H:%M:%S")}
Duration: {duration_str}
Participants: {participants}

----------

{body}"""
    return header


def transcribe(audio_path: str, language: str) -> str:
    """Transcribe audio file using Qwen ASR."""
    print("Transcribing...")
    model = load(MODEL_NAME)
    result = model.generate(audio_path, language=language)
    return result.text


def merge_transcripts(mic_segments: list[dict] | None, mic_text: str, sys_segments: list[dict] | None, sys_text: str) -> tuple[list[dict] | None, str]:
    """Merge mic and system audio transcripts. Tags segments with source and sorts chronologically."""
    if mic_segments and sys_segments:
        for seg in mic_segments:
            seg["source"] = "mic"
        for seg in sys_segments:
            seg["source"] = "system"
        merged = sorted(mic_segments + sys_segments, key=lambda s: s["start"])
        text = " ".join(seg["text"] for seg in merged)
        return merged, text
    if mic_segments and not sys_segments:
        for seg in mic_segments:
            seg["source"] = "mic"
        sys_seg = {"start": 0.0, "end": 0.0, "speaker": "SYSTEM", "text": sys_text, "source": "system"}
        merged = sorted(mic_segments + [sys_seg], key=lambda s: s["start"])
        text = " ".join(seg["text"] for seg in merged)
        return merged, text
    if sys_segments and not mic_segments:
        for seg in sys_segments:
            seg["source"] = "system"
        mic_seg = {"start": 0.0, "end": 0.0, "speaker": "MIC", "text": mic_text, "source": "mic"}
        merged = sorted([mic_seg] + sys_segments, key=lambda s: s["start"])
        text = " ".join(seg["text"] for seg in merged)
        return merged, text
    # Neither has segments
    combined = f"[mic] {mic_text}\n\n[system] {sys_text}"
    return None, combined


def summarize(text: str, settings: dict, timestamp: datetime, duration_seconds: float | None = None, segments: list[dict] | None = None) -> str | None:
    """Summarize transcript using LFM2."""
    import mlx.core as mx
    from mlx_lm import generate, load as load_lm
    from mlx_lm.sample_utils import make_sampler

    sum_settings = settings["summarization"]
    print("Summarizing...")

    # Free ASR model memory before loading summarization model
    mx.clear_cache()

    formatted_transcript = format_transcript_for_llm(text, timestamp, duration_seconds, segments)
    summary_type = sum_settings["summary_type"]
    user_prompt = sum_settings["prompts"].get(summary_type, sum_settings["prompts"]["executive"])

    messages = [
        {"role": "system", "content": sum_settings["system_prompt"]},
        {"role": "user", "content": f"{formatted_transcript}\n\n{user_prompt}"},
    ]

    model, tokenizer = load_lm(sum_settings["model"])
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    sampler = make_sampler(temp=sum_settings["temperature"])
    response = generate(model, tokenizer, prompt=prompt, max_tokens=sum_settings["max_tokens"], sampler=sampler, verbose=False)
    return response


def save_transcript(text: str, device_name: str, language: str, summary: str | None = None, summary_type: str | None = None, summary_model: str | None = None, segments: list[dict] | None = None, diarization_model: str | None = None, system_device_name: str | None = None) -> Path:
    """Save transcript as markdown with YAML frontmatter, organized by day."""
    now = datetime.now()
    day_dir = TRANSCRIPTS_DIR / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    filename = now.strftime("%Y-%m-%d_%H-%M-%S.md")
    filepath = day_dir / filename
    metadata = {"timestamp": now.isoformat(), "device": device_name, "model": MODEL_NAME, "language": language}
    if system_device_name:
        metadata["system_device"] = system_device_name
    if diarization_model:
        metadata["diarization_model"] = diarization_model
        unique_speakers = len(set(seg["speaker"] for seg in segments)) if segments else 0
        metadata["speaker_count"] = unique_speakers
    if summary_model:
        metadata["summary_model"] = summary_model
    if summary_type:
        metadata["summary_type"] = summary_type

    if segments:
        speaker_map = {}
        speaker_num = 1
        for seg in segments:
            if seg["speaker"] not in speaker_map:
                speaker_map[seg["speaker"]] = f"Speaker {speaker_num}"
                speaker_num += 1
        body_lines = []
        for seg in segments:
            speaker_label = speaker_map[seg["speaker"]]
            if "source" in seg:
                speaker_label += f" [{seg['source']}]"
            body_lines.append(f"**{speaker_label}**: {seg['text']}")
        body = "\n\n".join(body_lines)
    else:
        body = text

    content = f"---\n{yaml.dump(metadata, default_flow_style=False)}---\n\n{body}\n"
    if summary:
        content += f"\n---\n\n## Summary ({summary_type})\n\n{summary}\n"
    filepath.write_text(content)
    return filepath


def print_transcript(segments: list[dict] | None, text: str) -> None:
    """Print transcript to console with speaker labels if available."""
    if segments:
        speaker_map = {}
        speaker_num = 1
        for seg in segments:
            if seg["speaker"] not in speaker_map:
                speaker_map[seg["speaker"]] = f"Speaker {speaker_num}"
                speaker_num += 1
        print()
        for seg in segments:
            speaker_label = speaker_map[seg["speaker"]]
            if "source" in seg:
                speaker_label += f" [{seg['source']}]"
            print(f"**{speaker_label}**: {seg['text']}\n")
    else:
        print(f"\n{text}")


def _process_audio(audio_path: str, settings: dict) -> tuple[list[dict] | None, str]:
    """Run diarization (if enabled) and transcription on a single audio file. Returns (segments, text)."""
    diar_settings = settings["diarization"]
    if diar_settings["enabled"]:
        segments = diarize(audio_path, settings)
        segments = transcribe_segments(audio_path, segments, settings["language"])
        text = " ".join(seg["text"] for seg in segments)
        return segments, text
    text = transcribe(audio_path, settings["language"])
    return None, text


def _active_diarization_model(settings: dict) -> str:
    """Return the model name for the active diarization backend."""
    diar_settings = settings["diarization"]
    if diar_settings.get("backend") == "pyannote":
        return diar_settings["pyannote_model"]
    return diar_settings["model"]


def main() -> None:
    """Main entry point."""
    settings = load_settings()
    sum_settings = settings["summarization"]
    diar_settings = settings["diarization"]
    sys_settings = settings["system_audio"]
    now = datetime.now()

    if settings.get("transcribe_only"):
        transcribe_only = settings["transcribe_only"]
        # Support dict form: {mic: file1.wav, system: file2.wav}
        if isinstance(transcribe_only, dict):
            mic_path = transcribe_only["mic"]
            sys_path = transcribe_only["system"]
            print(f"Processing mic audio: {mic_path}")
            mic_segments, mic_text = _process_audio(mic_path, settings)
            print(f"Processing system audio: {sys_path}")
            sys_segments, sys_text = _process_audio(sys_path, settings)
            segments, text = merge_transcripts(mic_segments, mic_text, sys_segments, sys_text)
            system_device_name = "file"
        else:
            mic_segments, text = _process_audio(transcribe_only, settings)
            segments = mic_segments
            system_device_name = None
        summary = summarize(text, settings, now, segments=segments) if sum_settings["enabled"] else None
        filepath = save_transcript(text, "file", settings["language"], summary=summary, summary_type=sum_settings["summary_type"] if summary else None, summary_model=sum_settings["model"] if summary else None, segments=segments, diarization_model=_active_diarization_model(settings) if segments else None, system_device_name=system_device_name)
        print_transcript(segments, text)
        if summary:
            print(f"\n## Summary ({sum_settings['summary_type']})\n\n{summary}")
        print(f"\nSaved to: {filepath}")
        return

    device_index, device_name = select_input_device()
    sys_device = None
    sys_device_name = None
    if sys_settings["enabled"]:
        result = select_system_device(settings)
        if result:
            sys_device, sys_device_name = result

    if sys_device is not None:
        mic_audio, sys_audio = record_dual_audio(device_index, sys_device, settings["sample_rate"], settings["level_bar_width"], gain=sys_settings["gain"])
        save_audio(mic_audio, settings["output"], settings["sample_rate"])
        save_audio(sys_audio, sys_settings["output"], settings["sample_rate"])
        duration_seconds = len(mic_audio) / settings["sample_rate"]
        print(f"Processing mic audio: {settings['output']}")
        mic_segments, mic_text = _process_audio(settings["output"], settings)
        print(f"Processing system audio: {sys_settings['output']}")
        sys_segments, sys_text = _process_audio(sys_settings["output"], settings)
        segments, text = merge_transcripts(mic_segments, mic_text, sys_segments, sys_text)
    else:
        audio = record_audio(device_index, settings["sample_rate"], settings["level_bar_width"])
        save_audio(audio, settings["output"], settings["sample_rate"])
        duration_seconds = len(audio) / settings["sample_rate"]
        segments, text = _process_audio(settings["output"], settings)

    summary = summarize(text, settings, now, duration_seconds, segments=segments) if sum_settings["enabled"] else None
    filepath = save_transcript(text, device_name, settings["language"], summary=summary, summary_type=sum_settings["summary_type"] if summary else None, summary_model=sum_settings["model"] if summary else None, segments=segments, diarization_model=_active_diarization_model(settings) if segments else None, system_device_name=sys_device_name)
    print_transcript(segments, text)
    if summary:
        print(f"\n## Summary ({sum_settings['summary_type']})\n\n{summary}")
    print(f"\nSaved to: {filepath}")


if __name__ == "__main__":
    main()
