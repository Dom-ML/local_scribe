import os
from datetime import datetime
from pathlib import Path

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


def load_settings() -> dict:
    """Load settings from YAML file."""
    summarization_defaults = {
        "enabled": False,
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
    defaults = {"output": "recording.wav", "language": "English", "sample_rate": 16000, "level_bar_width": 30, "transcribe_only": None, "summarization": summarization_defaults}
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            user_settings = yaml.safe_load(f) or {}
            merged = {**defaults, **user_settings}
            if "summarization" in user_settings:
                merged["summarization"] = {**summarization_defaults, **user_settings["summarization"]}
                if "prompts" in user_settings["summarization"]:
                    merged["summarization"]["prompts"] = {**summarization_defaults["prompts"], **user_settings["summarization"]["prompts"]}
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


def save_audio(audio: np.ndarray, output_path: str, sample_rate: int) -> None:
    """Save audio data to a WAV file."""
    sf.write(output_path, audio, sample_rate)


def format_transcript_for_llm(text: str, timestamp: datetime, duration_seconds: float | None = None) -> str:
    """Format transcript for LFM2 model input."""
    duration_str = f"{int(duration_seconds // 60)}:{int(duration_seconds % 60):02d}" if duration_seconds else "Unknown"
    header = f"""Title: Audio Transcript
Date: {timestamp.strftime("%Y-%m-%d")}
Time: {timestamp.strftime("%H:%M:%S")}
Duration: {duration_str}
Participants: Speaker 1

----------

**Speaker 1**: {text}"""
    return header


def transcribe(audio_path: str, language: str) -> str:
    """Transcribe audio file using Qwen ASR."""
    print("Transcribing...")
    model = load(MODEL_NAME)
    result = model.generate(audio_path, language=language)
    return result.text


def summarize(text: str, settings: dict, timestamp: datetime, duration_seconds: float | None = None) -> str | None:
    """Summarize transcript using LFM2."""
    from mlx_lm import generate, load as load_lm
    from mlx_lm.sample_utils import make_sampler

    sum_settings = settings["summarization"]
    print("Summarizing...")

    formatted_transcript = format_transcript_for_llm(text, timestamp, duration_seconds)
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


def save_transcript(text: str, device_name: str, language: str, summary: str | None = None, summary_type: str | None = None, summary_model: str | None = None) -> Path:
    """Save transcript as markdown with YAML frontmatter, organized by day."""
    now = datetime.now()
    day_dir = TRANSCRIPTS_DIR / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    filename = now.strftime("%Y-%m-%d_%H-%M-%S.md")
    filepath = day_dir / filename
    metadata = {"timestamp": now.isoformat(), "device": device_name, "model": MODEL_NAME, "language": language}
    if summary_model:
        metadata["summary_model"] = summary_model
    if summary_type:
        metadata["summary_type"] = summary_type
    content = f"---\n{yaml.dump(metadata, default_flow_style=False)}---\n\n{text}\n"
    if summary:
        content += f"\n---\n\n## Summary ({summary_type})\n\n{summary}\n"
    filepath.write_text(content)
    return filepath


def main() -> None:
    """Main entry point."""
    settings = load_settings()
    sum_settings = settings["summarization"]
    now = datetime.now()

    if settings.get("transcribe_only"):
        text = transcribe(settings["transcribe_only"], settings["language"])
        summary = summarize(text, settings, now) if sum_settings["enabled"] else None
        filepath = save_transcript(text, "file", settings["language"], summary=summary, summary_type=sum_settings["summary_type"] if summary else None, summary_model=sum_settings["model"] if summary else None)
        print(f"\n{text}")
        if summary:
            print(f"\n## Summary ({sum_settings['summary_type']})\n\n{summary}")
        print(f"\nSaved to: {filepath}")
        return

    device_index, device_name = select_input_device()
    audio = record_audio(device_index, settings["sample_rate"], settings["level_bar_width"])
    save_audio(audio, settings["output"], settings["sample_rate"])
    text = transcribe(settings["output"], settings["language"])
    duration_seconds = len(audio) / settings["sample_rate"]
    summary = summarize(text, settings, now, duration_seconds) if sum_settings["enabled"] else None
    filepath = save_transcript(text, device_name, settings["language"], summary=summary, summary_type=sum_settings["summary_type"] if summary else None, summary_model=sum_settings["model"] if summary else None)
    print(f"\n{text}")
    if summary:
        print(f"\n## Summary ({sum_settings['summary_type']})\n\n{summary}")
    print(f"\nSaved to: {filepath}")


if __name__ == "__main__":
    main()
