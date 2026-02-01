from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml
from mlx_audio.stt import load

SETTINGS_FILE = Path(__file__).parent / "settings.yaml"
TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"
MODEL_NAME = "mlx-community/Qwen3-ASR-0.6B-8bit"


def load_settings() -> dict:
    """Load settings from YAML file."""
    defaults = {"output": "recording.wav", "language": "English", "sample_rate": 16000, "level_bar_width": 30, "transcribe_only": None}
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return {**defaults, **yaml.safe_load(f)}
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


def transcribe(audio_path: str, language: str) -> str:
    """Transcribe audio file using Qwen ASR."""
    print("Transcribing...")
    model = load(MODEL_NAME)
    result = model.generate(audio_path, language=language)
    return result.text


def save_transcript(text: str, device_name: str, language: str) -> Path:
    """Save transcript as markdown with YAML frontmatter, organized by day."""
    now = datetime.now()
    day_dir = TRANSCRIPTS_DIR / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    filename = now.strftime("%Y-%m-%d_%H-%M-%S.md")
    filepath = day_dir / filename
    metadata = {"timestamp": now.isoformat(), "device": device_name, "model": MODEL_NAME, "language": language}
    content = f"---\n{yaml.dump(metadata, default_flow_style=False)}---\n\n{text}\n"
    filepath.write_text(content)
    return filepath


def main() -> None:
    """Main entry point."""
    settings = load_settings()

    if settings.get("transcribe_only"):
        text = transcribe(settings["transcribe_only"], settings["language"])
        filepath = save_transcript(text, "file", settings["language"])
        print(f"\n{text}\n\nSaved to: {filepath}")
        return

    device_index, device_name = select_input_device()
    audio = record_audio(device_index, settings["sample_rate"], settings["level_bar_width"])
    save_audio(audio, settings["output"], settings["sample_rate"])
    text = transcribe(settings["output"], settings["language"])
    filepath = save_transcript(text, device_name, settings["language"])
    print(f"\n{text}\n\nSaved to: {filepath}")


if __name__ == "__main__":
    main()
