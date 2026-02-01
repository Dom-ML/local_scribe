#!/usr/bin/env python3
"""Local speech-to-text transcription using Qwen ASR on Apple Silicon."""

from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml
from mlx_audio.stt import load

SETTINGS_FILE = Path(__file__).parent / "settings.yaml"


def load_settings() -> dict:
    """Load settings from YAML file."""
    defaults = {"output": "recording.wav", "language": "English", "sample_rate": 16000, "level_bar_width": 30, "transcribe_only": None}
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return {**defaults, **yaml.safe_load(f)}
    return defaults


def select_input_device() -> int:
    """Display available input devices and let user select one."""
    devices = [(i, dev["name"]) for i, dev in enumerate(sd.query_devices()) if dev["max_input_channels"] > 0]
    print("Select audio input device:")
    for i, (idx, name) in enumerate(devices):
        print(f"  [{i + 1}] {name}")
    while True:
        try:
            choice = int(input("\nEnter number: ")) - 1
            if 0 <= choice < len(devices):
                return devices[choice][0]
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
    model = load("mlx-community/Qwen3-ASR-0.6B-8bit")
    result = model.generate(audio_path, language=language)
    return result.text


def main() -> None:
    """Main entry point."""
    settings = load_settings()

    if settings.get("transcribe_only"):
        text = transcribe(settings["transcribe_only"], settings["language"])
        print(f"\n{text}")
        return

    device = select_input_device()
    audio = record_audio(device, settings["sample_rate"], settings["level_bar_width"])
    save_audio(audio, settings["output"], settings["sample_rate"])
    text = transcribe(settings["output"], settings["language"])
    print(f"\n{text}")


if __name__ == "__main__":
    main()
