#!/usr/bin/env python3
"""Local speech-to-text transcription using Qwen ASR on Apple Silicon."""

import argparse

import numpy as np
import sounddevice as sd
import soundfile as sf
from mlx_audio.stt import load

DEFAULT_DURATION = 10
DEFAULT_OUTPUT_FILE = "recording.wav"
DEFAULT_LANGUAGE = "English"
SAMPLE_RATE = 16000
LEVEL_BAR_WIDTH = 30


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Local speech-to-text transcription using Qwen ASR")
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices")
    parser.add_argument("-d", "--device", type=int, help="Audio input device index")
    parser.add_argument("-t", "--duration", type=int, default=DEFAULT_DURATION, help=f"Recording duration in seconds (default: {DEFAULT_DURATION})")
    parser.add_argument("-o", "--output", type=str, default=DEFAULT_OUTPUT_FILE, help=f"Output audio file path (default: {DEFAULT_OUTPUT_FILE})")
    parser.add_argument("-l", "--language", type=str, default=DEFAULT_LANGUAGE, help=f"Transcription language (default: {DEFAULT_LANGUAGE})")
    parser.add_argument("-T", "--transcribe-only", type=str, metavar="FILE", help="Transcribe existing audio file")
    return parser.parse_args()


def list_input_devices() -> None:
    """Print available audio input devices."""
    print("Audio input devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"  [{i}] {dev['name']}")


def record_audio(device: int, duration: int) -> np.ndarray:
    """Record audio from the specified device with live level display."""
    print(f"Recording {duration}s... (speak now)")
    audio_chunks: list[np.ndarray] = []

    def callback(indata: np.ndarray, frames: int, time, status) -> None:
        audio_chunks.append(indata.copy())
        rms = np.sqrt(np.mean(indata**2))
        level = min(int(rms * 200), LEVEL_BAR_WIDTH)
        bar = "\u2588" * level + "\u2591" * (LEVEL_BAR_WIDTH - level)
        print(f"\r  [{bar}] ", end="", flush=True)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, device=device, dtype="float32", callback=callback):
        sd.sleep(duration * 1000)

    print()
    return np.concatenate(audio_chunks)


def save_audio(audio: np.ndarray, output_path: str) -> None:
    """Save audio data to a WAV file."""
    sf.write(output_path, audio, SAMPLE_RATE)


def transcribe(audio_path: str, language: str) -> str:
    """Transcribe audio file using Qwen ASR."""
    print("Transcribing...")
    model = load("mlx-community/Qwen3-ASR-0.6B-8bit")
    result = model.generate(audio_path, language=language)
    return result.text


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.list_devices:
        list_input_devices()
        return

    if args.transcribe_only:
        text = transcribe(args.transcribe_only, args.language)
        print(f"\n{text}")
        return

    if args.device is None:
        list_input_devices()
        print("\nError: Device index required. Use -d <index> to specify.")
        raise SystemExit(1)

    audio = record_audio(args.device, args.duration)
    save_audio(audio, args.output)
    text = transcribe(args.output, args.language)
    print(f"\n{text}")


if __name__ == "__main__":
    main()
