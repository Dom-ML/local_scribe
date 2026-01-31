#!/usr/bin/env python3
import sounddevice as sd
import soundfile as sf
from mlx_audio.stt import load

DURATION = 10
OUTPUT_FILE = "recording.wav"

# List only input devices
print("Audio input devices:")
for i, dev in enumerate(sd.query_devices()):
    if dev["max_input_channels"] > 0:
        print(f"  [{i}] {dev['name']}")

device = int(input("\nDevice index: "))

# Record
print(f"Recording {DURATION}s...")
audio = sd.rec(int(DURATION * 16000), samplerate=16000, channels=1, device=device, dtype="float32")
sd.wait()
sf.write(OUTPUT_FILE, audio, 16000)

# Transcribe
print("Transcribing...")
model = load("mlx-community/Qwen3-ASR-0.6B-8bit")
result = model.generate(OUTPUT_FILE, language="English")
print(f"\n{result.text}")
