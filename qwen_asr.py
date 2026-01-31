from mlx_audio.stt import load

# Speech recognition
model = load("mlx-community/Qwen3-ASR-0.6B-8bit")
result = model.generate("audio.wav", language="English")
print(result.text)

# Word-level forced alignment
#aligner = load("mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
#result = aligner.generate("audio.wav", text="I have a dream", language="English")
#for item in result:
#    print(f"[{item.start_time:.2f}s - {item.end_time:.2f}s] {item.text}")
