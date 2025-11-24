import torchaudio

# Quick helper to compare mean absolute difference before/after EQ and Compressor
w1, _ = torchaudio.load("P9d - Everyday.wav")
w2, _ = torchaudio.load("vocals_eq.wav")
diff_eq = (w1 - w2).abs().mean()
print("EQ difference:", diff_eq.item())

w1com, _ = torchaudio.load("P9d - Everyday.wav")
w2com, _ = torchaudio.load("P9d - Everyday_compressed.wav")
diff_com = (w1com - w2com).abs().mean()
print("Compressor difference:", diff_com.item())
