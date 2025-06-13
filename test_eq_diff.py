import torchaudio

# ความแตกต่างระหว่าง before และ EQ
w1, _ = torchaudio.load("P9d - Everyday.wav")
w2, _ = torchaudio.load("vocals_eq.wav")
diff_eq = (w1 - w2).abs().mean()
print("ค่าเฉลี่ยความต่าง EQ:", diff_eq.item())

# ความแตกต่างระหว่าง before และ Compressor
w1com, _ = torchaudio.load("P9d - Everyday.wav")
w2com, _ = torchaudio.load("P9d - Everyday_compressed.wav")
diff_com = (w1com - w2com).abs().mean()
print("ค่าเฉลี่ยความต่าง Compressor:", diff_com.item())
