from analysis.preprocess import preprocess_midi
from analysis.features import extract_features, FeatureFrame
import matplotlib.pyplot as plt

notes = preprocess_midi("data/test.mid")
frames = extract_features(notes)

# 提取特征曲线
times = [f.time for f in frames]
note_density = [f.note_density for f in frames]
mean_pitch = [f.mean_pitch for f in frames]
pitch_range = [f.pitch_range for f in frames]

# 绘图
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(times, note_density, label='Note Density')
plt.ylabel('Note Density')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(times, mean_pitch, label='Mean Pitch', color='orange')
plt.ylabel('Mean Pitch')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(times, pitch_range, label='Pitch Range', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Pitch Range')
plt.legend()

plt.tight_layout()
plt.show()
