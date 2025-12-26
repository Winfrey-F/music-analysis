import numpy as np
import matplotlib.pyplot as plt

from analysis.preprocess import preprocess_midi
from analysis.features import extract_features
from analysis.skeleton import (
    aggregate_windows,
    compute_state_changes,
    extract_skeleton_sections
)

# ==============================
# 1️⃣ MIDI 预处理
# ==============================
notes = preprocess_midi("data/test.mid")

# ==============================
# 2️⃣ 帧级特征提取
# ==============================
frames = extract_features(notes)

times = np.array([f.time for f in frames])

features = np.stack([
    np.array([f.note_density for f in frames]),
    np.array([f.mean_pitch for f in frames]),
    np.array([f.pitch_range for f in frames])
], axis=1)

# ==============================
# 3️⃣ 窗口聚合（观察尺度）
# ==============================
windows = aggregate_windows(
    times=times,
    features=features,
    window_size=1.5,   # 观察尺度
    hop_size=0.5
)

window_times = np.array([w["start"] for w in windows])
window_feats = np.stack([w["feat"] for w in windows])

# ==============================
# 4️⃣ 状态变化曲线
# ==============================
changes = compute_state_changes(
    windows,
    smooth_sigma=2.0
)

# ==============================
# 5️⃣ Skeleton Sections
# （不按秒切，只在变化峰值处允许切）
# ==============================
sections = extract_skeleton_sections(
    windows=windows,
    changes=changes,
    min_section_length=4.0,   # ≈ 至少 4 小节（硬约束）
    peak_ratio=0.6            # 自适应阈值
)

# ==============================
# 6️⃣ 可视化
# ==============================
plt.figure(figsize=(14, 4))

plt.plot(window_times, window_feats[:, 0], label="Note Density")
plt.plot(window_times, window_feats[:, 1], label="Mean Pitch")
plt.plot(window_times, window_feats[:, 2], label="Pitch Range")

# Skeleton 边界
for sec in sections:
    plt.axvline(sec["start"], color="red", linestyle="--", alpha=0.6)
    plt.axvline(sec["end"], color="red", linestyle="--", alpha=0.6)

plt.title("Stage 3 – Skeleton Sections")
plt.xlabel("Time (s)")
plt.ylabel("Aggregated Feature Value")
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# 7️⃣ 打印结果
# ==============================
print("\nDetected Skeleton Sections (Stage 3):")
for i, sec in enumerate(sections):
    dur = sec["end"] - sec["start"]
    print(f"{i+1}: {sec['start']:.2f}s -> {sec['end']:.2f}s  (len={dur:.2f}s)")
