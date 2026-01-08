import numpy as np
import matplotlib.pyplot as plt

from analysis.preprocess import preprocess_midi
from analysis.features import extract_features
from analysis.skeleton import aggregate_windows
from analysis.ssm import (
    compute_ssm,
    compute_novelty_curve,
    novelty_segmentation
)
from analysis.repetition import (
    extract_section_features,
    compute_section_similarity,
    analyze_repetition
)

# ========= 1️⃣ MIDI → Notes =========
notes = preprocess_midi("data/test.mid")

# ========= 2️⃣ Frame-level features =========
frames = extract_features(notes)

times = np.array([f.time for f in frames])
features = np.stack([
    [f.note_density for f in frames],
    [f.mean_pitch for f in frames],
    [f.pitch_range for f in frames]
], axis=1)

# ========= 3️⃣ Window aggregation =========
windows = aggregate_windows(
    times=times,
    features=features,
    window_size=2.0,
    hop_size=0.5
)

window_times = np.array([w["start"] for w in windows])
window_feats = np.stack([w["feat"] for w in windows])

# ========= 4️⃣ SSM =========
ssm = compute_ssm(window_feats)

# ========= 5️⃣ Novelty =========
novelty = compute_novelty_curve(ssm, kernel_size=8)

# ========= 6️⃣ Segmentation =========
stage4_sections = novelty_segmentation(
    times=window_times,
    novelty=novelty,
    peak_prominence=0.25,
    min_section_length=4.0
)

# ========= 7️⃣ Visualization =========
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.imshow(ssm, origin="lower", aspect="auto", cmap="magma")
plt.title("Self-Similarity Matrix")

plt.subplot(2, 1, 2)
plt.plot(window_times, novelty, label="Novelty Curve")
for sec in stage4_sections:
    plt.axvline(sec["start"], color="red", linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# ========= 8️⃣ Print Stage 4 sections =========
print("\nDetected Sections (Stage 4):")
for i, sec in enumerate(stage4_sections):
    dur = sec["end"] - sec["start"]
    print(f"{i+1}: {sec['start']:.2f}s -> {sec['end']:.2f}s (len={dur:.2f}s)")

# ========= 9️⃣ Stage 5: Repetition / Variation Analysis =========
# Extract section-level features
section_features = extract_section_features(stage4_sections, notes)

# Compute section similarity matrix
section_sim_matrix = compute_section_similarity(section_features)

# Analyze repetition/variation relationships
stage5_adjacency = analyze_repetition(section_features, section_sim_matrix, repeat_threshold=0.85)

# ========= 🔹 Print Stage 5 results =========
print("\nSection Repetition/Variation Adjacency (Stage 5):")
for i, related in stage5_adjacency.items():
    if related:
        related_str = ", ".join([str(r) for r in related])
        sec = stage4_sections[i]
        print(f"Section {i} ({sec['start']:.2f}-{sec['end']:.2f}s) repeats/varies with sections: {related_str}")

        # ========= 10️⃣ Stage 6: Salient Events Detection =========
from analysis.salient_events import detect_salient_events

# Detect events using frame-level features + Stage4 sections
stage6_events = detect_salient_events(frames, stage4_sections, peak_prominence=0.2)

# ========= 11️⃣ Visualization including salient events =========
plt.figure(figsize=(14, 5))

# Feature curves
plt.plot(times, [f.note_density for f in frames], label="Note Density")
plt.plot(times, [f.mean_pitch for f in frames], label="Mean Pitch", color="orange")
plt.plot(times, [f.pitch_range for f in frames], label="Pitch Range", color="green")

# Stage4 section boundaries
for sec in stage4_sections:
    plt.axvline(sec["start"], color="red", linestyle="--", alpha=0.5)
    plt.axvline(sec["end"], color="red", linestyle="--", alpha=0.5)

# Salient events
for ev in stage6_events:
    if ev["type"].startswith("peak"):
        plt.scatter(ev["time"], 0, color="black", marker="v", s=50, alpha=0.7, label="Peak Event")
    elif ev["type"] in ("section_start", "section_end"):
        plt.scatter(ev["time"], 0, color="blue", marker="o", s=50, alpha=0.6, label="Section Boundary Event")

plt.xlabel("Time (s)")
plt.ylabel("Feature Value")
plt.title("Stage 6: Salient Events with Stage 4 Sections")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# ========= 12️⃣ Print Stage 6 events =========
print("\nDetected Salient Events (Stage 6):")
for ev in stage6_events:
    print(f"{ev['time']:.2f}s: {ev['type']} (strength={ev['strength']:.2f})")
