import numpy as np
from scipy.ndimage import gaussian_filter1d

# =========================
# 1. 窗口聚合（观察用）
# =========================
def aggregate_windows(times, features, window_size=1.5, hop_size=0.5):
    # Convert to numpy arrays for element-wise operations
    times = np.array(times)
    features = np.array(features)
    
    windows = []
    t = times[0]
    t_end = times[-1]

    while t + window_size <= t_end:
        mask = (times >= t) & (times < t + window_size)
        if mask.sum() > 0:
            windows.append({
                "start": t,
                "end": t + window_size,
                "feat": features[mask].mean(axis=0)
            })
        t += hop_size

    return windows


# =========================
# 2. 状态变化曲线
# =========================
def compute_state_changes(windows, smooth_sigma=2.0):
    diffs = [0.0]
    for i in range(1, len(windows)):
        diffs.append(
            np.linalg.norm(windows[i]["feat"] - windows[i - 1]["feat"])
        )

    diffs = np.array(diffs)
    return gaussian_filter1d(diffs, sigma=smooth_sigma)


# =========================
# 3. Skeleton Sections（核心）
# =========================
def extract_skeleton_sections(
    windows,
    changes,
    min_section_length=4.0,   # 👈 音乐下限（例如 ≥4小节）
    peak_ratio=0.6,           # 👈 自适应阈值（相对比例）
    change_threshold=None     # 👈 绝对阈值（如果提供则优先使用）
):
    sections = []

    # 如果提供了 change_threshold，使用绝对阈值；否则使用相对比例
    if change_threshold is not None:
        threshold = change_threshold
    else:
        threshold = peak_ratio * np.max(changes)

    current_start = windows[0]["start"]

    for i in range(1, len(changes) - 1):
        is_peak = (
            changes[i] > threshold and
            changes[i] > changes[i - 1] and
            changes[i] > changes[i + 1]
        )

        if is_peak:
            candidate_end = windows[i]["start"]

            # 时间只是“否决条件”
            if candidate_end - current_start >= min_section_length:
                sections.append({
                    "start": current_start,
                    "end": candidate_end
                })
                current_start = candidate_end

    # 收尾段
    final_end = windows[-1]["end"]
    if final_end - current_start >= min_section_length:
        sections.append({
            "start": current_start,
            "end": final_end
        })

    return sections
