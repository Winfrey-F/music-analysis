"""
Stage 6: Detect Salient Events
- 输入：frames (特征曲线), sections (段落边界)
- 输出：事件列表，每个事件包含时间、类型、强度
"""

import numpy as np
from scipy.signal import find_peaks

def detect_salient_events(frames, sections,
                          feature_names=("note_density", "mean_pitch", "pitch_range"),
                          peak_prominence=0.2):
    """
    frames: 帧特征列表
    sections: 段落边界列表 [{"start": , "end": }, ...]
    feature_names: 哪些特征用于检测事件
    peak_prominence: 峰值显著性阈值
    """
    # 构造特征曲线字典
    features = {name: np.array([getattr(f, name) for f in frames]) for name in feature_names}
    times = np.array([f.time for f in frames])
    
    events = []

    # 1️⃣ 检测各特征峰值
    for name, curve in features.items():
        peaks, properties = find_peaks(curve, prominence=peak_prominence)
        for p, prom in zip(peaks, properties["prominences"]):
            events.append({
                "time": times[p],
                "type": f"peak_{name}",
                "strength": prom
            })

    # 2️⃣ 段落边界也作为事件
    for sec in sections:
        events.append({"time": sec["start"], "type": "section_start", "strength": 1.0})
        events.append({"time": sec["end"], "type": "section_end", "strength": 1.0})

    # 3️⃣ 按时间排序
    events.sort(key=lambda x: x["time"])

    return events