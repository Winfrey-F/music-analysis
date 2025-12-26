## stage 2: extract temporal feature curves and plot
from dataclasses import dataclass

@dataclass
class FeatureFrame:
    time: float           # 时间窗口起点
    note_density: float   # 每秒音符数
    mean_pitch: float     # 平均音高
    pitch_range: float    # 音高范围
    velocity_energy: float # 力度能量（平均力度平方）
    polyphony: float      # 同时响的音符数均值


from typing import List
from analysis.preprocess import Note
import numpy as np

def extract_features(notes: List[Note],
                     window_size: float = 1.0,
                     hop_size: float = 0.5) -> List[FeatureFrame]:
    """
    将 Note 列表转换为时间窗特征曲线
    """
    if not notes:
        return []

    max_time = notes[-1].start_time + notes[-1].duration
    frames = []

    start = 0.0
    while start < max_time:
        # 找到窗口内的音符
        window_notes = [n for n in notes if n.start_time >= start and n.start_time < start + window_size]

        if window_notes:
            pitches = np.array([n.pitch for n in window_notes])
            velocities = np.array([n.velocity for n in window_notes])
            durations = np.array([n.duration for n in window_notes])
            
            note_density = len(window_notes) / window_size
            mean_pitch = np.mean(pitches)
            pitch_range = np.max(pitches) - np.min(pitches)
            velocity_energy = np.mean(velocities ** 2)
            
            # 粗略 polyphony：每个时间点同时响的音符数均值
            time_points = np.linspace(start, start + window_size, num=100)
            polyphony_values = []
            for t in time_points:
                polyphony_values.append(sum(1 for n in window_notes if n.start_time <= t < n.start_time + n.duration))
            polyphony = np.mean(polyphony_values)
        else:
            note_density = mean_pitch = pitch_range = velocity_energy = polyphony = 0.0

        frames.append(FeatureFrame(
            time=start,
            note_density=note_density,
            mean_pitch=mean_pitch,
            pitch_range=pitch_range,
            velocity_energy=velocity_energy,
            polyphony=polyphony
        ))

        start += hop_size

    return frames
