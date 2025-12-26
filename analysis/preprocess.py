from dataclasses import dataclass

@dataclass
class Note:
    start_time: float
    duration: float
    pitch: int
    velocity: int
    track: int


import pretty_midi
from typing import List

def preprocess_midi(midi_file: str,
                    min_velocity: int = 20,
                    min_duration: float = 0.05) -> List[Note]:
    """
    将 MIDI 转换成标准化 Note 列表
    """
    pm = pretty_midi.PrettyMIDI(midi_file)
    notes = []

    for track_id, instrument in enumerate(pm.instruments):
        # 可选：跳过鼓声
        if instrument.is_drum:
            continue

        for n in instrument.notes:
            if n.velocity < min_velocity or n.end - n.start < min_duration:
                continue
            notes.append(Note(
                start_time=n.start,
                duration=n.end - n.start,
                pitch=n.pitch,
                velocity=n.velocity,
                track=track_id
            ))

    # 按起始时间排序
    notes.sort(key=lambda x: x.start_time)
    return notes
