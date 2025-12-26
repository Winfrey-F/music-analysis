import numpy as np
from typing import List, Dict
from analysis.preprocess import Note
from difflib import SequenceMatcher

def pitch_contour(notes: List[Note]) -> List[str]:
    """将音高序列转成 ↑ ↓ = 轮廓"""
    if not notes:
        return []
    contour = []
    for i in range(1, len(notes)):
        if notes[i].pitch > notes[i-1].pitch:
            contour.append("↑")
        elif notes[i].pitch < notes[i-1].pitch:
            contour.append("↓")
        else:
            contour.append("=")
    return contour

def contour_similarity(c1: List[str], c2: List[str]) -> float:
    """计算两个轮廓的相似度(0~1)"""
    s = SequenceMatcher(None, c1, c2)
    return s.ratio()

def refine_sections(notes: List[Note], sections: List[Dict],
                    window_size: int = 8, sim_threshold: float = 0.6,
                    min_section_length: float = 2.0) -> List[Dict]:
    """
    改进版主题微调段落
    """
    refined_sections = []

    for sec in sections:
        # 提取段落内音符
        seg_notes = [n for n in notes if sec["start_time"] <= n.start_time < sec["end_time"]]
        if not seg_notes:
            continue

        contour = pitch_contour(seg_notes)
        n = len(contour)
        # 边界微调
        start_idx, end_idx = 0, n
        # 向前看是否有与前一段相似的主题，微调起点
        if refined_sections:
            prev_sec_notes = [n for n in notes if refined_sections[-1]["start_time"] <= n.start_time < refined_sections[-1]["end_time"]]
            prev_contour = pitch_contour(prev_sec_notes)
            # 滑动匹配
            for i in range(min(window_size, n)):
                sim = contour_similarity(contour[i:i+window_size], prev_contour[-window_size:])
                if sim > sim_threshold:
                    start_idx = i + window_size
                    break
        # 向后看，微调终点
        for i in range(n - window_size):
            sim = contour_similarity(contour[i:i+window_size], contour[:window_size])
            if sim > sim_threshold:
                end_idx = i
                break

        new_start = seg_notes[start_idx].start_time if start_idx < len(seg_notes) else sec["start_time"]
        new_end = seg_notes[end_idx-1].start_time + seg_notes[end_idx-1].duration if end_idx-1 < len(seg_notes) else sec["end_time"]

        refined_sections.append({
            "start_time": new_start,
            "end_time": new_end,
            "feature_summary": sec.get("feature_summary", {})
        })

    # 合并过短段落
    merged_sections = []
    for sec in refined_sections:
        if merged_sections and (sec["end_time"] - merged_sections[-1]["start_time"]) < min_section_length:
            merged_sections[-1]["end_time"] = sec["end_time"]
        else:
            merged_sections.append(sec)

    return merged_sections
