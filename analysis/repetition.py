import numpy as np

def extract_section_features(sections, notes):
    """
    将每个 section 转成 feature 向量
    输出列表，每个元素是 dict:
    {"start": float, "end": float, "feature": np.array}
    """
    section_features = []

    for sec in sections:
        sec_notes = [n for n in notes if sec["start"] <= n.start_time < sec["end"]]
        if len(sec_notes) < 3:
            continue

        pitches = np.array([n.pitch for n in sec_notes])
        durations = np.array([n.duration for n in sec_notes])
        onsets = np.array([n.start_time for n in sec_notes])

        # 简单估计 polyphony
        polyphony = [sum(1 for n2 in sec_notes if n2.start_time <= t < n2.start_time + n2.duration) for t in onsets]
        intervals = np.diff(pitches)

        feat = np.array([
            pitches.mean(),
            pitches.std(),
            len(sec_notes) / (sec["end"] - sec["start"]),
            np.mean(np.abs(intervals)) if len(intervals) > 0 else 0,
            np.mean(polyphony)
        ])

        section_features.append({
            "start": sec["start"],
            "end": sec["end"],
            "feature": feat
        })

    return section_features

def compute_section_similarity(section_features):
    """
    使用 NumPy 手动计算 cosine similarity
    """
    feats = np.stack([s["feature"] for s in section_features])
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-6)  # 标准化

    norm = np.linalg.norm(feats, axis=1, keepdims=True)
    norm_feats = feats / (norm + 1e-6)

    sim_matrix = norm_feats @ norm_feats.T  # cosine similarity
    return sim_matrix

def analyze_repetition(section_features, sim_matrix, repeat_threshold=0.85):
    """
    分析重复/变奏关系
    输出 adjacency dict:
        {section_index: [list of indices of similar sections]}
    """
    adjacency = {i: [] for i in range(len(section_features))}

    for i in range(len(section_features)):
        for j in range(i+1, len(section_features)):
            if sim_matrix[i, j] >= repeat_threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)

    return adjacency
