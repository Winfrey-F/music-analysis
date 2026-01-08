import json

def export_music_structure(filename,
                           skeleton_sections,
                           ssm_sections,
                           repetition_adjacency,
                           salient_events):
    """
    导出音乐结构分析结果到 JSON 文件

    Parameters
    ----------
    filename: str
        输出 JSON 文件路径
    skeleton_sections: list of dict
        Stage 3 输出 [{"start": , "end": }, ...]
    ssm_sections: list of dict
        Stage 4 输出 [{"start": , "end": }, ...]
    repetition_adjacency: dict
        Stage 5 输出 {section_idx: [related_section_indices], ...}
    salient_events: list of dict
        Stage 6 输出 [{"time": , "type": , "strength": }, ...]
    """
    music_structure = {
        "skeleton_sections": skeleton_sections,
        "ssm_sections": ssm_sections,
        "repetition_adjacency": repetition_adjacency,
        "salient_events": salient_events
    }

    with open(filename, "w") as f:
        json.dump(music_structure, f, indent=4)

    print(f"Music structure exported to {filename}")
