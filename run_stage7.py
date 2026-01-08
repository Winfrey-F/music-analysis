from analysis.preprocess import preprocess_midi
from analysis.features import extract_features
from analysis.skeleton import aggregate_windows, compute_state_changes, extract_skeleton_sections
from analysis.repetition import extract_section_features, compute_section_similarity, analyze_repetition
from analysis.ssm import compute_ssm, compute_novelty_curve, novelty_segmentation
from analysis.salient_events import detect_salient_events
from analysis.music_structure import export_music_structure

# ========= 1️⃣ Stage 3: Skeleton Sections =========
notes = preprocess_midi("data/test.mid")
frames = extract_features(notes)

times = [f.time for f in frames]
features = [[f.note_density, f.mean_pitch, f.pitch_range] for f in frames]

windows = aggregate_windows(times, features, window_size=1.5, hop_size=0.5)
changes = compute_state_changes(windows)
skeleton_sections = extract_skeleton_sections(windows, changes, peak_ratio=0.8, min_section_length=3.0)

# ========= 2️⃣ Stage 4: SSM segmentation =========
window_feats = [w["feat"] for w in windows]
ssm = compute_ssm(window_feats)
novelty = compute_novelty_curve(ssm, kernel_size=8)
ssm_sections = novelty_segmentation(
    times=[w["start"] for w in windows],
    novelty=novelty,
    peak_prominence=0.25,
    min_section_length=4.0
)

# ========= 3️⃣ Stage 5: Repetition/Variation =========
section_features = extract_section_features(skeleton_sections, notes)
sim_matrix = compute_section_similarity(section_features)
repetition_adjacency = analyze_repetition(section_features, sim_matrix, repeat_threshold=0.85)

# ========= 4️⃣ Stage 6: Salient Events =========
salient_events = detect_salient_events(frames, skeleton_sections, peak_prominence=0.2)

# ========= 5️⃣ Stage 7: Export =========
export_music_structure(
    filename="output/music_structure.json",
    skeleton_sections=skeleton_sections,
    ssm_sections=ssm_sections,
    repetition_adjacency=repetition_adjacency,
    salient_events=salient_events
)
