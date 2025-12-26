import pretty_midi

pm = pretty_midi.PrettyMIDI("data/test.mid")

print("Duration:", pm.get_end_time())
print("Number of tracks:", len(pm.instruments))
