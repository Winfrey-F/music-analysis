from analysis.preprocess import preprocess_midi

notes = preprocess_midi("data/test.mid")

print(f"Total notes: {len(notes)}")
print("First 5 notes:")
for n in notes[:5]:
    print(n)