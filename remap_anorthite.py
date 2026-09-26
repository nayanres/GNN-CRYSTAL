from pathlib import Path

src = Path("data.Anorthite")
dst = Path("data.Anorthite.matsui")

mapping = {
    1: 5,  # O
    2: 1,  # Ca
    3: 3,  # Al
    4: 4,  # Si
}

lines = src.read_text().splitlines()

out = []
in_atoms = False
counts = {1: 0, 2: 0, 3: 0, 4: 0}

for line in lines:
    stripped = line.strip()

    if stripped == "Atoms":
        in_atoms = True
        out.append(line)
        continue

    if not in_atoms or not stripped:
        out.append(line)
        continue

    parts = line.split()

    if len(parts) >= 6 and parts[0].isdigit():
        old_type = int(parts[1])
        new_type = mapping[old_type]

        parts[1] = str(new_type)
        out.append("\t".join(parts))

        counts[old_type] += 1
    else:
        out.append(line)

dst.write_text("\n".join(out) + "\n")

print("Created:", dst)
print("Original type counts:", counts)
print("Matsui type counts:")
print("  Type 1 (Ca):", counts[2])
print("  Type 2 (Mg):", 0)
print("  Type 3 (Al):", counts[3])
print("  Type 4 (Si):", counts[4])
print("  Type 5 (O): ", counts[1])