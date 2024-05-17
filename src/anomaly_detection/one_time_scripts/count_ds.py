from collections import Counter, defaultdict

from anomaly_detection.config import (
    CONDITION_ROOT,
    SCAV_PORT_ROOT,
    SCAVPORT_CLF_CLASSES,
    VESSEL_ARCHIVE_ROOT,
)

counts = defaultdict(lambda: Counter(dict.fromkeys(["train", "dev", "test"], 0)))


for f in VESSEL_ARCHIVE_ROOT.rglob("*.*"):
    counts[f.parts[-3]][f.parts[-2]] += 1

for f in SCAV_PORT_ROOT.rglob("*.*"):
    counts[f.parts[-2]][f.parts[-3]] += 1

lines = []
for k, v in counts.items():
    lines.append([k, v["train"], v["dev"], v["test"]])
    # lines.append(f'{k:<50} {v["train"]:>6} {v["dev"]:>6} {v["test"]:>6}')
lines.sort(
    key=lambda x: "A"
    if any(c in x[0] for c in SCAVPORT_CLF_CLASSES)
    else "AA"
    if "good" in x[0]
    else str(sum(x[1:])),
    reverse=True,
)
# lines.insert(0, f'{"class name":<50} {"train":>6} {"dev":>6} {"test":>6}')
lines.insert(0, ["class name", "train", "dev", "test"])
print("\n".join([f"{x[0]:<50} {x[1]:>6} {x[2]:>6} {x[3]:>6}" for x in lines]))

print()

counts = defaultdict(lambda: Counter(dict.fromkeys(["train", "dev", "test"], 0)))
for f in CONDITION_ROOT.rglob("*.*"):
    key = f"{f.parts[-4]}[{f.parts[-2]}]"
    counts[key][f.parts[-3]] += 1

lines = []
for k, v in counts.items():
    lines.append([k, v["train"], v["dev"], v["test"]])
    # lines.append(f'{k:<50} {v["train"]:>6} {v["dev"]:>6} {v["test"]:>6}')
lines.sort(key=lambda k: f"0{k[0]}" if "good" in k[0] else k[0])
# lines.insert(0, f'{"class name":<50} {"train":>6} {"dev":>6} {"test":>6}')
lines.insert(0, ["class name", "train", "dev", "test"])
print("\n".join([f"{x[0]:<50} {x[1]:>6} {x[2]:>6} {x[3]:>6}" for x in lines]))
