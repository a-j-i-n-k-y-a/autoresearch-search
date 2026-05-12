# Run this once as a one-off script: fix_log.py
import json

def parse_agent_response(raw):
    raw = raw.strip()
    lines = raw.split("\n")
    first = lines[0].strip()
    is_code_line = (
        first.startswith("import") or first.startswith("from")
        or first.startswith("def ") or first.startswith("class ")
        or first.startswith("```") or first.startswith("#!")
    )
    if not is_code_line and len(first) < 80:
        raw = "\n".join(lines[1:]).strip()
    lines = raw.split("\n")
    lines = [l for l in lines if not l.strip().startswith("```")]
    return "\n".join(lines).strip()

with open("experiments/log.jsonl") as f:
    content = f.read()

records = []
for chunk in content.strip().split("\n\n"):
    chunk = chunk.strip()
    if chunk:
        try:
            records.append(json.loads(chunk))
        except json.JSONDecodeError:
            continue

for r in records:
    r["search_py"] = parse_agent_response(r["search_py"])

with open("experiments/log.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r, indent=2) + "\n\n")

print(f"Cleaned {len(records)} records.")