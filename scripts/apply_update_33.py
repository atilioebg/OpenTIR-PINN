import os

with open("scripts/update_33_content.txt", "r", encoding="utf-8") as f:
    new_content = f.read()

with open("clarification.md", "r", encoding="utf-8") as f:
    text = f.read()

p1 = text.find("3. **ASTER & HyTES Tooltip Parity with Explicit Wavelength Boundaries:**")
p2 = text.find("\n---\n\n## 34.", p1)

if p1 != -1 and p2 != -1:
    text = text[:p1] + new_content + "\n" + text[p2:]
    with open("clarification.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("Updated clarification.md successfully!")
else:
    print(f"Could not find markers: p1={p1}, p2={p2}")

