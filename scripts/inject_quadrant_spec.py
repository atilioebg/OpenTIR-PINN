import os

with open("scripts/quadrant_deep_spec.txt", "r", encoding="utf-8") as f:
    expander_code = f.read()

target_marker = '            "Extreme Joint OOD (**Quadrant IV:** 770 samples)."\n        )\n'

target_files = [
    "zenodo_release/dataset_explorer_dashboard.py",
    "scripts/dataset_explorer_dashboard.py"
]

for fp in target_files:
    if not os.path.exists(fp):
        continue
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()

    is_crlf = "\r\n" in content
    norm_content = content.replace("\r\n", "\n")

    if target_marker in norm_content:
        replacement = target_marker + "\n" + expander_code + "\n"
        norm_content = norm_content.replace(target_marker, replacement, 1)
        if is_crlf:
            final_content = norm_content.replace("\n", "\r\n")
        else:
            final_content = norm_content
        with open(fp, "w", encoding="utf-8") as f:
            f.write(final_content)
        print(f"Successfully injected quadrant deep specification into {fp}")
    else:
        print(f"Target marker not found in {fp}")

print("Injection complete.")

