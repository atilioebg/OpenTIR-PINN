"""Headless test of the dashboard's pipeline runner: click Run (validation-only), poll until done, check report reload."""
import sys, time
from streamlit.testing.v1 import AppTest

path = r"c:\Users\Atilio\Desktop\PROJETOS\AITIR\zenodo_release\dataset_explorer_dashboard.py"
at = AppTest.from_file(path, default_timeout=600)
at.run()
print("initial exceptions:", [e.value for e in at.exception])
btn = [b for b in at.button if b.key == "val_run_btn"]
sel = [s for s in at.selectbox if s.key == "val_mode"]
print("run button found:", bool(btn), "| modes:", sel[0].options if sel else None)
sel[0].set_value("Validation only (manifest + 28 tests, ~2-3 min)").run()
[b for b in at.button if b.key == "val_run_btn"][0].click().run()
print("after click exceptions:", [e.value for e in at.exception])
print("progress shown:", [p.value for p in at.get("progress")][:1])
t0 = time.time()
while "val_run" in at.session_state and time.time() - t0 < 600:
    time.sleep(15)
    at.run()
    prog = at.get("progress")
    print(f"  t={time.time() - t0:5.0f}s running={'val_run' in at.session_state} progress={[p.value for p in prog][:1]}")
print("finished; exceptions:", [e.value for e in at.exception])
print("success msgs:", [s.value[:120] for s in at.success])
print("error msgs:", [e.value[:200] for e in at.error])
print("val_last:", at.session_state["val_last"] if "val_last" in at.session_state else None)

