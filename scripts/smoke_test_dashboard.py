"""Headless smoke test of the dashboard (all tabs) with streamlit.testing AppTest."""
import sys
from streamlit.testing.v1 import AppTest

path = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\Atilio\Desktop\PROJETOS\AITIR\zenodo_release\dataset_explorer_dashboard.py"
at = AppTest.from_file(path, default_timeout=300)
at.run()
print("exceptions:", [e.value for e in at.exception])
print("errors:", [e.value for e in at.error])
print("tabs:", [t.label for t in at.tabs])
subs = [s.value for s in at.subheader]
print("validation subheader present:", any("Dataset Validation Report" in s for s in subs))
print("metrics (first 12):", [(m.label, m.value) for m in at.metric][:12])
print("dataframes rendered:", len(at.dataframe))

