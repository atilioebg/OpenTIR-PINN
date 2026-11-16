for path in ['zenodo_release/dataset_explorer_dashboard.py', 'scripts/dataset_explorer_dashboard.py']:
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
    txt = txt.replace('\\"\\"\\")', '""")')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(txt)
    print(f'Fixed {path}')

