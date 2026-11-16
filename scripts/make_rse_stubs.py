"""Writes the RSE section stubs (idempotent; never overwrites a non-empty drafted file)."""
import os

D = os.path.join(os.path.dirname(__file__), '..', 'manuscript_rse', 'sections')
NL = '\n'
STUBS = {
    '01_introduction': [r'\section{Introduction}', r'\label{sec:intro}',
        '% Blueprint I (peer_review_report_v2.md Sec. 7). Grounding: clarification.md Entry 53; references.bib.',
        '% Drafting order: after Sections 3-7.', r'\pivalidate{Section to be drafted.}'],
    '02_background': [r'\section{Background: radiative transfer and the TES algorithm}', r'\label{sec:background}',
        '% Blueprint II. RTE (Eqs. of Sec. 3.4 reused), hemispheric L_down (D4), TES: NEM, ratio, MMD (gillespie1998temperature).',
        r'\pivalidate{Section to be drafted.}'],
    '04_methods': [r'\section{Sobolev-regularised physics-informed network}', r'\label{sec:methods}',
        '% Blueprint IV. Grounding: D8 (pinn_grounding.md, Run D addendum), D11 (best_config_*.json), D10 code.',
        r'\pivalidate{Section to be drafted.}'],
    '05_experimental_design': [r'\section{Experimental design}', r'\label{sec:design}',
        '% 5-fold OOF protocol; quadrants Q-I..Q-IV (Sec. 3.5); paired TES baseline; metrics; cluster bootstrap.',
        '% LOCO protocol (D13) -- MUST disclose: pool = fold-0 train (augmented: random T, 20 atm, m~U(0.5,1.5), s~U(1.84,4.1) mm)',
        '%   + fold-0 val (pristine grid); hyperparameters come from the Fold-0 HPO in which the held-out class was present;',
        '%   checkpoint selection on a group-aware 85/15 in-distribution split grouped by sample_id; LOCO classes with >= 15 spectra.',
        r'\pivalidate{Section to be drafted.}'],
    '06_results': [r'\section{Results}', r'\label{sec:results}',
        '% Grounding: D12 (analysis_grounding_run_D.md, docs/analysis_D tables/figures) and D13 (LOCO master).',
        r'\pivalidate{Section to be drafted.}'],
    '07_discussion': [r'\section{Discussion}', r'\label{sec:discussion}', r'\subsection{Limitations}',
        r'\label{sec:discussion:limitations}',
        '% Synthetic atmospheres; known temperature profile (only PWV error); fixed rural aerosol (5 km) and CO2 300 ppmv (D3 Sec. 3);',
        '% thermal decoupling; kaolinite contrast shrinkage; single seed per fold.',
        r'\pivalidate{Section to be drafted.}'],
    '08_conclusions': [r'\section{Conclusions}', r'\label{sec:conclusions}', r'\pivalidate{Section to be drafted.}'],
    'appB_validation': [r'\section{Dataset validation protocol}', r'\label{app:validation}',
        '% 32 test IDs; families A (partition integrity and leakage), B (physical and numerical integrity),',
        '% C (statistical design and bias), D (cross-validation structure). Source: docs/dataset_validation_protocol.md, D6.',
        '% A0 A1 A1b A2 A3a A3b A3c A4 A5 A5b A6 | B1 B2 B2b B2c B2d B3 B4 B5 B6 B7 B8 | C1 C2 C3 C4 C5 C7 C7b C8 C9 | D1',
        r'\pivalidate{Table of tests (ID, purpose, rule, result) to be generated from the protocol and report.}'],
}

os.makedirs(D, exist_ok=True)
for name, lines in STUBS.items():
    path = os.path.join(D, name + '.tex')
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print('kept (non-empty):', name)
        continue
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(NL.join(lines) + NL)
    print('wrote stub:', name)

