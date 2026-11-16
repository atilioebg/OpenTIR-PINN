"""Publication-Grade LaTeX to Word (.docx) Converter for OpenTIR-PINN Manuscript.
Specifically engineered for Elsevier / Remote Sensing of Environment (RSE) review and submission standards.

Key Features:
1. Native Microsoft Word Equations (OMML <m:oMath> in Cambria Math) for ALL inline math ($...$)
   and ALL display equations (equation and align environments), including real stacked fractions,
   subscripts, superscripts, combined sub/superscripts, hats, bars, overlines, and scaling delimiters.
2. Full hierarchical numbering for Sections (1., 2., ..., Appendix A.), Subsections (2.1., 2.2.),
   and Subsubsections (2.4.1., 2.4.2.), dynamically synchronized with ref cross-references.
3. Structural block isolation so lead-in paragraphs before enumerate/itemize lists and text
   following subsubsection headings are never dropped.
4. Unambiguous list formatting (indented at 0.25"/0.50" with bold item numbers/bullets and bold lead-in phrases).
5. Full multi-column table geometry (multicolumn horizontal cell merging, multi-row headers
   for Tables 7 & 8, unescaped-& splitting for Table 11, and proper footnote extraction for Table 3).
"""

import os
import re
from pathlib import Path
from lxml import etree
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls


# =============================================================================
# XML & Table Styling Helpers
# =============================================================================
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NSMAP = {'m': M_NS, 'w': W_NS}


def m_el(tag):
    return etree.Element(f"{{{M_NS}}}{tag}", nsmap=NSMAP)


def w_el(tag):
    return etree.Element(f"{{{W_NS}}}{tag}", nsmap=NSMAP)


def set_cell_margins(cell, top=70, bottom=70, left=90, right=90):
    """Set inner cell padding (20 dxa = 1 pt)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for name, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{name}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def set_cell_background(cell, hex_color):
    """Set background fill of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)


def set_table_borders(table, color="8FAADC", sz="6", val="single"):
    """Set clean academic booktabs-style horizontal borders."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="{val}" w:sz="10" w:space="0" w:color="1F3864"/>\n'
        f'  <w:bottom w:val="{val}" w:sz="10" w:space="0" w:color="1F3864"/>\n'
        f'  <w:insideH w:val="{val}" w:sz="4" w:space="0" w:color="{color}"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


# =============================================================================
# Native Word Equation Compiler (LaTeX -> OMML <m:oMath>)
# =============================================================================
def make_math_run(text, italic=True, bold=False):
    r = m_el('r')
    if not italic:
        rPr = m_el('rPr')
        nor = m_el('nor')
        rPr.append(nor)
        r.append(rPr)
    wrPr = w_el('rPr')
    rFonts = w_el('rFonts')
    rFonts.set(f"{{{W_NS}}}ascii", "Cambria Math")
    rFonts.set(f"{{{W_NS}}}hAnsi", "Cambria Math")
    wrPr.append(rFonts)
    if bold:
        b = w_el('b')
        wrPr.append(b)
    r.append(wrPr)
    t = m_el('t')
    if text.startswith(' ') or text.endswith(' ') or '\u2009' in text:
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    r.append(t)
    return r


GREEK_AND_SYMBOLS = {
    'alpha': ('α', True),
    'beta': ('β', True),
    'gamma': ('γ', True),
    'delta': ('δ', True),
    'varepsilon': ('ε', True),
    'epsilon': ('ϵ', True),
    'zeta': ('ζ', True),
    'eta': ('η', True),
    'theta': ('θ', True),
    'vartheta': ('ϑ', True),
    'kappa': ('κ', True),
    'lambda': ('λ', True),
    'mu': ('μ', True),
    'nu': ('ν', True),
    'xi': ('ξ', True),
    'pi': ('π', True),
    'rho': ('ρ', True),
    'sigma': ('σ', True),
    'tau': ('τ', True),
    'phi': ('ϕ', True),
    'varphi': ('φ', True),
    'chi': ('χ', True),
    'psi': ('ψ', True),
    'omega': ('ω', True),
    'Delta': ('Δ', False),
    'Gamma': ('Γ', False),
    'Lambda': ('Λ', False),
    'Omega': ('Ω', False),
    'Phi': ('Φ', False),
    'Pi': ('Π', False),
    'Sigma': ('Σ', False),
    'Theta': ('Θ', False),
    'partial': ('∂', False),
    'infty': ('∞', False),
    'emptyset': ('∅', False),
    'forall': ('∀', False),
    'exists': ('∃', False),
    'in': ('∈', False),
    'notin': ('∉', False),
    'cap': ('∩', False),
    'cup': ('∪', False),
    'equiv': ('≡', False),
    'approx': ('≈', False),
    'sim': ('∼', False),
    'le': ('≤', False),
    'leq': ('≤', False),
    'ge': ('≥', False),
    'geq': ('≥', False),
    'neq': ('≠', False),
    'ne': ('≠', False),
    'times': ('×', False),
    'cdot': ('·', False),
    'pm': ('±', False),
    'mp': ('∓', False),
    'oslash': ('⊘', False),
    'otimes': ('⊗', False),
    'to': ('→', False),
    'rightarrow': ('→', False),
    'leftarrow': ('←', False),
    'uparrow': ('↑', False),
    'downarrow': ('↓', False),
    'circ': ('°', False),
    'top': ('⊤', False),
    'dots': ('…', False),
    'ldots': ('…', False),
    'cdots': ('⋯', False),
    'langle': ('⟨', False),
    'rangle': ('⟩', False),
    'sum': ('∑', False),
    'int': ('∫', False),
    'prod': ('∏', False),
    'cos': ('cos', False),
    'sin': ('sin', False),
    'tan': ('tan', False),
    'exp': ('exp', False),
    'ln': ('ln', False),
    'log': ('log', False),
    'max': ('max', False),
    'min': ('min', False),
    'sup': ('sup', False),
    'inf': ('inf', False),
    'arg': ('arg', False),
    'quad': ('\u2009\u2009\u2009', False),
    'qquad': ('\u2009\u2009\u2009\u2009\u2009', False),
}

CALLIGRAPHIC_MAP = {
    'A': '𝒜', 'B': 'ℬ', 'C': '𝒞', 'D': '𝒟', 'E': 'ℰ', 'F': 'ℱ',
    'G': '𝒢', 'H': 'ℋ', 'I': 'ℐ', 'J': '𝒥', 'K': '𝒦', 'L': 'ℒ',
    'M': 'ℳ', 'N': '𝒩', 'O': '𝒪', 'P': '𝒫', 'Q': '𝒬', 'R': 'ℛ',
    'S': '𝒮', 'T': '𝒯', 'U': '𝒰', 'V': '𝒱', 'W': '𝒲', 'X': '𝒳',
    'Y': '𝒴', 'Z': '𝒵'
}

BLACKBOARD_MAP = {
    'R': 'ℝ', 'N': 'ℕ', 'Z': 'ℤ', 'Q': 'ℚ', 'C': 'ℂ', 'E': '𝔼', 'P': 'ℙ'
}


def preprocess_latex_math(s):
    """Expand project-specific macros before tokenizing into OMML."""
    s = s.replace(r'\Ldown', r'L^{\downarrow}')
    s = s.replace(r'\Lup', r'L^{\uparrow}')
    s = s.replace(r'\NEdT', r'\mathrm{NE\Delta T}')
    s = s.replace(r'\um', r'\,\mu\mathrm{m}')
    s = s.replace(r'\gcm', r'\,\mathrm{g\,cm}^{-2}')
    s = s.replace(r'\allowbreak', '')
    s = s.replace(r'\{', '⦃').replace(r'\}', '⦄')
    s = s.replace(r'\|', '‖')
    s = s.replace(r'\%', '%')
    s = s.replace(r'\&', '&')
    s = s.replace(r'\,', '\u2009')
    s = s.replace(r'\;', '\u2009')
    s = s.replace(r'\:', '\u2009')
    s = s.replace(r'\!', '')
    s = s.replace(r'\ ', '\u2009')
    s = s.replace('~', '\u2009')
    s = re.sub(r'\\(?:big|Big|bigg|Bigg)[lrm]?(?![a-zA-Z])', '', s)
    return s.strip('\t\r\n \u2009')


def read_braced_or_token(s, pos):
    """Read a {braced group} or a single command/character starting at pos."""
    while pos < len(s) and s[pos] in ' \t\r\n\u2009':
        pos += 1
    if pos >= len(s):
        return "", pos
    if s[pos] == '{':
        depth = 1
        start = pos + 1
        idx = start
        while idx < len(s) and depth > 0:
            if s[idx] == '{':
                depth += 1
            elif s[idx] == '}':
                depth -= 1
            idx += 1
        return s[start:idx-1], idx
    if s[pos] == '\\':
        idx = pos + 1
        if idx < len(s) and s[idx].isalpha():
            while idx < len(s) and s[idx].isalpha():
                idx += 1
            cmd = s[pos+1:idx]
            if cmd in ('hat', 'bar', 'tilde', 'overline', 'mathrm', 'mathbf', 'bm', 'mathcal', 'mathbb', 'text', 'textit', 'textbf', 'texttt'):
                arg, new_pos = read_braced_or_token(s, idx)
                return f"\\{cmd}{{{arg}}}", new_pos
            return s[pos:idx], idx
        else:
            return s[pos:pos+2], pos + 2
    return s[pos], pos + 1


def find_matching_right(s, pos):
    """Given position right after \\left<delim>, find matching \\right<delim> at depth 0."""
    depth = 1
    idx = pos
    while idx < len(s):
        if s.startswith(r'\left', idx) and (idx + 5 == len(s) or not s[idx+5].isalpha()):
            depth += 1
            idx += 5
            while idx < len(s) and s[idx] in ' \u2009':
                idx += 1
            if idx < len(s):
                idx += 1
            continue
        if s.startswith(r'\right', idx) and (idx + 6 == len(s) or not s[idx+6].isalpha()):
            depth -= 1
            if depth == 0:
                end_inner = idx
                idx += 6
                while idx < len(s) and s[idx] in ' \u2009':
                    idx += 1
                r_delim = ''
                if idx < len(s):
                    if s[idx] == '\\' and idx + 1 < len(s):
                        r_delim = s[idx:idx+2]
                        idx += 2
                    else:
                        r_delim = s[idx]
                        idx += 1
                return s[pos:end_inner], r_delim, idx
            idx += 6
            continue
        idx += 1
    return s[pos:], '', len(s)


def parse_math_nodes(s, italic_default=True, bold_default=False):
    """Parse a LaTeX math string into a list of OMML XML elements."""
    nodes = []
    pos = 0
    n = len(s)

    while pos < n:
        ch = s[pos]

        if ch in ' \t\r\n':
            pos += 1
            continue

        if ch == '\u2009':
            pos += 1
            nodes.append(make_math_run('\u2009', italic=False, bold=False))
            continue

        base_elems = None

        # 1. Braced group { ... }
        if ch == '{':
            inner, pos = read_braced_or_token(s, pos)
            if inner == ',':
                base_elems = [make_math_run(',', italic=False, bold=bold_default)]
            else:
                base_elems = parse_math_nodes(inner, italic_default=italic_default, bold_default=bold_default)

        # 2. LaTeX command \...
        elif ch == '\\':
            idx = pos + 1
            if idx < n and s[idx].isalpha():
                while idx < n and s[idx].isalpha():
                    idx += 1
                cmd = s[pos+1:idx]
                pos = idx

                if cmd in ('frac', 'tfrac', 'dfrac'):
                    num_str, pos = read_braced_or_token(s, pos)
                    den_str, pos = read_braced_or_token(s, pos)
                    f_el = m_el('f')
                    fPr = m_el('fPr')
                    t_el = m_el('type')
                    t_el.set(f"{{{M_NS}}}val", "bar")
                    fPr.append(t_el)
                    f_el.append(fPr)
                    num_el = m_el('num')
                    for c in parse_math_nodes(num_str, italic_default=italic_default, bold_default=bold_default):
                        num_el.append(c)
                    den_el = m_el('den')
                    for c in parse_math_nodes(den_str, italic_default=italic_default, bold_default=bold_default):
                        den_el.append(c)
                    f_el.append(num_el)
                    f_el.append(den_el)
                    base_elems = [f_el]

                elif cmd == 'sqrt':
                    arg_str, pos = read_braced_or_token(s, pos)
                    rad_el = m_el('rad')
                    radPr = m_el('radPr')
                    degHide = m_el('degHide')
                    degHide.set(f"{{{M_NS}}}val", "1")
                    radPr.append(degHide)
                    rad_el.append(radPr)
                    rad_el.append(m_el('deg'))
                    e_el = m_el('e')
                    for c in parse_math_nodes(arg_str, italic_default=italic_default, bold_default=bold_default):
                        e_el.append(c)
                    rad_el.append(e_el)
                    base_elems = [rad_el]

                elif cmd in ('hat', 'bar', 'tilde'):
                    arg_str, pos = read_braced_or_token(s, pos)
                    acc_el = m_el('acc')
                    accPr = m_el('accPr')
                    chr_el = m_el('chr')
                    acc_map = {'hat': '\u0302', 'bar': '\u0304', 'tilde': '\u0303'}
                    chr_el.set(f"{{{M_NS}}}val", acc_map[cmd])
                    accPr.append(chr_el)
                    acc_el.append(accPr)
                    e_el = m_el('e')
                    for c in parse_math_nodes(arg_str, italic_default=italic_default, bold_default=bold_default):
                        e_el.append(c)
                    acc_el.append(e_el)
                    base_elems = [acc_el]

                elif cmd == 'overline':
                    arg_str, pos = read_braced_or_token(s, pos)
                    bar_el = m_el('bar')
                    barPr = m_el('barPr')
                    pos_el = m_el('pos')
                    pos_el.set(f"{{{M_NS}}}val", "top")
                    barPr.append(pos_el)
                    bar_el.append(barPr)
                    e_el = m_el('e')
                    for c in parse_math_nodes(arg_str, italic_default=italic_default, bold_default=bold_default):
                        e_el.append(c)
                    bar_el.append(e_el)
                    base_elems = [bar_el]

                elif cmd == 'left':
                    while pos < n and s[pos] in ' \u2009':
                        pos += 1
                    l_delim = ''
                    if pos < n:
                        if s[pos] == '\\' and pos + 1 < n:
                            l_delim = s[pos:pos+2]
                            pos += 2
                        else:
                            l_delim = s[pos]
                            pos += 1
                    inner_str, r_delim, pos = find_matching_right(s, pos)
                    d_map = {'.': '', '⦃': '{', '⦄': '}', r'\{': '{', r'\}': '}', r'\|': '‖'}
                    b_chr = d_map.get(l_delim, l_delim)
                    e_chr = d_map.get(r_delim, r_delim)
                    d_el = m_el('d')
                    dPr = m_el('dPr')
                    begChr = m_el('begChr')
                    begChr.set(f"{{{M_NS}}}val", b_chr)
                    endChr = m_el('endChr')
                    endChr.set(f"{{{M_NS}}}val", e_chr)
                    dPr.append(begChr)
                    dPr.append(endChr)
                    d_el.append(dPr)
                    e_el = m_el('e')
                    for c in parse_math_nodes(inner_str, italic_default=italic_default, bold_default=bold_default):
                        e_el.append(c)
                    d_el.append(e_el)
                    base_elems = [d_el]

                elif cmd == 'right':
                    while pos < n and s[pos] in ' \u2009':
                        pos += 1
                    if pos < n:
                        r_ch = s[pos]
                        pos += 1
                        if r_ch != '.':
                            base_elems = [make_math_run(r_ch, italic=False, bold=bold_default)]
                        else:
                            base_elems = []
                    else:
                        base_elems = []

                elif cmd in ('mathrm', 'text', 'texttt'):
                    arg_str, pos = read_braced_or_token(s, pos)
                    arg_str = arg_str.replace('--', '–').replace('\u2009', ' ')
                    base_elems = parse_math_nodes(arg_str, italic_default=False, bold_default=bold_default)

                elif cmd == 'textbf':
                    arg_str, pos = read_braced_or_token(s, pos)
                    base_elems = parse_math_nodes(arg_str, italic_default=False, bold_default=True)

                elif cmd == 'textit':
                    arg_str, pos = read_braced_or_token(s, pos)
                    base_elems = parse_math_nodes(arg_str, italic_default=True, bold_default=bold_default)

                elif cmd == 'mathbf':
                    arg_str, pos = read_braced_or_token(s, pos)
                    base_elems = parse_math_nodes(arg_str, italic_default=False, bold_default=True)

                elif cmd == 'bm':
                    arg_str, pos = read_braced_or_token(s, pos)
                    base_elems = parse_math_nodes(arg_str, italic_default=True, bold_default=bold_default)

                elif cmd == 'mathcal':
                    arg_str, pos = read_braced_or_token(s, pos)
                    cal_txt = "".join(CALLIGRAPHIC_MAP.get(c, c) for c in arg_str.strip())
                    base_elems = [make_math_run(cal_txt, italic=False, bold=bold_default)]

                elif cmd == 'mathbb':
                    arg_str, pos = read_braced_or_token(s, pos)
                    bb_txt = "".join(BLACKBOARD_MAP.get(c, c) for c in arg_str.strip())
                    base_elems = [make_math_run(bb_txt, italic=False, bold=bold_default)]

                elif cmd == 'ensuremath':
                    arg_str, pos = read_braced_or_token(s, pos)
                    base_elems = parse_math_nodes(arg_str, italic_default=italic_default, bold_default=bold_default)

                elif cmd in GREEK_AND_SYMBOLS:
                    sym_str, is_it = GREEK_AND_SYMBOLS[cmd]
                    use_it = is_it and italic_default
                    if cmd in ('cos', 'sin', 'tan', 'exp', 'ln', 'log') and pos < n and (s[pos].isalpha() or s[pos] == '\\'):
                        sym_str = sym_str + '\u2009'
                    base_elems = [make_math_run(sym_str, italic=use_it, bold=bold_default)]

                else:
                    base_elems = [make_math_run(cmd, italic=False, bold=bold_default)]
            else:
                esc_ch = s[idx] if idx < n else ''
                pos = idx + 1
                base_elems = [make_math_run(esc_ch, italic=False, bold=bold_default)]

        # 3. Standalone ^ or _
        elif ch in ('_', '^'):
            base_elems = [make_math_run('', italic=False)]

        # 4. Digits / numbers
        elif ch.isdigit():
            start = pos
            while pos < n and (s[pos].isdigit() or s[pos] == '.'):
                pos += 1
            num_txt = s[start:pos]
            base_elems = [make_math_run(num_txt, italic=False, bold=bold_default)]

        # 5. Letters
        elif ch.isalpha():
            if not italic_default:
                start = pos
                while pos < n and (s[pos].isalpha() or s[pos] in ' ,.-–'):
                    pos += 1
                word = s[start:pos]
                base_elems = [make_math_run(word, italic=False, bold=bold_default)]
            else:
                pos += 1
                base_elems = [make_math_run(ch, italic=True, bold=bold_default)]

        # 6. Operators & punctuation
        else:
            pos += 1
            op_map = {
                '-': '−',
                '⦃': '{',
                '⦄': '}',
            }
            out_ch = op_map.get(ch, ch)
            base_elems = [make_math_run(out_ch, italic=False, bold=bold_default)]

        # Check for trailing _ and/or ^
        save_pos = pos
        while pos < n and s[pos] in ' \t\r\n':
            pos += 1

        sub_str = None
        sup_str = None
        while pos < n and s[pos] in ('_', '^'):
            kind = s[pos]
            pos += 1
            arg, pos = read_braced_or_token(s, pos)
            if kind == '_':
                sub_str = arg if sub_str is None else sub_str + " " + arg
            else:
                sup_str = arg if sup_str is None else sup_str + " " + arg
            while pos < n and s[pos] in ' \t\r\n':
                pos += 1

        if sub_str is None and sup_str is None:
            pos = save_pos

        if sub_str is not None and sup_str is not None:
            ss = m_el('sSubSup')
            e_el = m_el('e')
            for b_e in (base_elems or [make_math_run('', italic=False)]):
                e_el.append(b_e)
            sub_el = m_el('sub')
            for c in parse_math_nodes(sub_str, italic_default=italic_default, bold_default=bold_default):
                sub_el.append(c)
            sup_el = m_el('sup')
            for c in parse_math_nodes(sup_str, italic_default=italic_default, bold_default=bold_default):
                sup_el.append(c)
            ss.append(e_el)
            ss.append(sub_el)
            ss.append(sup_el)
            nodes.append(ss)
        elif sub_str is not None:
            ss = m_el('sSub')
            e_el = m_el('e')
            for b_e in (base_elems or [make_math_run('', italic=False)]):
                e_el.append(b_e)
            sub_el = m_el('sub')
            for c in parse_math_nodes(sub_str, italic_default=italic_default, bold_default=bold_default):
                sub_el.append(c)
            ss.append(e_el)
            ss.append(sub_el)
            nodes.append(ss)
        elif sup_str is not None:
            if (not base_elems or (len(base_elems) == 1 and base_elems[0].xpath('.//m:t/text()', namespaces=NSMAP) == [])) and sup_str.strip() in (r'\circ', '°'):
                nodes.append(make_math_run('°', italic=False, bold=bold_default))
            else:
                ss = m_el('sSup')
                e_el = m_el('e')
                for b_e in (base_elems or [make_math_run('', italic=False)]):
                    e_el.append(b_e)
                sup_el = m_el('sup')
                for c in parse_math_nodes(sup_str, italic_default=italic_default, bold_default=bold_default):
                    sup_el.append(c)
                ss.append(e_el)
                ss.append(sup_el)
                nodes.append(ss)
        else:
            if base_elems:
                nodes.extend(base_elems)

    return nodes


def latex_to_omml(latex_str):
    """Compile a LaTeX math string into a native Word <m:oMath> element."""
    clean = preprocess_latex_math(latex_str)
    omath = m_el('oMath')
    for node in parse_math_nodes(clean):
        omath.append(node)
    return omath


# =============================================================================
# BibTeX & Accents
# =============================================================================
def clean_latex_accents(s):
    """Normalize LaTeX accent macros into Unicode characters."""
    if not s:
        return ""
    subs = [
        (r"\{\\'e\}|\\'e|\\'\{e\}", 'é'),
        (r"\{\\`e\}|\\`e|\\`\{e\}", 'è'),
        (r"\{\\^e\}|\\^e|\\^\{e\}", 'ê'),
        (r"\{\\'a\}|\\'a|\\'\{a\}", 'á'),
        (r"\{\\`a\}|\\`a|\\`\{a\}", 'à'),
        (r"\{\\~a\}|\\~a|\\~\{a\}", 'ã'),
        (r"\{\\'o\}|\\'o|\\'\{o\}", 'ó'),
        (r"\{\\~o\}|\\~o|\\~\{o\}", 'õ'),
        (r"\{\\`o\}|\\`o|\\`\{o\}", 'ò'),
        (r"\{\\'i\}|\\'i|\\'\{i\}", 'í'),
        (r"\{\\`i\}|\\`i|\\`\{i\}", 'ì'),
        (r"\{\\'u\}|\\'u|\\'\{u\}", 'ú'),
        (r"\{\\`u\}|\\`u|\\`\{u\}", 'ù'),
        (r"\\c\{c\}|\{\\c c\}|\\c c", 'ç'),
        (r'\\"\{u\}|\{\\"\{u\}\}|\\"u', 'ü'),
        (r'\\"\{o\}|\{\\"\{o\}\}|\\"o', 'ö'),
        (r'\\"\{a\}|\{\\"\{a\}\}|\\"a', 'ä'),
    ]
    for pat, rep in subs:
        s = re.sub(pat, rep, s)
    return s


def parse_bibtex(bib_path):
    with open(bib_path, 'r', encoding='utf-8') as f:
        text = f.read()

    entries = {}
    pattern = re.compile(r'@(\w+)\s*\{\s*([^,]+),(.*?)\n\}', re.DOTALL)
    for m in pattern.finditer(text):
        entry_type = m.group(1).lower()
        key = m.group(2).strip()
        body = m.group(3)

        fields = {}
        field_pat = re.compile(r'(\w+)\s*=\s*(?:\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}|"([^"]*)"|(\d+))', re.DOTALL)
        for fm in field_pat.finditer(body):
            fname = fm.group(1).lower()
            val = fm.group(2) or fm.group(3) or fm.group(4) or ''
            val = val.replace(r'$\mu$', 'μ').replace(r'\mu', 'μ')
            val = clean_latex_accents(val.replace('{', '').replace('}', '').strip())
            val = val.replace('--', '–')
            fields[fname] = val

        authors_raw = clean_latex_accents(fields.get('author', ''))
        authors = [a.strip() for a in authors_raw.split(' and ') if a.strip()]
        short_names = []
        full_names = []
        for a in authors:
            if ',' in a:
                parts = [p.strip() for p in a.split(',')]
                last = clean_latex_accents(parts[0])
                first = clean_latex_accents(parts[1]) if len(parts) > 1 else ''
                short_names.append(last)
                full_names.append(f"{last}, {first[0]}." if first else last)
            else:
                parts = a.split()
                last = clean_latex_accents(parts[-1]) if parts else ''
                first = clean_latex_accents(parts[0]) if len(parts) > 1 else ''
                short_names.append(last)
                full_names.append(f"{last}, {first[0]}." if first else last)

        year = fields.get('year', 'n.d.')

        if len(short_names) == 1:
            cite_str = f"{short_names[0]}, {year}"
            cite_text = f"{short_names[0]} ({year})"
        elif len(short_names) == 2:
            cite_str = f"{short_names[0]} and {short_names[1]}, {year}"
            cite_text = f"{short_names[0]} and {short_names[1]} ({year})"
        elif len(short_names) > 2:
            cite_str = f"{short_names[0]} et al., {year}"
            cite_text = f"{short_names[0]} et al. ({year})"
        else:
            cite_str = f"{key}, {year}"
            cite_text = f"{key} ({year})"

        author_str = ", ".join(full_names) if full_names else "Unknown"
        title_str = fields.get('title', '').rstrip('.')
        journal_str = fields.get('journal') or fields.get('booktitle') or fields.get('institution') or fields.get('publisher') or ''
        vol_str = fields.get('volume', '')
        pages_str = fields.get('pages', '')
        doi_str = fields.get('doi', '')

        ref_parts = [f"{author_str}, {year}. {title_str}."]
        if journal_str:
            j_part = journal_str
            if vol_str:
                j_part += f" {vol_str}"
            if pages_str:
                j_part += f", {pages_str}."
            else:
                j_part += "."
            ref_parts.append(j_part)
        if doi_str:
            ref_parts.append(f"https://doi.org/{doi_str}")

        entries[key] = {
            'type': entry_type,
            'key': key,
            'fields': fields,
            'cite_paren': cite_str,
            'cite_text': cite_text,
            'first_author': short_names[0] if short_names else key,
            'year': year,
            'full_ref': " ".join(ref_parts)
        }

    return entries


# =============================================================================
# Rich Text & Inline Math Renderer
# =============================================================================
def preprocess_rich_text(text, bib_entries, cited_keys_set, label_map):
    """Resolve citations, cross-references, detached sub/superscripts, and text-mode macros."""
    if not text:
        return ""

    text = clean_latex_accents(text)

    # 1. Citations: \citep
    def replace_citep(match):
        pre = (match.group(1) or "").strip()
        post = (match.group(2) or "").strip()
        keys = [k.strip() for k in match.group(3).split(',')]
        cite_strs = []
        for k in keys:
            cited_keys_set.add(k)
            if k in bib_entries:
                cite_strs.append(bib_entries[k]['cite_paren'])
            else:
                cite_strs.append(k)
        inner = "; ".join(cite_strs)
        if pre and post:
            return f"({pre} {inner}, {post})"
        elif pre:
            return f"({pre} {inner})"
        elif post:
            return f"({inner}, {post})"
        return f"({inner})"

    text = re.sub(r'\\citep(?:\[(.*?)\])?(?:\[(.*?)\])?\{([^}]+)\}', replace_citep, text)

    # 2. Citations: \citet
    def replace_citet(match):
        post = (match.group(1) or "").strip()
        keys = [k.strip() for k in match.group(2).split(',')]
        cite_strs = []
        for k in keys:
            cited_keys_set.add(k)
            if k in bib_entries:
                cite_strs.append(bib_entries[k]['cite_text'])
            else:
                cite_strs.append(k)
        inner = ", ".join(cite_strs)
        if post:
            return f"{inner} ({post})"
        return inner

    text = re.sub(r'\\citet(?:\[(.*?)\])?\{([^}]+)\}', replace_citet, text)

    # 3. Citations: \citealt & \citealp
    def replace_citealt(match):
        keys = [k.strip() for k in match.group(1).split(',')]
        cite_strs = []
        for k in keys:
            cited_keys_set.add(k)
            if k in bib_entries:
                cite_strs.append(bib_entries[k]['cite_paren'])
            else:
                cite_strs.append(k)
        return "; ".join(cite_strs)

    text = re.sub(r'\\citeal[tp]\{([^}]+)\}', replace_citealt, text)

    # 4. Equation & General Cross-References
    def rep_eqs_range(m):
        lbl1 = m.group(1).strip()
        lbl2 = m.group(2).strip()
        n1 = label_map.get(lbl1, lbl1)
        n2 = label_map.get(lbl2, lbl2)
        return f"Eqs. ({n1})--({n2})"

    text = re.sub(r'(?:Eqs?\.~|Equations?~|Eqs?\.\s+|Equations?\s+)\\(?:eq)?ref\{([^}]+)\}--\\(?:eq)?ref\{([^}]+)\}', rep_eqs_range, text)

    def rep_eq_ref(m):
        lbl = m.group(2).strip()
        num = label_map.get(lbl, lbl)
        return f"Eq. ({num})"

    text = re.sub(r'(?:(Eq\.~|Equation~|Eq\.\s+|Equation\s+))\\(?:eq)?ref\{([^}]+)\}', rep_eq_ref, text)
    text = re.sub(r'\\eqref\{([^}]+)\}', lambda m: f"({label_map.get(m.group(1).strip(), m.group(1))})", text)

    def rep_general_ref(m):
        prefix = m.group(1).replace('~', ' ')
        lbl = m.group(2).strip()
        resolved = label_map.get(lbl, lbl)
        return f"{prefix}{resolved}"

    text = re.sub(r'(Sections?~|Tables?~|Figures?~|Figs?\.~|Append(?:ix|ices)~|Supplementary Table~|Supplementary Section~|Supplementary~)\\ref\{([^}]+)\}', rep_general_ref, text)
    text = re.sub(r'\\ref\{([^}]+)\}', lambda m: str(label_map.get(m.group(1).strip(), m.group(1))), text)

    # 5. Normalize detached superscripts/subscripts & chemical formulas
    text = re.sub(r'\$\s*\^\{?\\circ\}?\s*\$', '°', text)
    text = re.sub(r'\\textmu(?:\{\}|\s+)m\$\^\{([^}]+)\}\$', r'$\\mu\\mathrm{m}^{\1}$', text)
    text = re.sub(r'\\textmu(?:\{\}|\s*)', 'μ', text)
    text = re.sub(r'\$\\Delta\$([A-Za-z]+)\$\_\{?([^}\$]+)\}?\$(?!\$)', r'$\\Delta\\mathrm{\1}_{\2}$', text)
    text = re.sub(r'\$\\Delta\$([A-Za-z]+)', r'$\\Delta\\mathrm{\1}$', text)

    # Chemical formulas & unit powers like CaCO$_3$, Al$_2$Si$_2$O$_5$(OH)$_4$, H$_2$O, cm$^{-1}$
    def repl_chem(m):
        full = m.group(0)
        parts = re.split(r'\$([^\$]+)\$', full)
        out = []
        for idx, p in enumerate(parts):
            if idx % 2 == 0:
                if p:
                    out.append(f'\\mathrm{{{p}}}')
            else:
                out.append(p)
        return '$' + ''.join(out) + '$'

    text = re.sub(r'[A-Za-z]+(?:\([A-Za-z]+\))?(?:\$[\^_][^\$]+\$[A-Za-z]*(?:\([A-Za-z]+\))?)+', repl_chem, text)

    # 6. Text-mode custom macros outside $...$
    math_split = re.split(r'(\$[^\$]+\$)', text)
    for i in range(0, len(math_split), 2):
        seg = math_split[i]
        seg = re.sub(r'\\um\\\s*', ' μm ', seg)
        seg = re.sub(r'\\um(?![a-zA-Z])', ' μm', seg)
        seg = re.sub(r'\\gcm\\\s*', r' $\\mathrm{g\\,cm}^{-2}$ ', seg)
        seg = re.sub(r'\\gcm(?![a-zA-Z])', r' $\\mathrm{g\\,cm}^{-2}$', seg)
        seg = re.sub(r'\\NEdT\\\s*', 'NEΔT ', seg)
        seg = re.sub(r'\\NEdT(?![a-zA-Z])', 'NEΔT', seg)
        seg = re.sub(r'\\Ldown(?![a-zA-Z])', r'$L^{\\downarrow}$', seg)
        seg = re.sub(r'\\Lup(?![a-zA-Z])', r'$L^{\\uparrow}$', seg)
        seg = re.sub(r'\\allowbreak\{\}|\\allowbreak(?![a-zA-Z])', '', seg)
        seg = re.sub(r'\\url\{([^}]+)\}', r'\1', seg)
        seg = re.sub(r'\\pivalidate\{([^}]+)\}', r'[PI TO VALIDATE: \1]', seg)
        seg = re.sub(r'\\citeneeded\{([^}]+)\}', r'[FIND REFERENCE: \1]', seg)
        seg = seg.replace('{,}', ',')
        math_split[i] = seg

    text = "".join(math_split)
    return text.strip()


def clean_plain_text_segment(seg):
    """Clean LaTeX escapes in a plain-text (non-math) segment."""
    seg = seg.replace('---', '—')
    seg = seg.replace('--', '–')
    seg = seg.replace('``', '“')
    seg = seg.replace("''", '”')
    seg = seg.replace(r'\%', '%')
    seg = seg.replace(r'\&', '&')
    seg = seg.replace(r'\$', '$')
    seg = seg.replace(r'\_', '_')
    seg = seg.replace(r'\{', '{')
    seg = seg.replace(r'\}', '}')
    seg = seg.replace(r'\,', '\u2009')
    seg = seg.replace(r'\;', ' ')
    seg = seg.replace(r'\ ', ' ')
    seg = seg.replace('~', ' ')
    seg = re.sub(r'\\quad(?![a-zA-Z])', '   ', seg)
    seg = re.sub(r'\\qquad(?![a-zA-Z])', '     ', seg)
    seg = re.sub(r'\s*\n\s*', ' ', seg)
    return seg


def extract_styled_Command(s, pos):
    """If s[pos:] starts with \\textbf{, \\textit{, \\emph{, \\texttt{, \\text{, return (cmd, inner, next_pos)."""
    for cmd in ('textbf', 'textit', 'emph', 'texttt', 'text'):
        prefix = f"\\{cmd}{{"
        if s.startswith(prefix, pos):
            start = pos + len(prefix)
            depth = 1
            idx = start
            while idx < len(s) and depth > 0:
                if s[idx] == '{':
                    depth += 1
                elif s[idx] == '}':
                    depth -= 1
                idx += 1
            return cmd, s[start:idx-1], idx
    return None, None, pos


def populate_rich_paragraph(p, raw_text, bib_entries, cited_keys_set, label_map,
                            font_name='Calibri', font_size=11, bold=False, italic=False, color=None):
    """Render mixed text, styling macros, and inline $...$ OMML equations into a Word paragraph."""
    prepared = preprocess_rich_text(raw_text, bib_entries, cited_keys_set, label_map)
    if not prepared:
        return

    def emit_Plain(txt, b_flag, i_flag):
        cleaned = clean_plain_text_segment(txt)
        if not cleaned:
            return
        r = p.add_run(cleaned)
        r.font.name = font_name
        r.font.size = Pt(font_size)
        if b_flag:
            r.bold = True
        if i_flag:
            r.italic = True
        if color is not None:
            r.font.color.rgb = color

    def parse_segment(s, b_flag, i_flag):
        idx = 0
        n = len(s)
        buf = []
        while idx < n:
            if s[idx] == '$':
                if buf:
                    emit_Plain("".join(buf), b_flag, i_flag)
                    buf = []
                end_m = s.find('$', idx + 1)
                if end_m == -1:
                    buf.append(s[idx:])
                    break
                math_str = s[idx+1:end_m].strip()
                if math_str:
                    if b_flag and not math_str.startswith(r'\mathbf') and not math_str.startswith(r'\bm'):
                        math_str = f"\\mathbf{{{math_str}}}"
                    omml_el = latex_to_omml(math_str)
                    p._p.append(omml_el)
                idx = end_m + 1
            elif s[idx] == '\\':
                cmd, inner, next_idx = extract_styled_Command(s, idx)
                if cmd is not None:
                    if buf:
                        emit_Plain("".join(buf), b_flag, i_flag)
                        buf = []
                    new_b = b_flag or (cmd == 'textbf')
                    new_i = i_flag or (cmd in ('textit', 'emph'))
                    parse_segment(inner, new_b, new_i)
                    idx = next_idx
                else:
                    buf.append(s[idx])
                    idx += 1
            else:
                buf.append(s[idx])
                idx += 1
        if buf:
            emit_Plain("".join(buf), b_flag, i_flag)

    parse_segment(prepared, bold, italic)


def add_body_paragraph(doc, raw_text, bib_entries, cited_keys_set, label_map,
                       space_after=6, line_spacing=1.15, font_size=11):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0)
    p.paragraph_format.right_indent = Inches(0)
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    populate_rich_paragraph(
        p, raw_text, bib_entries, cited_keys_set, label_map,
        font_name='Calibri', font_size=font_size
    )
    return p


# =============================================================================
# Display Equation Formatter (Borderless 2-Column Table + Native <m:oMath>)
# =============================================================================
def insert_equation_block(doc, eq_latex, eq_num):
    """Insert a centered display equation (<m:oMath>) with right-aligned equation number."""
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    cell_eq = table.cell(0, 0)
    cell_num = table.cell(0, 1)

    cell_eq.width = Inches(5.75)
    cell_num.width = Inches(0.75)

    set_cell_margins(cell_eq, top=60, bottom=60, left=40, right=40)
    set_cell_margins(cell_num, top=60, bottom=60, left=40, right=40)

    p_eq = cell_eq.paragraphs[0]
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq.paragraph_format.space_before = Pt(2)
    p_eq.paragraph_format.space_after = Pt(2)
    omml_el = latex_to_omml(eq_latex)
    p_eq._p.append(omml_el)

    p_num = cell_num.paragraphs[0]
    p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_num.paragraph_format.space_before = Pt(2)
    p_num.paragraph_format.space_after = Pt(2)
    if eq_num is not None:
        r_num = p_num.add_run(f"({eq_num})")
        r_num.bold = True
        r_num.font.name = 'Calibri'
        r_num.font.size = Pt(11)

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(4)


# =============================================================================
# Publication-Grade Table Parser & Builder
# =============================================================================
def parse_multicolumn_cell(cell_str):
    """If cell_str is \\multicolumn{K}{align}{content}, return (content, K, align). Otherwise (cell_str, 1, 'l')."""
    s = cell_str.strip()
    m = re.match(r'^\\multicolumn\{(\d+)\}\s*\{((?:[^{}]|\{[^{}]*\})*)\}\s*\{', s)
    if not m:
        return s, 1, 'l'
    span = int(m.group(1))
    align_spec = m.group(2)
    start = m.end()
    depth = 1
    idx = start
    while idx < len(s) and depth > 0:
        if s[idx] == '{':
            depth += 1
        elif s[idx] == '}':
            depth -= 1
        idx += 1
    content = s[start:idx-1].strip()
    align = 'c' if 'c' in align_spec else ('r' if 'r' in align_spec else 'l')
    return content, span, align


def parse_tabular_block(table_latex):
    """Parse a LaTeX table/longtable into (caption, structured_rows, footnotes, has_two_header_rows)."""
    m = re.search(r'\\begin\{(tabular\*?|longtable)\}', table_latex)
    if not m:
        return "", [], [], False

    start = m.end()
    while start < len(table_latex) and table_latex[start] in ' \t\r\n':
        start += 1
    if start < len(table_latex) and table_latex[start] == '[':
        start = table_latex.find(']', start) + 1
        while start < len(table_latex) and table_latex[start] in ' \t\r\n':
            start += 1

    if start < len(table_latex) and table_latex[start] == '{':
        depth = 1
        idx = start + 1
        while idx < len(table_latex) and depth > 0:
            if table_latex[idx] == '{':
                depth += 1
            elif table_latex[idx] == '}':
                depth -= 1
            idx += 1
        body = table_latex[idx:]
    else:
        body = table_latex[start:]

    end_m = re.search(r'\\end\{(tabular\*?|longtable)\}', body)
    if end_m:
        body = body[:end_m.start()]

    caption = ""
    for src in (body, table_latex):
        cap_idx = src.find('\\caption{')
        if cap_idx != -1:
            cap_start = cap_idx + len('\\caption{')
            d = 1
            c_i = cap_start
            while c_i < len(src) and d > 0:
                if src[c_i] == '{':
                    d += 1
                elif src[c_i] == '}':
                    d -= 1
                c_i += 1
            caption = src[cap_start:c_i-1]
            if src is body:
                body = body[:cap_idx] + body[c_i:]
            break

    body = re.sub(r'\\label\{[^}]*\}', '', body)
    body = re.sub(r'^\s*\\\\', '', body)

    if r'\endfirsthead' in body:
        parts = body.split(r'\endfirsthead')
        header_part = parts[0]
        rest = parts[1]
        for tag in (r'\endhead', r'\endfoot', r'\endlastfoot'):
            if tag in rest:
                rest = rest.split(tag)[1]
        body = header_part + rest

    has_cmidrule = bool(re.search(r'\\cmidrule', body))

    raw_rows = re.split(r'\\\\(?:\s*\[.*?\])?', body)
    structured_rows = []
    for r in raw_rows:
        r = r.strip()
        if not r:
            continue
        r_clean = re.sub(r'\\(?:hline|toprule|midrule|bottomrule)', '', r)
        r_clean = re.sub(r'\\cmidrule(?:\([^)]*\))?\{[^}]*\}', '', r_clean).strip()
        if not r_clean:
            continue

        raw_cells = re.split(r'(?<!\\)&', r_clean)
        row_cells = []
        for c in raw_cells:
            content, span, align = parse_multicolumn_cell(c)
            row_cells.append((content, span, align))

        if any(c[0] for c in row_cells):
            structured_rows.append(row_cells)

    footnotes = []
    if len(structured_rows) > 3:
        n_cols = max(sum(c[1] for c in r) for r in structured_rows)
        while len(structured_rows) > 2:
            last_r = structured_rows[-1]
            if len(last_r) == 1 and last_r[0][1] == n_cols and not last_r[0][0].startswith(r'\textit{'):
                footnotes.insert(0, last_r[0][0])
                structured_rows.pop()
            else:
                break

    return caption, structured_rows, footnotes, has_cmidrule


def parse_table_panels(table_latex):
    """Parse one or more tabular/longtable environments inside a LaTeX table float.

    Returns (outer_caption, panels), where panels is a list of
    (panel_label, structured_rows, footnotes, has_two_header_rows).
    """
    matches = list(re.finditer(r'\\begin\{(tabular\*?|longtable)\}', table_latex))
    if not matches:
        return "", []
    if len(matches) == 1:
        cap, rows, fnotes, two_hdr = parse_tabular_block(table_latex)
        return cap, [("", rows, fnotes, two_hdr)]

    # Extract outer caption placed in the enclosing table float
    outer_caption = ""
    cap_idx = table_latex.find('\\caption{')
    if cap_idx != -1 and cap_idx < matches[0].start():
        cap_start = cap_idx + len('\\caption{')
        d = 1
        c_i = cap_start
        while c_i < len(table_latex) and d > 0:
            if table_latex[c_i] == '{':
                d += 1
            elif table_latex[c_i] == '}':
                d -= 1
            c_i += 1
        outer_caption = table_latex[cap_start:c_i - 1]

    panels = []
    prev_end = 0
    for idx_m, m in enumerate(matches):
        between = table_latex[prev_end:m.start()]
        # Capture any panel subheading such as \textbf{Panel A: ...}
        panel_lbl = ""
        lbl_m = re.findall(r'\\textbf\{([^}]+)\}', between)
        if lbl_m:
            panel_lbl = lbl_m[-1].strip()
        end_m = re.search(r'\\end\{' + re.escape(m.group(1)) + r'\}', table_latex[m.start():])
        tab_end = (m.start() + end_m.end()) if end_m else len(table_latex)
        tab_snippet = table_latex[m.start():tab_end]
        cap_i, rows_i, fnotes_i, two_hdr_i = parse_tabular_block(tab_snippet)
        if not outer_caption and cap_i:
            outer_caption = cap_i
        panels.append((panel_lbl, rows_i, fnotes_i, two_hdr_i))
        prev_end = tab_end

    return outer_caption, panels


def insert_word_table(doc, caption, structured_rows, footnotes, has_two_header_rows,
                      table_num, bib_entries, cited_keys_set, label_map):
    """Insert a publication-grade Word table with proper column spans and OMML math."""
    if caption:
        cap_p = doc.add_paragraph()
        cap_p.paragraph_format.space_before = Pt(14)
        cap_p.paragraph_format.space_after = Pt(4)
        cap_p.paragraph_format.left_indent = Inches(0)
        cap_p.paragraph_format.first_line_indent = Inches(0)
        run_lbl = cap_p.add_run(f"Table {table_num}: ")
        run_lbl.bold = True
        run_lbl.font.name = 'Calibri'
        run_lbl.font.size = Pt(10.5)
        populate_rich_paragraph(
            cap_p, caption, bib_entries, cited_keys_set, label_map,
            font_name='Calibri', font_size=10.5
        )

    if not structured_rows:
        return

    n_cols = max(sum(c[1] for c in r) for r in structured_rows)
    n_rows = len(structured_rows)
    header_row_count = 2 if has_two_header_rows else 1

    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_table_borders(table)

    font_sz = 8.5 if n_cols >= 8 else (9.0 if n_cols >= 6 else 9.5)
    hdr_font_sz = 9.0 if n_cols >= 8 else 9.5

    for r_idx, row_cells in enumerate(structured_rows):
        is_header = (r_idx < header_row_count)
        is_full_banner = (len(row_cells) == 1 and row_cells[0][1] == n_cols and not is_header)

        col_cursor = 0
        for content, span, align in row_cells:
            if col_cursor >= n_cols:
                break
            end_col = min(col_cursor + span - 1, n_cols - 1)
            cell = table.cell(r_idx, col_cursor)
            if end_col > col_cursor:
                cell.merge(table.cell(r_idx, end_col))

            set_cell_margins(cell, top=65, bottom=65, left=80, right=80)

            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.05
            p.paragraph_format.left_indent = Inches(0)
            p.paragraph_format.first_line_indent = Inches(0)

            if span > 1 and not is_full_banner and align == 'c':
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif align == 'r':
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT

            if is_header:
                set_cell_background(cell, "E6ECF5")
                populate_rich_paragraph(
                    p, content, bib_entries, cited_keys_set, label_map,
                    font_name='Calibri', font_size=hdr_font_sz, bold=True,
                    color=RGBColor(15, 34, 64)
                )
            elif is_full_banner:
                set_cell_background(cell, "F0F4F8")
                populate_rich_paragraph(
                    p, content, bib_entries, cited_keys_set, label_map,
                    font_name='Calibri', font_size=font_sz, bold=True,
                    color=RGBColor(15, 34, 64)
                )
            else:
                if r_idx % 2 == 1:
                    set_cell_background(cell, "F9FAFC")
                populate_rich_paragraph(
                    p, content, bib_entries, cited_keys_set, label_map,
                    font_name='Calibri', font_size=font_sz
                )

            col_cursor = end_col + 1

    if footnotes:
        joined_notes = " ".join(f.replace(r'\quad', '').strip() for f in footnotes)
        p_fn = doc.add_paragraph()
        p_fn.paragraph_format.space_before = Pt(3)
        p_fn.paragraph_format.space_after = Pt(8)
        p_fn.paragraph_format.left_indent = Inches(0.1)
        p_fn.paragraph_format.first_line_indent = Inches(0)
        r_nlbl = p_fn.add_run("Note: ")
        r_nlbl.bold = True
        r_nlbl.italic = True
        r_nlbl.font.name = 'Calibri'
        r_nlbl.font.size = Pt(9.5)
        populate_rich_paragraph(
            p_fn, joined_notes, bib_entries, cited_keys_set, label_map,
            font_name='Calibri', font_size=9.5, italic=True
        )
    else:
        sp = doc.add_paragraph()
        sp.paragraph_format.space_before = Pt(0)
        sp.paragraph_format.space_after = Pt(6)


# =============================================================================
# Pre-Scan Label Mapping & Dynamic Section Numbering
# =============================================================================
def isolate_structural_blocks(content):
    """Ensure headings, lists, and math environments are separated by \\n\\n so no text is ever dropped."""
    lines = [l for l in content.split('\n') if not l.strip().startswith('%')]
    content = '\n'.join(lines)

    content = re.sub(r'\{\\footnotesize\s*\\setlength\{\\tabcolsep\}\{[^}]+\}', '', content)
    content = re.sub(r'\\(?:footnotesize|small|scriptsize)(?![a-zA-Z])', '', content)
    content = re.sub(r'\\setlength\{\\tabcolsep\}\{[^}]+\}', '', content)
    content = re.sub(r'^\s*\}\s*$', '', content, flags=re.MULTILINE)

    content = re.sub(r'(\\(?:begin|end)\{(?:enumerate|itemize)\})', r'\n\n\1\n\n', content)
    content = re.sub(r'(\\(?:section|subsection|subsubsection|paragraph)\{)', r'\n\n\1', content)

    def split_after_heading(text):
        for cmd in (r'\section{', r'\subsection{', r'\subsubsection{'):
            pos = 0
            while True:
                idx = text.find(cmd, pos)
                if idx == -1:
                    break
                start = idx + len(cmd)
                d = 1
                c_i = start
                while c_i < len(text) and d > 0:
                    if text[c_i] == '{':
                        d += 1
                    elif text[c_i] == '}':
                        d -= 1
                    c_i += 1
                rest = text[c_i:]
                lbl_m = re.match(r'^\s*\\label\{[^}]+\}', rest)
                if lbl_m:
                    c_i += lbl_m.end()
                text = text[:c_i] + "\n\n" + text[c_i:]
                pos = c_i + 2
        return text

    content = split_after_heading(content)
    return content


def build_label_map(base_dir, sections_order):
    """Dynamically map all section, subsection, subsubsection, equation, table, and figure labels."""
    label_map = {}

    eq_idx = 1
    table_idx = 1
    fig_idx = 1

    sec_idx = 0
    subsec_idx = 0
    subsubsec_idx = 0

    for rel_path, sec_prefix in sections_order:
        sec_file = base_dir / rel_path
        if not sec_file.exists():
            continue
        content = sec_file.read_text(encoding='utf-8')
        for inp in re.findall(r'\\input\{([^}]+)\}', content):
            inp_path = base_dir / f"{inp}.tex"
            if inp_path.exists():
                content = content.replace(f"\\input{{{inp}}}", "\n" + inp_path.read_text(encoding='utf-8') + "\n")

        for m in re.finditer(r'\\(section|subsection|subsubsection)\{.*?\}(?:\s*\\label\{([^}]+)\})?', content, re.DOTALL):
            level = m.group(1)
            lbl = m.group(2).strip() if m.group(2) else None
            if level == 'section':
                if sec_prefix.startswith('Appendix'):
                    cur_num = sec_prefix.replace('Appendix ', '')
                else:
                    sec_idx += 1
                    cur_num = str(sec_idx)
                subsec_idx = 0
                subsubsec_idx = 0
            elif level == 'subsection':
                subsec_idx += 1
                subsubsec_idx = 0
                base_s = sec_prefix.replace('Appendix ', '') if sec_prefix.startswith('Appendix') else str(sec_idx)
                cur_num = f"{base_s}.{subsec_idx}"
            else:
                subsubsec_idx += 1
                base_s = sec_prefix.replace('Appendix ', '') if sec_prefix.startswith('Appendix') else str(sec_idx)
                cur_num = f"{base_s}.{subsec_idx}.{subsubsec_idx}"
            if lbl:
                label_map[lbl] = cur_num

        for m in re.finditer(r'\\begin\{(equation|align)\}(.*?)\\end\{\1\}', content, re.DOTALL):
            kind = m.group(1)
            body = m.group(2).strip()
            lines = re.split(r'\\\\', body) if kind == 'align' else [body]
            for l in lines:
                if not l.strip():
                    continue
                lbl_m = re.search(r'\\label\{([^}]+)\}', l)
                if lbl_m:
                    label_map[lbl_m.group(1).strip()] = str(eq_idx)
                eq_idx += 1

        for m in re.finditer(r'\\begin\{(?:table\*?|longtable)\}(.*?)\\end\{(?:table\*?|longtable)\}', content, re.DOTALL):
            lbl_m = re.search(r'\\label\{([^}]+)\}', m.group(1))
            if lbl_m:
                label_map[lbl_m.group(1).strip()] = str(table_idx)
            table_idx += 1

        for m in re.finditer(r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}', content, re.DOTALL):
            lbl_m = re.search(r'\\label\{([^}]+)\}', m.group(1))
            if lbl_m:
                label_map[lbl_m.group(1).strip()] = str(fig_idx)
            fig_idx += 1

    return label_map


# =============================================================================
# Main Document Builder
# =============================================================================
def build_manuscript_docx(output_path="OpenTIR_PINN_Manuscript_RSE_Latest.docx"):
    print(f"Starting publication-grade conversion of manuscript_rse to {output_path}...")
    base_dir = Path("manuscript_rse")
    bib_path = base_dir / "references.bib"

    bib_entries = parse_bibtex(bib_path)
    print(f"Loaded {len(bib_entries)} references from references.bib.")

    sections_order = [
        ("sections/01_introduction.tex", "1"),
        ("sections/02_background.tex", "2"),
        ("sections/03_dataset.tex", "3"),
        ("sections/04_methods.tex", "4"),
        ("sections/05_experimental_design.tex", "5"),
        ("sections/06_results.tex", "6"),
        ("sections/07_discussion.tex", "7"),
        ("sections/08_conclusions.tex", "8"),
        ("sections/appA_modtran.tex", "Appendix A"),
        ("sections/appB_validation.tex", "Appendix B"),
        ("sections/appC_tes_sensitivity.tex", "Appendix C"),
    ]

    label_map = build_label_map(base_dir, sections_order)
    print(f"Mapped {len(label_map)} labels for cross-references.")

    cited_keys_set = set()
    doc = docx.Document()

    for sec in doc.sections:
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)

    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(25, 25, 25)

    # 1. Frontmatter
    main_tex = (base_dir / "main.tex").read_text(encoding='utf-8')

    title_m = re.search(r'\\title\{([^}]+)\}', main_tex)
    title_raw = title_m.group(1).strip() if title_m else "OpenTIR-PINN Manuscript"

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(10)
    p_title.paragraph_format.left_indent = Inches(0)
    populate_rich_paragraph(
        p_title, title_raw, bib_entries, cited_keys_set, label_map,
        font_name='Calibri', font_size=16.5, bold=True, color=RGBColor(15, 34, 64)
    )

    authors_text = "Atilio Grondona¹⁴* · Fernanda Zunkowski Grondona¹ · Bijeesh Kozhikkodan Veettil² · Luis Morales-Salinas³"
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_after = Pt(6)
    p_auth.paragraph_format.left_indent = Inches(0)
    r_auth = p_auth.add_run(authors_text)
    r_auth.bold = True
    r_auth.font.size = Pt(11)

    affil_text = (
        "¹ AgroRisco AI\n"
        "² Department of Environmental Technology and Management, Faculty of Applied Technology, Van Lang University, Ho Chi Minh City, Vietnam\n"
        "³ Laboratory for Research in Environmental Sciences (LARES), Faculty of Agricultural Sciences, University of Chile, Santiago, Chile\n"
        "⁴ Universidade Federal do Rio Grande do Sul (UFRGS), Porto Alegre, Brazil\n"
        "* Corresponding author: atilio.grondona@ufrgs.br"
    )
    p_aff = doc.add_paragraph()
    p_aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_aff.paragraph_format.space_after = Pt(12)
    p_aff.paragraph_format.left_indent = Inches(0)
    r_aff = p_aff.add_run(affil_text)
    r_aff.font.size = Pt(9.5)
    r_aff.font.color.rgb = RGBColor(85, 85, 85)

    p_j = doc.add_paragraph()
    p_j.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_j.paragraph_format.space_after = Pt(16)
    p_j.paragraph_format.left_indent = Inches(0)
    r_j = p_j.add_run("Target Journal: Remote Sensing of Environment (Elsevier) | Complete Draft for Review & Annotations")
    r_j.italic = True
    r_j.font.size = Pt(10)
    r_j.font.color.rgb = RGBColor(0, 102, 204)

    # Abstract
    abs_m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', main_tex, re.DOTALL)
    if abs_m:
        p_absh = doc.add_paragraph()
        p_absh.paragraph_format.space_before = Pt(10)
        p_absh.paragraph_format.space_after = Pt(4)
        p_absh.paragraph_format.left_indent = Inches(0)
        r_absh = p_absh.add_run("Abstract")
        r_absh.bold = True
        r_absh.font.size = Pt(13)
        r_absh.font.color.rgb = RGBColor(15, 34, 64)

        p_abs = doc.add_paragraph()
        p_abs.paragraph_format.left_indent = Inches(0.2)
        p_abs.paragraph_format.right_indent = Inches(0.2)
        p_abs.paragraph_format.first_line_indent = Inches(0)
        p_abs.paragraph_format.space_after = Pt(10)
        p_abs.paragraph_format.line_spacing = 1.15
        p_abs.alignment = WD_ALIGN_PARAGRAPH.LEFT
        populate_rich_paragraph(
            p_abs, abs_m.group(1).strip(), bib_entries, cited_keys_set, label_map,
            font_name='Calibri', font_size=10.5
        )

    # Keywords
    key_m = re.search(r'\\begin\{keyword\}(.*?)\\end\{keyword\}', main_tex, re.DOTALL)
    if key_m:
        keys_clean = key_m.group(1).replace(r'\sep', ' · ').strip()
        p_key = doc.add_paragraph()
        p_key.paragraph_format.left_indent = Inches(0.2)
        p_key.paragraph_format.first_line_indent = Inches(0)
        p_key.paragraph_format.space_after = Pt(16)
        rk_lbl = p_key.add_run("Keywords: ")
        rk_lbl.bold = True
        rk_lbl.font.size = Pt(10.5)
        rk_val = p_key.add_run(keys_clean)
        rk_val.italic = True
        rk_val.font.size = Pt(10.5)

    # 2. Sequential Sections Parsing
    table_counter = 1
    fig_counter = 1
    eq_counter = 1

    sec_idx = 0
    subsec_idx = 0
    subsubsec_idx = 0

    for rel_path, sec_prefix in sections_order:
        sec_file = base_dir / rel_path
        if not sec_file.exists():
            continue

        print(f"Processing {rel_path}...")
        content = sec_file.read_text(encoding='utf-8')

        for inp in re.findall(r'\\input\{([^}]+)\}', content):
            inp_path = base_dir / f"{inp}.tex"
            if inp_path.exists():
                content = content.replace(f"\\input{{{inp}}}", "\n" + inp_path.read_text(encoding='utf-8') + "\n")

        placeholders = {}

        def table_sub(m):
            nonlocal table_counter
            tbl_latex = m.group(0)
            tag = f"___TABLE_PLACEHOLDER_{table_counter}___"
            cap, panels = parse_table_panels(tbl_latex)
            if len(panels) > 1:
                placeholders[tag] = ('multi_table', cap, panels, table_counter)
            elif panels:
                _, rows, fnotes, two_hdr = panels[0]
                placeholders[tag] = ('table', cap, rows, fnotes, two_hdr, table_counter)
            else:
                cap, rows, fnotes, two_hdr = parse_tabular_block(tbl_latex)
                placeholders[tag] = ('table', cap, rows, fnotes, two_hdr, table_counter)
            table_counter += 1
            return f"\n\n{tag}\n\n"

        content = re.sub(r'\\begin\{(?:table\*?|longtable)\}.*?\\end\{(?:table\*?|longtable)\}', table_sub, content, flags=re.DOTALL)

        def fig_sub(m):
            nonlocal fig_counter
            fig_latex = m.group(0)
            tag = f"___FIG_PLACEHOLDER_{fig_counter}___"
            cap = ""
            c_idx = fig_latex.find('\\caption{')
            if c_idx != -1:
                c_start = c_idx + len('\\caption{')
                d = 1
                i_c = c_start
                while i_c < len(fig_latex) and d > 0:
                    if fig_latex[i_c] == '{':
                        d += 1
                    elif fig_latex[i_c] == '}':
                        d -= 1
                    i_c += 1
                cap = fig_latex[c_start:i_c-1]
            img_m = re.search(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}', fig_latex)
            img = img_m.group(1) if img_m else ""
            placeholders[tag] = ('figure', cap, img, fig_counter)
            fig_counter += 1
            return f"\n\n{tag}\n\n"

        content = re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', fig_sub, content, flags=re.DOTALL)

        def math_env_sub(m):
            nonlocal eq_counter
            kind = m.group(1)
            body = m.group(2).strip()
            is_starred = kind.endswith('*')
            lines = re.split(r'\\\\', body) if kind.startswith('align') else [body]
            tags = []
            for l in lines:
                clean_eq = re.sub(r'\\label\{[^}]+\}', '', l).replace('&=', '=').strip()
                if not clean_eq:
                    continue
                tag = f"___EQ_PLACEHOLDER_{len(placeholders)}___"
                if is_starred:
                    placeholders[tag] = ('equation', clean_eq, None)
                else:
                    placeholders[tag] = ('equation', clean_eq, eq_counter)
                    eq_counter += 1
                tags.append(tag)
            return "\n\n" + "\n\n".join(tags) + "\n\n"

        content = re.sub(r'\\begin\{(equation\*?|align\*?)\}(.*?)\\end\{\1\}', math_env_sub, content, flags=re.DOTALL)

        content = isolate_structural_blocks(content)

        list_counter = 1
        def list_sub(m):
            nonlocal list_counter
            kind = m.group(1)
            body = m.group(2)
            tag = f"___LIST_PLACEHOLDER_{list_counter}___"
            list_counter += 1
            placeholders[tag] = ('list', kind == 'enumerate', body)
            return f"\n\n{tag}\n\n"

        content = re.sub(r'\\begin\{(enumerate|itemize)\}(.*?)\\end\{\1\}', list_sub, content, flags=re.DOTALL)

        content = re.sub(r'\\label\{[^}]+\}', '', content)

        blocks = content.split('\n\n')
        for block in blocks:
            b = block.strip()
            if not b:
                continue

            if b in placeholders:
                ptype = placeholders[b][0]
                if ptype == 'table':
                    _, cap, rows, fnotes, two_hdr, t_num = placeholders[b]
                    insert_word_table(doc, cap, rows, fnotes, two_hdr, t_num, bib_entries, cited_keys_set, label_map)
                elif ptype == 'multi_table':
                    _, cap, panels, t_num = placeholders[b]
                    if cap:
                        cap_p = doc.add_paragraph()
                        cap_p.paragraph_format.space_before = Pt(14)
                        cap_p.paragraph_format.space_after = Pt(4)
                        cap_p.paragraph_format.left_indent = Inches(0)
                        cap_p.paragraph_format.first_line_indent = Inches(0)
                        run_lbl = cap_p.add_run(f"Table {t_num}: ")
                        run_lbl.bold = True
                        run_lbl.font.name = 'Calibri'
                        run_lbl.font.size = Pt(10.5)
                        populate_rich_paragraph(
                            cap_p, cap, bib_entries, cited_keys_set, label_map,
                            font_name='Calibri', font_size=10.5
                        )
                    for p_lbl, rows, fnotes, two_hdr in panels:
                        if p_lbl:
                            p_sub = doc.add_paragraph()
                            p_sub.paragraph_format.space_before = Pt(6)
                            p_sub.paragraph_format.space_after = Pt(2)
                            p_sub.paragraph_format.left_indent = Inches(0)
                            populate_rich_paragraph(
                                p_sub, f"\\textbf{{{p_lbl}}}", bib_entries, cited_keys_set, label_map,
                                font_name='Calibri', font_size=9.5
                            )
                        insert_word_table(doc, "", rows, fnotes, two_hdr, t_num, bib_entries, cited_keys_set, label_map)
                elif ptype == 'figure':
                    _, cap, img_name, f_num = placeholders[b]
                    if img_name:
                        img_path = base_dir / "figures" / img_name
                        if img_path.is_file():
                            try:
                                doc.add_picture(str(img_path), width=Inches(6.0))
                                last_p = doc.paragraphs[-1]
                                last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                                last_p.paragraph_format.space_before = Pt(8)
                            except Exception as e:
                                print(f"Could not insert image {img_name}: {e}")
                    p_fcap = doc.add_paragraph()
                    p_fcap.paragraph_format.space_before = Pt(4)
                    p_fcap.paragraph_format.space_after = Pt(12)
                    p_fcap.paragraph_format.left_indent = Inches(0)
                    r_flbl = p_fcap.add_run(f"Figure {f_num}: ")
                    r_flbl.bold = True
                    r_flbl.font.name = 'Calibri'
                    r_flbl.font.size = Pt(10)
                    populate_rich_paragraph(
                        p_fcap, cap, bib_entries, cited_keys_set, label_map,
                        font_name='Calibri', font_size=10
                    )
                elif ptype == 'equation':
                    _, eq_str, num = placeholders[b]
                    insert_equation_block(doc, eq_str, num)
                elif ptype == 'list':
                    _, is_num, list_body = placeholders[b]
                    items = re.findall(r'\\item\s*(.*?)(?=(?:\\item|$))', list_body, re.DOTALL)
                    for item_idx, itm in enumerate(items):
                        itm_str = itm.strip()
                        if not itm_str:
                            continue
                        itm_str = re.sub(r'^\\textit\{([^}]+:)\}', r'\\textbf{\1}', itm_str)
                        prefix = f"{item_idx + 1}. " if is_num else "• "
                        p_list = doc.add_paragraph()
                        p_list.paragraph_format.left_indent = Inches(0.50)
                        p_list.paragraph_format.first_line_indent = Inches(-0.25)
                        p_list.paragraph_format.space_before = Pt(2)
                        p_list.paragraph_format.space_after = Pt(4)
                        p_list.paragraph_format.line_spacing = 1.15
                        p_list.alignment = WD_ALIGN_PARAGRAPH.LEFT
                        r_pre = p_list.add_run(prefix)
                        r_pre.bold = True
                        r_pre.font.name = 'Calibri'
                        r_pre.font.size = Pt(11)
                        populate_rich_paragraph(
                            p_list, itm_str, bib_entries, cited_keys_set, label_map,
                            font_name='Calibri', font_size=11
                        )
                continue

            if b.startswith(r'\section{'):
                start = len(r'\section{')
                d = 1
                c_i = start
                while c_i < len(b) and d > 0:
                    if b[c_i] == '{': d += 1
                    elif b[c_i] == '}': d -= 1
                    c_i += 1
                sec_raw = b[start:c_i-1]
                if sec_prefix.startswith('Appendix'):
                    num_prefix = f"{sec_prefix}. "
                else:
                    sec_idx += 1
                    num_prefix = f"{sec_idx}. "
                subsec_idx = 0
                subsubsec_idx = 0

                p_h1 = doc.add_paragraph()
                p_h1.paragraph_format.space_before = Pt(18)
                p_h1.paragraph_format.space_after = Pt(6)
                p_h1.paragraph_format.left_indent = Inches(0)
                p_h1.paragraph_format.first_line_indent = Inches(0)
                r_num = p_h1.add_run(num_prefix)
                r_num.bold = True
                r_num.font.size = Pt(14)
                r_num.font.color.rgb = RGBColor(15, 34, 64)
                populate_rich_paragraph(
                    p_h1, sec_raw, bib_entries, cited_keys_set, label_map,
                    font_name='Calibri', font_size=14, bold=True, color=RGBColor(15, 34, 64)
                )
                continue

            if b.startswith(r'\subsection{'):
                start = len(r'\subsection{')
                d = 1
                c_i = start
                while c_i < len(b) and d > 0:
                    if b[c_i] == '{': d += 1
                    elif b[c_i] == '}': d -= 1
                    c_i += 1
                ssec_raw = b[start:c_i-1]
                subsec_idx += 1
                subsubsec_idx = 0
                base_s = sec_prefix.replace('Appendix ', '') if sec_prefix.startswith('Appendix') else str(sec_idx)
                num_prefix = f"{base_s}.{subsec_idx}. "

                p_h2 = doc.add_paragraph()
                p_h2.paragraph_format.space_before = Pt(14)
                p_h2.paragraph_format.space_after = Pt(4)
                p_h2.paragraph_format.left_indent = Inches(0)
                p_h2.paragraph_format.first_line_indent = Inches(0)
                r_num = p_h2.add_run(num_prefix)
                r_num.bold = True
                r_num.font.size = Pt(12)
                r_num.font.color.rgb = RGBColor(30, 60, 100)
                populate_rich_paragraph(
                    p_h2, ssec_raw, bib_entries, cited_keys_set, label_map,
                    font_name='Calibri', font_size=12, bold=True, color=RGBColor(30, 60, 100)
                )
                continue

            if b.startswith(r'\subsubsection{'):
                start = len(r'\subsubsection{')
                d = 1
                c_i = start
                while c_i < len(b) and d > 0:
                    if b[c_i] == '{': d += 1
                    elif b[c_i] == '}': d -= 1
                    c_i += 1
                sssec_raw = b[start:c_i-1]
                subsubsec_idx += 1
                base_s = sec_prefix.replace('Appendix ', '') if sec_prefix.startswith('Appendix') else str(sec_idx)
                num_prefix = f"{base_s}.{subsec_idx}.{subsubsec_idx}. "

                p_h3 = doc.add_paragraph()
                p_h3.paragraph_format.space_before = Pt(10)
                p_h3.paragraph_format.space_after = Pt(3)
                p_h3.paragraph_format.left_indent = Inches(0)
                p_h3.paragraph_format.first_line_indent = Inches(0)
                r_num = p_h3.add_run(num_prefix)
                r_num.bold = True
                r_num.font.size = Pt(11)
                r_num.font.color.rgb = RGBColor(40, 75, 115)
                populate_rich_paragraph(
                    p_h3, sssec_raw, bib_entries, cited_keys_set, label_map,
                    font_name='Calibri', font_size=11, bold=True, color=RGBColor(40, 75, 115)
                )
                continue

            if b.startswith(r'\paragraph{'):
                start = len(r'\paragraph{')
                d = 1
                c_i = start
                while c_i < len(b) and d > 0:
                    if b[c_i] == '{': d += 1
                    elif b[c_i] == '}': d -= 1
                    c_i += 1
                par_raw = b[start:c_i-1].rstrip('.')
                rest = b[c_i:].strip()
                p_par = doc.add_paragraph()
                p_par.paragraph_format.space_before = Pt(6)
                p_par.paragraph_format.space_after = Pt(6)
                p_par.paragraph_format.left_indent = Inches(0)
                p_par.paragraph_format.first_line_indent = Inches(0)
                p_par.paragraph_format.line_spacing = 1.15
                combined_par = f"\\textbf{{\\textit{{{par_raw}.}}}} {rest}" if rest else f"\\textbf{{\\textit{{{par_raw}.}}}}"
                populate_rich_paragraph(
                    p_par, combined_par, bib_entries, cited_keys_set, label_map,
                    font_name='Calibri', font_size=11
                )
                continue

            add_body_paragraph(doc, b, bib_entries, cited_keys_set, label_map, space_after=6, line_spacing=1.15)

    # 3. Data and Code Availability
    p_dch = doc.add_paragraph()
    p_dch.paragraph_format.space_before = Pt(18)
    p_dch.paragraph_format.space_after = Pt(6)
    p_dch.paragraph_format.left_indent = Inches(0)
    r_dch = p_dch.add_run("Data and Code Availability")
    r_dch.bold = True
    r_dch.font.size = Pt(14)
    r_dch.font.color.rgb = RGBColor(15, 34, 64)

    dc_text = (
        r"To support community benchmarking, independent algorithmic validation, and further methodological development for thermal infrared missions, "
        r"the complete OpenTIR-PINN v1.3.0 benchmark dataset, MODTRAN~5.2 radiative transfer look-up tables, "
        r"5-fold cross-validation split manifests, golden target spectra, pretrained model checkpoints, and reproduction scripts "
        r"are publicly archived on Zenodo at \url{https://doi.org/10.5281/zenodo.24085242} "
        r"(\url{https://zenodo.org/records/24085242}; concept DOI for all versions: \url{https://doi.org/10.5281/zenodo.23932830}) \citep{grondona2026opentir}."
    )
    add_body_paragraph(doc, dc_text, bib_entries, cited_keys_set, label_map, space_after=12)

    # 4. References Section
    p_refh = doc.add_paragraph()
    p_refh.paragraph_format.space_before = Pt(20)
    p_refh.paragraph_format.space_after = Pt(8)
    p_refh.paragraph_format.left_indent = Inches(0)
    r_refh = p_refh.add_run("References")
    r_refh.bold = True
    r_refh.font.size = Pt(14)
    r_refh.font.color.rgb = RGBColor(15, 34, 64)

    sorted_cited_keys = sorted(
        [k for k in cited_keys_set if k in bib_entries],
        key=lambda k: (bib_entries[k]['first_author'].lower(), bib_entries[k]['year'])
    )

    print(f"Writing {len(sorted_cited_keys)} cited references into docx...")
    for k in sorted_cited_keys:
        ref_entry = bib_entries[k]
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.35)
        p_ref.paragraph_format.first_line_indent = Inches(-0.35)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.line_spacing = 1.05
        r = p_ref.add_run(ref_entry['full_ref'])
        r.font.name = 'Calibri'
        r.font.size = Pt(9.5)

    doc.save(output_path)
    print(f"Successfully generated: {output_path}")
    print(f"File size: {os.path.getsize(output_path):,} bytes")


if __name__ == "__main__":
    build_manuscript_docx("OpenTIR_PINN_Manuscript_RSE_Latest.docx")
