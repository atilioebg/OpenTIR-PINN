"""Repair UTF-8 text that was mis-decoded as cp1252 and re-saved (double encoding), and strip BOMs."""
import sys

def to_bytes(s):
    out = bytearray()
    for ch in s:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            o = ord(ch)
            if 0x80 <= o <= 0xFF:
                out.append(o)          # undefined cp1252 slots were mapped 1:1
            else:
                out += ch.encode("utf-8")
    return bytes(out)

for f in sys.argv[1:]:
    b = open(f, "rb").read()
    if b[:3] == b"\xef\xbb\xbf":
        b = b[3:]
    s = b.decode("utf-8")
    try:
        fixed = to_bytes(s).decode("utf-8")
    except UnicodeDecodeError:
        fixed = s  # not double-encoded
    open(f, "w", encoding="utf-8", newline="").write(fixed)
    print(f, "repaired" if fixed != s else "unchanged (BOM stripped)")

