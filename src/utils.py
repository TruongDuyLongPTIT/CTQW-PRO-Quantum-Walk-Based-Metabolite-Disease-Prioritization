"""
utils.py — Helper functions dùng chung toàn bộ pipeline.
"""
import re


def standardize_hmdb_id(hid: str) -> str:
    """Chuẩn hóa HMDB ID về dạng HMDB0000001 (7 digits)."""
    if not hid or not isinstance(hid, str):
        return hid
    hid = hid.strip().upper()
    if not hid.startswith('HMDB'):
        return hid
    digits = hid[4:]
    return ('HMDB' + digits.zfill(7)) if digits.isdigit() else hid


def normalize_name(s: str) -> str:
    """Lowercase, alphanumeric only, underscore separator."""
    if not s:
        return ''
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return re.sub(r'_+', '_', s).strip('_')


def normalize_chem_aggressive(s: str) -> str:
    """Normalize chemical name: strip stereochemistry prefixes."""
    if not s:
        return ''
    s = s.lower().strip()
    s = re.sub(r'^\([\(\[rs+\-,\s]+[\)\]][\-]?', '', s)
    s = re.sub(r'^(dl|l|d|alpha|beta|gamma|cis|trans)\-', '', s)
    s = re.sub(r'\s*\([^)]*\)\s*', ' ', s)
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return re.sub(r'_+', '_', s).strip('_')


def short_inchikey(ik: str) -> str:
    """Lấy phần đầu của InChIKey (connectivity layer)."""
    if not ik or not isinstance(ik, str):
        return ''
    parts = ik.split('-')
    return parts[0] if parts else ''

