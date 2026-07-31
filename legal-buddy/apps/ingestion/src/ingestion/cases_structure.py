"""Header parser: extracted judgment text -> one structured case record.

The corpus has a consistent (if noisy) header: court, division, parenthesised
jurisdiction, ``Present:`` judges, a ``<Type> No. N of YYYY`` case reference,
parties around a *Versus* delimiter, advocates, hearing/judgment dates, then the
body after a ``<JUDGE>, J:`` anchor. None of that is perfectly uniform, so every
field is anchor/regex-extracted defensively and cross-checked against the
filename, which independently encodes ``<id>_<type>_<no>_<year>_<status>``.

This is the case-law analogue of an ``data/acts/*.json`` record. Pure functions
over ``(stem, text)`` so they validate on a sample without touching PDFs.
"""
import re

# ---- filename -----------------------------------------------------------------

# Trailing status/disposition tokens some filenames carry after the year.
_STATUS_TOKENS = (
    "summarily_rejected", "disposed_of", "disposed", "allowed", "dismissed",
    "rejected", "rs_fix", "abated", "discharged",
)
_NUM_YEAR_RE = re.compile(r"(\d+)\D{0,4}((?:19|20)\d{2})")
_MULTIPART_RE = re.compile(r"_(\d)$")

# High-confidence expansions for the short filename type tokens, used only when
# the header has no case reference of its own. Conservative on purpose: anything
# not listed is kept as the cleaned raw token rather than guessed.
_TYPE_ABBREV = {
    "cr": "Civil Revision", "c r": "Civil Revision", "civil revision": "Civil Revision",
    "crl": "Criminal", "crl rev": "Criminal Revision", "criminal rev": "Criminal Revision",
    "crl appl": "Criminal Appeal", "crl appeal": "Criminal Appeal",
    "crl misc": "Criminal Miscellaneous", "criminal misc": "Criminal Miscellaneous",
    "fa": "First Appeal", "f a": "First Appeal", "first appeal": "First Appeal",
    "fma": "First Miscellaneous Appeal", "first misc appeal": "First Miscellaneous Appeal",
    "wp": "Writ Petition", "writ petition": "Writ Petition", "writ": "Writ Petition",
    "deathref": "Death Reference", "deref": "Death Reference",
    "death reference": "Death Reference", "tn": "Tenancy", "so": "Second Appeal",
}


def normalize_case_type(raw: str | None) -> str | None:
    """Clean a filename type token into a citable type, expanding safe abbreviations."""
    if not raw:
        return None
    tok = raw.replace("_", " ")
    if " " not in tok and re.search(r"[a-z][A-Z]", tok):  # camelCase -> words
        tok = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", tok)
    tok = re.sub(r"(?i)\bno\.?$", "", tok)            # trailing "No"
    tok = re.sub(r"\s+", " ", tok).strip(" .-")
    key = re.sub(r"[.\s]+", " ", tok).strip().lower()
    return _TYPE_ABBREV.get(key, tok) or None


def parse_filename(stem: str) -> dict:
    """Pull a unique ``case_id`` + (type, no, year, status, part) from the name.

    The full stem is the id: the leading numeric prefix is NOT unique -- 2.5k
    files share the placeholder prefix ``1`` (e.g. ``1_TN_18218_2024_R2W``), so
    keying on it silently collapses thousands of distinct judgments into one.
    The prefix is kept as ``seq`` (informational only). The last
    ``<digits><sep><year>`` pair is the case number/year; the alphabetic run
    before it is the type token.
    """
    out = {"case_id": stem, "seq": None, "fn_type": None, "fn_no": None,
           "fn_year": None, "fn_status": None, "fn_part": None}

    m = re.match(r"^(\d+)[_\s]+(.*)$", stem)
    if not m:
        return out
    out["seq"], rest = m.group(1), m.group(2)

    part = _MULTIPART_RE.search(rest)
    if part:
        out["fn_part"] = int(part.group(1))
        rest = rest[: part.start()]

    low = rest.lower()
    for tok in _STATUS_TOKENS:
        idx = low.find(tok)
        if idx != -1:
            out["fn_status"] = tok
            rest = rest[:idx].rstrip("_. ")
            break

    pairs = list(_NUM_YEAR_RE.finditer(rest))
    if pairs:
        last = pairs[-1]
        out["fn_no"] = last.group(1)
        out["fn_year"] = int(last.group(2))
        type_tok = rest[: last.start()].strip(" _.-")
        type_tok = re.sub(r"(?i)\bno\b\.?", "", type_tok).strip(" _.-")
        out["fn_type"] = re.sub(r"[_]+", " ", type_tok).strip() or None
    return out


# ---- header field extraction --------------------------------------------------

_BODY_ANCHOR_RE = re.compile(
    r"(?m)^[\s>~|*#\d]*([A-Z][A-Za-z.\-'\s]{3,60}?),?\s+J\.?\s*[:.]?\s*$"
)
_COURT_RE = re.compile(r"(?i)(supreme\s+court\s+of\s+bangladesh)")
_DIVISION_RE = re.compile(r"(?i)\b((?:high\s+court|appellate)\s+division)\b")
_JURISDICTION_RE = re.compile(r"(?i)\(([^()]*?jurisdiction)\)")
_JUDGE_RE = re.compile(r"(?i)\bjustice\s+([^\n,]+?)\s*$", re.MULTILINE)
# Type is the run of words right before "No. N of YYYY"; space-only (no newline)
# so it can't bleed across lines into the judge block, and unanchored so it still
# fires on "In the matter of: Writ Petition No. 1298 of 2020".
_CASE_REF_RE = re.compile(
    r"(?i)([A-Za-z][A-Za-z. \t]{2,45}?)\s*No\.?\s*(\d+)\s*of\s*(\d{4})\b"
)
_VERSUS_RE = re.compile(r"[\-\s=.]*\b(?:versus|vs)\b\.?[\-\s=.]*", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b")
_JUDGMENT_DATE_RE = re.compile(
    r"(?i)judg?ment\s+(?:on|delivered\s+on|dated|pronounced\s+on)\s*:?\s*"
    r"([0-9]{1,2}[./][0-9]{1,2}[./][0-9]{2,4})"
)
_HEARD_RE = re.compile(r"(?i)heard\s+(?:and\s+judgment\s+)?on\s*:?\s*([^\n]+)")
_DISTRICT_RE = re.compile(r"(?i)\bdistrict[\s:\-]+([A-Z][A-Za-z]+)")
# Advocate lines are name lines: a title (Mr/Mrs/Ms/Dr) on the same line as an
# Advocate/Counsel/law-officer role. Requiring the title keeps prose like
# "the learned Advocate was ill" out.
_ADVOCATE_LINE_RE = re.compile(
    r"(?im)^[\s.\-]*((?:Mr|Mrs|Ms|Dr|Mr\.|Mrs\.|Ms\.)\b.*?\b"
    r"(?:Advocate|Counsel|D\.A\.G|A\.A\.G|Attorney)\b.*)$"
)


def split_body(text: str) -> tuple[str, str]:
    """Return (header_region, body). Body starts at the first ``<Judge>, J:`` line."""
    m = _BODY_ANCHOR_RE.search(text)
    if m:
        return text[: m.start()].strip(), text[m.start():].strip()
    # No anchor (some orders): treat the first ~60 non-empty lines as header.
    lines = [ln for ln in text.splitlines()]
    return "\n".join(lines[:60]).strip(), text.strip()


def _first(rx: re.Pattern, text: str, group: int = 1) -> str | None:
    m = rx.search(text)
    return m.group(group).strip() if m else None


def _judges(header: str) -> list[str]:
    seen, out = set(), []
    for m in _JUDGE_RE.finditer(header):
        name = re.sub(r"(?i)^(mr|mrs|ms|madam)\.?\s+", "", m.group(1)).strip(" .,-")
        # Drop "Justice" used as a common noun ("...the Hon'ble Chief Justice...").
        if not name or len(name) < 3 or name.lower().startswith("of "):
            continue
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out


def _case_ref(header: str, fn_no: str | None, fn_year: int | None) -> dict:
    """Pick the header ``Type No. N of YYYY`` that corroborates the filename.

    The filename independently encodes the court's catalog ``<type>_<no>_<year>``,
    so it is the authoritative id. A header match is trusted ONLY when its number
    matches the filename's: on Bengali/OCR'd judgments the English regex otherwise
    scrapes garbage or a *cited* lower-court reference (e.g. a "Dinajpur ... Appeal
    No. 72 of 1999" that the judgment merely discusses), which is worse than the
    clean filename. When the filename has a number but no header match corroborates
    it, return empty so the caller synthesizes the citation from the filename. Only
    when the filename lacks a number do we fall back to the first header match.
    """
    none = {"case_type": None, "case_no": None, "case_year": None, "full_case_ref": None}
    matches = []
    for m in _CASE_REF_RE.finditer(header):
        ctype = re.sub(r"\s+", " ", m.group(1)).strip(" .-")
        if not ctype or ctype.lower() in {"no", "in", "the"}:
            continue
        matches.append((ctype, m.group(2), int(m.group(3))))
    if not matches:
        return none

    if fn_no is not None:
        # Number is the discriminator: a header ref whose NUMBER matches the
        # filename is this case (a year off-by-one is catalog-vs-judgment noise, so
        # keep the header's year). A number that doesn't match is a cited/garbage
        # ref -> reject and let the caller synthesize from the filename.
        for cand in matches:
            if cand[1] == fn_no:
                ctype, no, year = cand
                return {"case_type": ctype, "case_no": no, "case_year": year,
                        "full_case_ref": f"{ctype} No. {no} of {year}"}
        return none

    ctype, no, year = matches[0]
    return {"case_type": ctype, "case_no": no, "case_year": year,
            "full_case_ref": f"{ctype} No. {no} of {year}"}


def _parties(header: str) -> tuple[str | None, str | None]:
    m = _VERSUS_RE.search(header)
    if not m:
        return None, None
    before, after = header[: m.start()], header[m.end():]
    # Parties sit just around the delimiter; keep the nearest few non-empty lines.
    pet = [ln.strip() for ln in before.splitlines() if ln.strip()][-6:]
    resp = [ln.strip() for ln in after.splitlines() if ln.strip()][:6]
    return ("\n".join(pet) or None), ("\n".join(resp) or None)


def _dates(header: str) -> tuple[list[str], str | None]:
    heard = []
    hm = _HEARD_RE.search(header)
    if hm:
        heard = _DATE_RE.findall(hm.group(1))
    jd = _first(_JUDGMENT_DATE_RE, header)
    if not jd:
        # Fall back to the last date in the header region.
        all_dates = _DATE_RE.findall(header)
        jd = all_dates[-1] if all_dates else None
    return heard, jd


def _disposition(fn_status: str | None, body: str) -> str | None:
    if fn_status:
        return fn_status.replace("_", " ")
    tail = body[-1500:].lower()
    for cue, label in (
        ("made absolute", "rule made absolute"),
        ("discharged", "discharged"),
        ("dismissed", "dismissed"),
        ("disposed of", "disposed of"),
        ("is allowed", "allowed"),
        ("rejected", "rejected"),
        ("set aside", "set aside"),
    ):
        if cue in tail:
            return label
    return None


def _advocates(header: str) -> list[str]:
    out = []
    for m in _ADVOCATE_LINE_RE.finditer(header):
        line = re.sub(r"\s+", " ", m.group(0)).strip(" .-")
        if line and line not in out:
            out.append(line)
    return out[:12]


def parse_case(stem: str, text: str, header_text: str | None = None) -> dict:
    """Build a structured case record from a filename stem and extracted text.

    Header fields (judges, parties, advocates, case ref) are parsed with
    line-anchored regexes, so they need line-structured text. ``text`` may be
    reflowed into paragraphs for the stored body, which would collapse those
    anchors -- pass the original line-structured text as ``header_text`` so the
    metadata is parsed from it while ``body_text`` keeps the reflowed prose.
    """
    fn = parse_filename(stem)
    src = header_text if header_text is not None else text
    # split_body gives the header *region* for field parsing; body_text itself is
    # kept lossless (full text) so retrieval never silently drops content.
    header, _ = split_body(src)
    # Court/division/jurisdiction sit at the very top, before any body anchor;
    # read them from the document head so an early anchor can't hide them.
    head = src[:2500]
    ref = _case_ref(header, fn["fn_no"], fn["fn_year"])
    pet, resp = _parties(header)
    heard, judgment_date = _dates(header)

    # Header reference is authoritative; otherwise synthesize a citation from the
    # filename, which independently encodes type/number/year.
    case_type = ref["case_type"] or normalize_case_type(fn["fn_type"])
    case_no = ref["case_no"] or fn["fn_no"]
    case_year = ref["case_year"] or fn["fn_year"]
    full_ref = ref["full_case_ref"]
    if not full_ref and case_type and case_no and case_year:
        full_ref = f"{case_type} No. {case_no} of {case_year}"

    return {
        "case_id": fn["case_id"],
        "seq": fn["seq"],
        "source_file": f"{stem}.pdf",
        "court": _first(_COURT_RE, head),
        "division": _first(_DIVISION_RE, head),
        "jurisdiction": _first(_JURISDICTION_RE, head),
        "district": _first(_DISTRICT_RE, header),
        "judges": _judges(header),
        "case_type": case_type,
        "case_no": case_no,
        "case_year": case_year,
        "full_case_ref": full_ref,
        "ref_source": "header" if ref["full_case_ref"] else "filename",
        "fn_type": fn["fn_type"],
        "fn_part": fn["fn_part"],
        "petitioners_raw": pet,
        "respondents_raw": resp,
        "advocates": _advocates(header),
        "heard_dates": heard,
        "judgment_date": judgment_date,
        "disposition": _disposition(fn["fn_status"], text),
        "body_text": text,
    }
