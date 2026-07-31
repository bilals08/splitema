import re

_TOK_RE = re.compile(r'\[[^\]]+\]-|[A-Z]\[[^\]]+\]|[A-Z]')

_BRACKET_MOD_MAP: dict[str, str] = {
    "[Carbamidomethyl]": "c",
    "[Oxidation]":       "o",
    "[Deamidated]":      "d",
    "[Acetyl]-":         "a",
    "[Carbamyl]-":       "k",
    "[Ammonia-loss]-":   "q",
    "[+25.980265]-":     "",
}


# Convert bracketed peptide modifications into compact lowercase modification codes.
def bracket_to_lowerletter(seq: str) -> str:
    tokens = _TOK_RE.findall(seq)
    result = []
    for tok in tokens:
        if tok.startswith("["):
            result.append(_BRACKET_MOD_MAP.get(tok, ""))
        elif "[" in tok:
            aa     = tok[0]
            bracket = tok[1:]
            result.append(aa + _BRACKET_MOD_MAP.get(bracket, ""))
        else:
            result.append(tok)
    return "".join(result)


# Return the peptide token length, or 0 when the sequence is malformed.
def peptide_length(seq: str) -> int:
    tokens = _TOK_RE.findall(seq)
    if not tokens or "".join(tokens) != seq:
        return 0
    return len(tokens)
