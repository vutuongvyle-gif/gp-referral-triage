"""NHI number validation per HISO 10046 (both formats) and test-number generation."""
import random
import re
import string

ALPHA = [c for c in string.ascii_uppercase if c not in "IO"]   # A=1 ... Z=24
OLD = re.compile(r"[A-HJ-NP-Z]{3}\d{4}")
NEW = re.compile(r"[A-HJ-NP-Z]{3}\d{2}[A-HJ-NP-Z]{2}")


def _value(c: str) -> int:
    return ALPHA.index(c) + 1 if c.isalpha() else int(c)


def _weighted_sum(first_six: str) -> int:
    return sum(_value(c) * (7 - i) for i, c in enumerate(first_six))


def is_valid_nhi(nhi: str) -> bool:
    n = nhi.strip().upper()
    if OLD.fullmatch(n):
        m = _weighted_sum(n[:6]) % 11
        if m == 0:
            return False
        check = 11 - m
        return (0 if check == 10 else check) == int(n[6])
    if NEW.fullmatch(n):
        m = _weighted_sum(n[:6]) % 23
        return ALPHA[(23 - m) - 1] == n[6]
    return False


def make_test_nhi(rng: random.Random = random) -> str:
    """A valid new-format NHI in the Z (test) range, for synthetic patients only."""
    body = ("Z" + "".join(rng.choice(ALPHA) for _ in range(2))
            + f"{rng.randint(0, 99):02d}" + rng.choice(ALPHA))
    m = _weighted_sum(body) % 23
    return body + ALPHA[(23 - m) - 1]