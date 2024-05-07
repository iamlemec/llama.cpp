# generate NFD tables from Unicode data

# UnicodeData.txt from the Unicode Character Database:
# https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt

# adapted from (MIT license):
# https://github.com/n1t0/unicode-normalization/blob/master/scripts/unicode.py

# entries should be pairs
def format_cpp_table(entries, max_width=120):
    full = ''
    curr = ''

    for a, b in entries:
        pair = f'{{0x{a:08X}, 0x{b:08X}}}, '

        if len(curr) + len(pair) > max_width:
            full += f'{curr[:-1]}\n'
            curr = ''

        curr += pair

    if len(curr) > 0:
        full += f'{curr[:-1]}'

    return full

def load_unicode_decomp(path):
    with open(path) as fid:
        data = fid.read()

    canon_decomp = {}

    for line in data.splitlines():
        pieces = line.split(';')
        assert len(pieces) == 15

        char, decomp = pieces[0], pieces[5]
        char_int = int(char, 16)

        if decomp == '' or decomp.startswith('<'):
            continue

        canon_decomp[char_int] = [int(c, 16) for c in decomp.split()]

    return canon_decomp

def compute_fully_decomposed(canon_decomp):
    # Constants from Unicode 9.0.0 Section 3.12 Conjoining Jamo Behavior
    # http://www.unicode.org/versions/Unicode9.0.0/ch03.pdf#M9.32468.Heading.310.Combining.Jamo.Behavior
    S_BASE, L_COUNT, V_COUNT, T_COUNT = 0xAC00, 19, 21, 28
    S_COUNT = L_COUNT * V_COUNT * T_COUNT

    def _decompose(char_int):
        # 7-bit ASCII never decomposes
        if char_int <= 0x7f:
            yield char_int
            return

        # Assert that we're handling Hangul separately.
        assert not (S_BASE <= char_int < S_BASE + S_COUNT)

        decomp = canon_decomp.get(char_int)
        if decomp is not None:
            for decomposed_ch in decomp:
                for fully_decomposed_ch in _decompose(decomposed_ch):
                    yield fully_decomposed_ch
            return

        yield char_int

    canon_fully_decomp = {}

    for char_int in canon_decomp:
        # Always skip Hangul, since it's more efficient to represent its
        # decomposition programmatically.
        if S_BASE <= char_int < S_BASE + S_COUNT:
            continue

        canon = list(_decompose(char_int))
        if not (len(canon) == 1 and canon[0] == char_int):
            canon_fully_decomp[char_int] = canon

    return canon_fully_decomp

def gen_nfd_table(path):
    canon_decomp = load_unicode_decomp(path)
    canon_fully_decomp = compute_fully_decomposed(canon_decomp)
    canon_fully_pairs = [
        (k, v) for k, vs in canon_fully_decomp.items() for v in vs
    ]
    return format_cpp_table(canon_fully_pairs)

if __name__ == '__main__':
    print('const std::multimap<uint32_t, uint32_t> unicode_map_nfd = {')
    print(gen_nfd_table('UnicodeData.txt'))
    print('};')
