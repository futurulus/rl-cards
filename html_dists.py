import gzip
import numpy as np
from scipy.misc import logsumexp
import shutil

from stanza.research import config


def output_html_dists():
    config.options(read=True)
    with gzip.open(config.get_file_path('dists.b64.gz'), 'r') as infile:
        rows = list(infile)
    with config.open('dists.js', 'w') as outfile:
        write_json_dists(rows, outfile)
        write_json_ents(rows, outfile)

    with config.open('data.eval.jsons', 'r') as infile:
        insts = list(infile)
    with config.open('insts.js', 'w') as outfile:
        write_json_insts(insts, outfile)
    shutil.copy('dists.html', config.get_file_path('dists.html'))


def write_json_dists(rows, outfile):
    outfile.write('ROWS = [\n')
    for row in rows:
        outfile.write(' "{}",\n'.format(row.strip()))
    outfile.write(']\n')


def write_json_ents(rows, outfile):
    ents = [entropies(row.strip()) for row in rows]
    outfile.write('ENTS = [\n')
    for card_ents in ents:
        max_ent = max(card_ents)
        min_ent = min(card_ents)
        outfile.write(' [')
        for ent in card_ents:
            red = int((max_ent - ent) / (max_ent - min_ent) * 7.0 + 1.0)
            red = max(0, min(red, 7))
            outfile.write('{},'.format(red))
        outfile.write('],\n')
    outfile.write(']\n')


def entropies(row):
    ents = []
    NUM_LOCS = 26 * 34
    for suit in range(4):
        for rank in range(13):
            start = (rank * 4 + suit) * (NUM_LOCS + 2) / 2
            ents.append(entropy(row[start:start + (NUM_LOCS + 2) / 2]))
    ents.append(entropy(row[-NUM_LOCS / 2:]))
    return ents


def entropy(card_row):
    logps = []
    for char in card_row:
        red1, red2 = char_base64(char)
        if red1:
            logps.append(1.0 - 2.0 * red1)
        if red2:
            logps.append(1.0 - 2.0 * red2)
    normalized = np.array(logps) - logsumexp(logps)
    total = (-normalized * np.exp(normalized)).sum()
    return total


def write_json_insts(insts, outfile):
    outfile.write('INSTS = [\n')
    for row in insts:
        outfile.write(' {},\n'.format(row.strip()))
    outfile.write(']\n')


def base64_char(n1, n2):
    '''
    0 through 9, A through Z, a through z, '.', '/'

    >>> base64_char(7, 1)
    'v'
    >>> base64_char(1, 7)
    'F'
    >>> base64_char(0, 7)
    '7'
    '''
    assert 0 <= n1 < 8 and 0 <= n2 < 8, (n1, n2)
    combined = n1 * 8 + n2
    if combined < 10:
        return str(combined)
    elif combined < 36:
        return chr(ord('A') + (combined - 10))
    elif combined < 62:
        return chr(ord('a') + (combined - 36))
    else:
        return chr(ord('.') + (combined - 62))


def char_base64(c):
    '''
    0 through 9, A through Z, a through z, '.', '/'

    >>> char_base64('v')
    (7, 1)
    >>> char_base64('F')
    (1, 7)
    >>> char_base64('7')
    (0, 7)
    '''
    code = ord(c)
    if ord('0') <= code <= ord('9'):
        combined = code - ord('0')
    elif ord('A') <= code <= ord('Z'):
        combined = code - ord('A') + 10
    elif ord('a') <= code <= ord('z'):
        combined = code - ord('a') + 36
    else:
        combined = code - ord('.') + 62
    return (combined // 8), (combined % 8)


if __name__ == '__main__':
    output_html_dists()
