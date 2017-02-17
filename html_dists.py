import gzip
import shutil

from stanza.research import config


def output_html_dists():
    config.options(read=True)
    with gzip.open(config.get_file_path('dists.b64.gz'), 'r') as infile:
        rows = list(infile)
    with config.open('dists.js', 'w') as outfile:
        write_json_dists(rows, outfile)

    with config.open('data.eval.jsons', 'r') as infile:
        insts = list(infile)
    with config.open('insts.js', 'w') as outfile:
        write_json_insts(insts, outfile)
    shutil.copy('dists.html', config.get_file_path('dists.html'))


def write_json_dists(rows, outfile):
    outfile.write('ROWS = [\n')
    for row in rows:
        outfile.write(' "{}",'.format(row.strip()))
    outfile.write(']\n')


def write_json_insts(insts, outfile):
    outfile.write('INSTS = [\n')
    for row in insts:
        outfile.write(' {},'.format(row.strip()))
    outfile.write(']\n')


if __name__ == '__main__':
    output_html_dists()
