from cards import cards

PATH = 'cards/transcripts'

_transcripts = None


def all_transcripts():
    global _transcripts
    if _transcripts is None:
        corpus = cards.Corpus('cards/transcripts')
        _transcripts = list(corpus.iter_transcripts())
    return _transcripts
