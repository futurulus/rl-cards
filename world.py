from cards import cards
from cards_env import MAX_BOARD_SIZE


class CardsWorld(object):
    def __init__(self, transcript, verbosity=0):
        self.verbosity = verbosity
        self.walls = [[1.0 for c in range(MAX_BOARD_SIZE[1])]
                      for r in range(MAX_BOARD_SIZE[0])]

        self.p1_loc = (0, 0)
        self.p2_loc = (0, 0)
        self.cards_to_loc = {}

        self.load_transcript(transcript)

    def load_transcript(self, transcript):
        for event in transcript.iter_events():
            if event.action == cards.ENVIRONMENT:
                board_info, card_info = event.contents.split('NEW_SECTION')
                rows = board_info.split(';')[:-1]
                for r, row in enumerate(rows):
                    if self.verbosity >= 4:
                        print(row)
                    for c, col in enumerate(row):
                        self.walls[r][c] = 1. if col == '-' else -1. if col == 'b' else 0.
                card_locs = card_info.split(';')[:-1]
                for card_loc in card_locs:
                    loc_str, card_str = card_loc.split(':')
                    loc = tuple(int(x) for x in loc_str.split(','))
                    assert len(loc) == 2, loc
                    self.cards_to_loc[card_str] = loc
                    if self.verbosity >= 4:
                        print('%s: %s' % (loc, card_str))
            elif event.action == cards.INITIAL_LOCATION:
                if event.agent == cards.PLAYER1:
                    self.p1_loc = event.parse_contents()
                else:
                    self.p2_loc = event.parse_contents()
            else:
                continue


class MockTranscript(object):
    def iter_events(self):
        return []


def build_world(walls, cards_to_loc, p1_loc=(1, 1), p2_loc=(1, 1), verbosity=0):
    world = CardsWorld(MockTranscript())
    for i, row in enumerate(walls):
        for j, val in enumerate(row):
            world.walls[i][j] = val
    world.cards_to_loc = cards_to_loc
    world.p1_loc = p1_loc
    world.p2_loc = p2_loc
    return world
