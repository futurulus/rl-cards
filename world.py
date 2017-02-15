import copy
import numpy as np

from cards import cards
from cards_env import MAX_BOARD_SIZE


class CardsWorld(object):
    def __init__(self, transcript, verbosity=0, apply_actions=False, event_limit=None):
        self.verbosity = verbosity
        self.walls = [[1.0 for c in range(MAX_BOARD_SIZE[1])]
                      for r in range(MAX_BOARD_SIZE[0])]

        self.p1_loc = (0, 0)
        self.p2_loc = (0, 0)
        self.cards_to_loc = {}

        self.load_transcript(transcript, apply_actions=apply_actions, event_limit=event_limit)

    def load_transcript(self, transcript, apply_actions, event_limit):
        for i, event in enumerate(transcript.iter_events()):
            if event_limit is not None and i >= event_limit:
                break
            self.apply_event(event)

    def apply_event(self, event):
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
            return True
        elif event.action in (cards.INITIAL_LOCATION, cards.MOVE):
            loc = event.parse_contents()
            if event.agent == cards.PLAYER1:
                self.p1_loc = loc
            else:
                self.p2_loc = loc
            return True
        elif event.action == cards.PICKUP:
            card = event.parse_contents()[2]
            if event.agent == cards.PLAYER1:
                self.cards_to_loc[card] = cards.PLAYER1
            else:
                self.cards_to_loc[card] = cards.PLAYER2
            return True
        elif event.action == cards.DROP:
            lx, ly, card = event.parse_contents()
            self.cards_to_loc[card] = (lx, ly)
            return True
        else:
            return False

    def swap_players(self):
        swapped = copy.deepcopy(self)
        swapped.p1_loc, swapped.p2_loc = swapped.p2_loc, swapped.p1_loc
        swap = lambda loc: (cards.PLAYER2 if loc == cards.PLAYER1 else
                            cards.PLAYER1 if loc == cards.PLAYER2 else loc)
        swapped.cards_to_loc = {
            card: swap(loc)
            for card, loc in swapped.cards_to_loc.iteritems()
        }
        return swapped


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


def event_world_pairs(transcript, verbosity=0):
    pairs = []
    eoi = end_of_init(transcript)
    world = CardsWorld(transcript, event_limit=eoi, verbosity=verbosity)

    event = transcript.events[eoi - 1]
    pairs.append((event, copy.deepcopy(world)))
    for event in transcript.events[eoi:]:
        world.apply_event(event)
        pairs.append((event, copy.deepcopy(world)))
    return pairs


def end_of_init(transcript):
    '''
    Return 1 + the index of the last "initialization" event in `transcript`, where
    "initialization" includes ENVIRONMENT and INITIAL_LOCATION events. This is the
    lowest appropriate value of `event_limit` to make sure the world is properly
    initialized.
    '''
    init_inds = [i for i, e in enumerate(transcript.events)
                 if e.action in (cards.ENVIRONMENT, cards.INITIAL_LOCATION)]
    if init_inds:
        return 1 + max(init_inds)
    else:
        return 0


def build_cards_obs(world, los=None):
    '''
    >>> w = build_world(walls=np.zeros(MAX_BOARD_SIZE),
    ...                 p1_loc=(1, 1),
    ...                 cards_to_loc={'AS': (2, 1), 'KD': (9, 9)})
    >>> obs = build_cards_obs(w, los=3)
    >>> obs[0, 0]  # no card here
    0.0
    >>> obs[2, 1]  # yep, card visible here
    1.0
    >>> obs[7, 8]  # unknown, not in line of sight
    -1.0
    >>> obs[9, 9]  # unknown, not in line of sight
    -1.0
    '''
    obs = -np.ones(MAX_BOARD_SIZE)

    los_start_row = los_end_row = los_start_col = los_end_col = None
    if los is not None:
        p1r, p1c = world.p1_loc
        los_start_row, los_end_row, los_start_col, los_end_col = np.clip(
            [p1r - los, p1r + los + 1, p1c - los, p1c + los + 1], 0,
            [MAX_BOARD_SIZE[0] + 1, MAX_BOARD_SIZE[0] + 1,
             MAX_BOARD_SIZE[1] + 1, MAX_BOARD_SIZE[1] + 1])

    obs[los_start_row:los_end_row, los_start_col:los_end_col] = 0.0

    for _, loc in world.cards_to_loc.iteritems():
        if loc is None or isinstance(loc, basestring):
            continue
        if obs[loc] == 0.0:
            obs[loc] = 1.0

    return obs
