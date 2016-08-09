from collections import defaultdict
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import pyglet
from gym.envs.classic_control import rendering

from cards_cache import all_transcripts


ACTIONS = ['nop',
           'right', 'up', 'left', 'down',
           'pick',
           'drop0', 'drop1', 'drop2',
           'speak']
SUITS = list('SCHD')
RANKS = list('A23456789') + ['10'] + list('JQK')
MAX_BOARD_SIZE = (26, 34)


class CardsEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self):
        self.viewer = None

        self.verbosity = 4

        # One action for each player
        player = spaces.Discrete(len(ACTIONS))
        self.action_space = player  # should this be spaces.Tuple((player, player)) for 2 players?
        # One board for walls, one for card observations, one for player location
        board = spaces.Box(np.zeros(MAX_BOARD_SIZE), np.ones(MAX_BOARD_SIZE))
        # TODO: represent language in observations
        language = spaces.Box(np.array(0.), np.array(1.))
        hand = spaces.Box(np.zeros((3, len(RANKS), len(SUITS))),
                          np.ones((3, len(RANKS), len(SUITS))))
        floor = spaces.Box(np.zeros((len(RANKS), len(SUITS))),
                           np.ones((len(RANKS), len(SUITS))))
        self.observation_space = spaces.Tuple((board, board, board, hand, floor, language))

        self.clear_board()
        import world
        self.default_world = world.CardsWorld(all_transcripts()[0])

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, u):
        action = ACTIONS[u]

        if action == 'nop':
            pass
        elif action == 'right':
            self._move(0, 1)
        elif action == 'up':
            self._move(-1, 0)
        elif action == 'left':
            self._move(0, -1)
        elif action == 'down':
            self._move(1, 0)
        elif action == 'pick':
            self._pick()
        elif action == 'drop0':
            self._drop(0)
        elif action == 'drop1':
            self._drop(1)
        elif action == 'drop2':
            self._drop(2)
        elif action == 'speak':
            self._speak()
        else:
            raise RuntimeError('invalid action name: %s' % (action,))

        done = self._is_done()
        #      obs, reward, done, info
        return self._get_obs(), (0.0 if done else -1.0), done, {}

    def _move(self, dr, dc):
        p1r, p1c = self.p1_loc
        new_loc = (p1r + dr, p1c + dc)
        if not self._is_in_bounds(new_loc):
            if self.verbosity >= 4:
                print('move: OOB')
        elif self.walls[new_loc]:
            if self.verbosity >= 4:
                print('move: WALL')
        else:
            self.p1_loc = new_loc
            if self.verbosity >= 4:
                print('move: %s' % (self.p1_loc,))

    def _is_in_bounds(self, loc):
        r, c = loc
        return (0 <= r < self.walls.shape[0]) and (0 <= c < self.walls.shape[1])

    def _pick(self):
        if self.verbosity >= 4:
            print('pick')
        pass  # TODO

    def _drop(self, slot):
        if self.verbosity >= 4:
            print('drop %s' % (slot,))
        pass  # TODO

    def _speak(self):
        if self.verbosity >= 4:
            print('speak')
        pass  # TODO

    def _is_done(self):
        done = (0, 0) in self.loc_to_cards[self.p1_loc]
        if done and self.verbosity >= 2:
            print('FOUND THE ACE OF SPADES!')
        return done

    def _reset(self):
        return self._configure(self.default_world, verbosity=0)

    def clear_board(self):
        self.walls = np.ones(MAX_BOARD_SIZE)
        self.p1_loc = (0, 0)
        self.p2_loc = (0, 0)
        self.loc_to_cards = defaultdict(list)
        self.cards_to_loc = defaultdict(lambda: None)

    def _configure(self, world, verbosity=None):
        if verbosity is not None:
            self.verbosity = verbosity
        self.clear_board()

        self.walls = np.copy(world.walls)
        for card, loc in world.cards_to_loc.iteritems():
            card = (RANKS.index(card[:-1]), SUITS.index(card[-1]))
            self.cards_to_loc[card] = loc
            self.loc_to_cards[loc].append(card)

        self.p1_loc = world.p1_loc
        self.p2_loc = world.p2_loc

        if self.viewer and self.viewer.geoms:
            self.viewer.geoms = []

        return self._get_obs()

    def _get_obs(self):
        # Invisible walls (walls = -1.0) are not observed (but still prevent movement)
        wall_obs = np.maximum(self.walls, 0.0)
        player_obs = np.zeros(self.walls.shape)
        player_obs[self.p1_loc] = 1.0
        language = np.array(0.)  # TODO
        return (wall_obs, self._card_obs(), player_obs,
                self._hand_obs(), self._floor_obs(), language)

    def _card_obs(self):
        obs = np.zeros(self.walls.shape)
        for (r, c), cards_here in self.loc_to_cards.iteritems():
            if cards_here:
                obs[r, c] = 1.
        return obs

    def _floor_obs(self):
        cards_here = self.loc_to_cards[self.p1_loc]
        obs = np.zeros((len(RANKS), len(SUITS)))
        for rank, suit in cards_here:
            obs[rank, suit] = 1.
        return obs

    def _hand_obs(self):
        return np.zeros((3, len(RANKS), len(SUITS)))  # TODO

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0, MAX_BOARD_SIZE[1], 0, MAX_BOARD_SIZE[0] + 2)

        if not self.viewer.geoms:
            for r in range(self.walls.shape[0]):
                for c in range(self.walls.shape[1]):
                    if (r, c) in self.loc_to_cards:
                        special = (0, 0) in self.loc_to_cards[(r, c)]
                        card = make_card(r, c, special=special)
                        self.viewer.add_geom(card)

                    if self.walls[r, c]:
                        wall = make_wall(r, c, invisible=self.walls[r, c] < 0.0)
                        self.viewer.add_geom(wall)

            player = make_player()
            self.player_transform = rendering.Transform()
            player.add_attr(self.player_transform)
            self.viewer.add_geom(player)

            self.floor_hud = CardHUD()
            self.floor_hud.add_attr(rendering.Transform(translation=(1.0, MAX_BOARD_SIZE[0] + 1.0)))
            self.viewer.add_geom(self.floor_hud)
            # fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            # self.img = rendering.Image(fname, 1., 1.)
            # self.imgtrans = rendering.Transform()
            # self.img.add_attr(self.imgtrans)

        pr, pc = self.p1_loc
        self.player_transform.set_translation(pc, MAX_BOARD_SIZE[0] - pr)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))


def make_rect(row, col, cr, cg, cb):
    l, r, t, b = 0, 1, 0, -1
    wall = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
    wall.set_color(cr, cg, cb)
    wall_transform = rendering.Transform()
    wall_transform.set_translation(col, MAX_BOARD_SIZE[0] - row)
    wall.add_attr(wall_transform)
    return wall


def make_wall(row, col, invisible=False):
    return make_rect(row, col, 0., float(invisible), float(invisible))


def make_card(row, col, special=False):
    return make_rect(row, col, 1., 0. if special else 0.75, 0.)


def make_player():
    l, c, r, t, m, b = 0, 0.5, 1, 0, -0.5, -1
    player = rendering.make_polygon([(l, m), (c, t), (r, m), (c, b)])
    player.set_color(0.0, 0.0, 1.0)
    return player


class CardHUD(rendering.Geom):
    def __init__(self):
        self.card = ('A', 'S')
        self._load_imgs()
        self.rank_img = rendering.Image('images/A.png', 1., 1.)
        self.suit_img = rendering.Image('images/S.png', 1., 1.)
        suit_transform = rendering.Transform(translation=(1., 0.))
        self.suit_img.add_attr(suit_transform)

        super(CardHUD, self).__init__()

    def _load_imgs(self):
        self.imgs = {}
        for ident in RANKS + SUITS:
            filename = path.join(path.dirname(__file__), 'images/%s.png' % ident)
            self.imgs[ident] = pyglet.image.load(filename)

    def render1(self):
        self.card = None
        if True:  # np.random.choice([True, False]):
            rank = RANKS[np.random.choice(range(len(RANKS)))]
            suit = SUITS[np.random.choice(range(len(SUITS)))]
            self.card = (rank, suit)

        if self.card:
            rank, suit = self.card
            self.rank_img.img = self.imgs[rank]
            self.suit_img.img = self.imgs[suit]

            self.rank_img.render()
            self.suit_img.render()


_registered = False


def register():
    global _registered
    name = 'Cards-v0'
    if not _registered:
        from gym.envs.registration import register as gym_register
        gym_register(
            id=name,
            entry_point='cards_env:CardsEnv',
            nondeterministic=True,
        )
        _registered = True
    return name
