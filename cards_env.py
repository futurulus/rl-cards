from collections import defaultdict
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import pyglet
try:
    from gym.envs.classic_control import rendering
    Geom = rendering.Geom
except pyglet.canvas.xlib.NoSuchDisplayException:
    Geom = object

from stanza.research import config

from cards_cache import all_transcripts
import cards_config
from helpers import profile


ACTIONS = ['nop',
           'right', 'up', 'left', 'down',
           'pick',
           'drop0', 'drop1', 'drop2',
           'speak']
SUITS = list('SCHD')
RANKS = list('A23456789') + ['10'] + list('JQK')
MAX_BOARD_SIZE = (26, 34)
MAX_BATCH_SIZE = 10


class CardsEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self):
        options = config.options()
        self.game_config = cards_config.new(options.game_config)

        self.viewer = None

        self.verbosity = 4

        # One action for each player
        player = spaces.Discrete(len(ACTIONS))
        # should this be spaces.Tuple((player, player)) for 2 players?
        self.action_space = spaces.Tuple([player for _ in range(MAX_BATCH_SIZE)])
        # One board for walls, one for card observations, one for player location
        board = spaces.Box(np.zeros(MAX_BOARD_SIZE), np.ones(MAX_BOARD_SIZE))
        language_player = spaces.Box(np.array(0.), np.array(1.))
        language = spaces.Tuple([language_player for _ in range(self.game_config.num_players - 1)])
        hand = spaces.Box(np.zeros((3, len(RANKS), len(SUITS))),
                          np.ones((3, len(RANKS), len(SUITS))))
        floor = spaces.Box(np.zeros((len(RANKS), len(SUITS))),
                           np.ones((len(RANKS), len(SUITS))))
        all_obs = (board, board, board, hand, floor, language)
        self.observation_space = spaces.Tuple([e
                                               for _ in range(MAX_BATCH_SIZE)
                                               for e in all_obs])

        self.clear_boards()
        import world
        self.default_world = world.CardsWorld(all_transcripts()[0])

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @profile
    def _step(self, u):
        actions = [ACTIONS[a] for a in u]
        rewards = []
        done = []
        for w, action in enumerate(actions):
            old_value = self.game_config.value(self, w)

            if action == 'nop':
                pass
            elif action == 'right':
                self._move(w, 0, 1)
            elif action == 'up':
                self._move(w, -1, 0)
            elif action == 'left':
                self._move(w, 0, -1)
            elif action == 'down':
                self._move(w, 1, 0)
            elif action == 'pick':
                self._pick(w)
            elif action == 'drop0':
                self._drop(w, 0)
            elif action == 'drop1':
                self._drop(w, 1)
            elif action == 'drop2':
                self._drop(w, 2)
            elif action == 'speak':
                self._speak(w)
            else:
                raise RuntimeError('invalid action name: %s' % (action,))

            done_w = self._is_done(w)
            done.append(done_w)
            if done_w:
                rewards.append(0.0)
            else:
                new_value = self.game_config.value(self, w)
                action_reward = self.game_config.action_reward(self, w, action)
                rewards.append(new_value - old_value + action_reward)
        #      obs, reward, done, info
        return self._get_obs(), rewards, done, {}

    def _move(self, w, dr, dc):
        p1r, p1c = self.p1_loc[w]
        new_loc = (p1r + dr, p1c + dc)
        if not self._is_in_bounds(new_loc):
            if self.verbosity >= 9:
                print('move: OOB')
        elif self.walls[w][new_loc]:
            if self.verbosity >= 9:
                print('move: WALL')
        else:
            self.p1_loc[w] = new_loc
            if self.verbosity >= 9:
                print('move: %s' % (self.p1_loc[w],))

    def _is_in_bounds(self, loc):
        r, c = loc
        return (0 <= r < self.walls.shape[1]) and (0 <= c < self.walls.shape[2])

    def _pick(self, w):
        if self.verbosity >= 9:
            print('pick')
        pass  # TODO

    def _drop(self, w, slot):
        if self.verbosity >= 9:
            print('drop %s' % (slot,))
        pass  # TODO

    def _speak(self, w):
        if self.verbosity >= 9:
            print('speak')
        pass  # TODO

    def _is_done(self, w):
        if self.done[w]:
            return True
        elif (0, 0) in self.loc_to_cards[w][self.p1_loc[w]]:
            self.done[w] = True
            if self.verbosity >= 9:
                print('FOUND THE ACE OF SPADES!')
            return True
        else:
            return False

    def _reset(self):
        return self._configure(verbosity=0)

    def clear_boards(self):
        self.done = np.zeros((MAX_BATCH_SIZE,), dtype=np.bool)
        self.walls = np.ones((MAX_BATCH_SIZE,) + MAX_BOARD_SIZE)
        self.p1_loc = [(0, 0) for _ in range(MAX_BATCH_SIZE)]
        self.p2_loc = [(0, 0) for _ in range(MAX_BATCH_SIZE)]
        self.loc_to_cards = [defaultdict(list) for _ in range(MAX_BATCH_SIZE)]
        self.cards_to_loc = [defaultdict(lambda: None) for _ in range(MAX_BATCH_SIZE)]

    def _configure(self, worlds=None, verbosity=None):
        if verbosity is not None:
            self.verbosity = verbosity

        if worlds is None:
            worlds = [self.default_world for _ in range(MAX_BATCH_SIZE)]

        self.clear_boards()

        for w, world in enumerate(worlds):
            self.walls[w, :, :] = world.walls
            for card, loc in world.cards_to_loc.iteritems():
                card = (RANKS.index(card[:-1]), SUITS.index(card[-1]))
                self.cards_to_loc[w][card] = loc
                self.loc_to_cards[w][loc].append(card)

            self.p1_loc[w] = world.p1_loc
            self.p2_loc[w] = world.p2_loc

        if self.viewer and self.viewer.geoms:
            self.viewer.geoms = []

        return self._get_obs()

    @profile
    def _get_obs(self):
        all_obs = []
        self.game_config.update_language_obs(self)
        for w in range(MAX_BATCH_SIZE):
            # Invisible walls (walls = -1.0) are not observed (but still prevent movement)
            wall_obs = np.maximum(self.walls[w], 0.0)
            player_obs = np.zeros(self.walls.shape[1:])
            player_obs[self.p1_loc[w]] = 1.0
            language_producers = np.array([(u is not None)
                                           for u in self.game_config.get_language_obs(self, w)],
                                          dtype=np.int)
            if language_producers.sum() >= 2.0:
                language_observers = np.ones(language_producers.shape, dtype=np.float32)
            elif language_producers.any():
                language_observers = 1.0 - language_producers.astype(np.float32)
            else:
                language_observers = np.zeros(language_producers.shape, dtype=np.float32)
            all_obs.extend([wall_obs, self._card_obs(w), player_obs,
                            self._hand_obs(w), self._floor_obs(w), language_observers])
        return all_obs

    def get_language_obs(self, w, requesting_player):
        return [u for i, u in enumerate(self.game_config.get_language_obs(self, w))
                if i != requesting_player]

    @profile
    def _card_obs(self, w):
        obs = np.zeros(self.walls.shape[1:])
        for loc, cards_here in self.loc_to_cards[w].iteritems():
            if cards_here and not isinstance(loc, basestring):
                r, c = loc
                obs[r, c] = 1.
        return obs

    def _floor_obs(self, w):
        cards_here = self.loc_to_cards[w][self.p1_loc[w]]
        obs = np.zeros((len(RANKS), len(SUITS)))
        for rank, suit in cards_here:
            obs[rank, suit] = 1.
        return obs

    def _hand_obs(self, w):
        return np.zeros((3, len(RANKS), len(SUITS)))  # TODO

    def _render(self, mode='human', close=False):
        RENDER_WORLD = 0

        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.done[RENDER_WORLD]:
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0, MAX_BOARD_SIZE[1], 0, MAX_BOARD_SIZE[0] + 2)

        if not self.viewer.geoms:
            for r in range(self.walls.shape[1]):
                for c in range(self.walls.shape[2]):
                    if (r, c) in self.loc_to_cards[RENDER_WORLD]:
                        special = (0, 0) in self.loc_to_cards[RENDER_WORLD][(r, c)]
                        card = make_card(r, c, special=special)
                        self.viewer.add_geom(card)

                    if self.walls[RENDER_WORLD, r, c]:
                        wall = make_wall(r, c, invisible=self.walls[RENDER_WORLD, r, c] < 0.0)
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

        pr, pc = self.p1_loc[RENDER_WORLD]
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


class CardHUD(Geom):
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
