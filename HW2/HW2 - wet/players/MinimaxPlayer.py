"""
MiniMax Player
"""
from utils import MyPlayer
from SearchAlgos import MiniMax
DEBUG_PRINT = False


class Player(MyPlayer):
    def __init__(self, game_time, penalty_score):
        MyPlayer.__init__(self, game_time, penalty_score)
        self.search_alg = MiniMax(None, None, None)
