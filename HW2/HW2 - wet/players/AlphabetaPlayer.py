"""
MiniMax Player with AlphaBeta pruning
"""
from utils import MyPlayer
from SearchAlgos import AlphaBeta


class Player(MyPlayer):
    def __init__(self, game_time, penalty_score):
        MyPlayer.__init__(self, game_time, penalty_score)
        self.search_alg = AlphaBeta(None, None, None)
