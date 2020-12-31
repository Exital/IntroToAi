"""
MiniMax Player with AlphaBeta pruning
"""
from utils import MyPlayer
from SearchAlgos import AlphaBeta
#TODO: you can import more modules, if needed


class Player(MyPlayer):
    def __init__(self, game_time, penalty_score):
        MyPlayer.__init__(self, game_time, penalty_score) # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.search_alg = AlphaBeta(None, None, None)
