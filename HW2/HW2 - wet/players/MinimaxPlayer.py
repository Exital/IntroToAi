"""
MiniMax Player
"""
from utils import MyPlayer
from utils import tup_add
import numpy as np
import time as t
from SearchAlgos import MiniMax
DEBUG_PRINT = False


class Player(MyPlayer):
    def __init__(self, game_time, penalty_score):
        MyPlayer.__init__(self, game_time, penalty_score)
        self.search_alg = MiniMax(None, None, None)