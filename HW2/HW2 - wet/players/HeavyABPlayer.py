"""
MiniMax Player with AlphaBeta pruning with heavy heuristic
"""
from utils import MyPlayer
from SearchAlgos import AlphaBeta
DEBUG_PRINT = False


class Player(MyPlayer):
    def __init__(self, game_time, penalty_score):
        MyPlayer.__init__(self, game_time, penalty_score)
        self.search_alg = AlphaBeta(None, None, None)
        self.depth = 3

    def make_move(self, time_limit, players_score):
        only_move = self.check_one_move()
        if only_move is not None:
            max_move = only_move
        else:
            max_move, val = self.search_alg.search(self, self.depth, True)
            if DEBUG_PRINT:
                print(f"Best move till now is {max_move} with val {val}")
        self.perform_move(maximizing_player=True, move=max_move)
        return max_move
