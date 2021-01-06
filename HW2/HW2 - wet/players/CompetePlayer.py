"""
Player for the competition
"""
from SearchAlgos import AlphaBeta
from utils import MyPlayer
import time as t
DEBUG_PRINT = False


class Player(MyPlayer):
    def __init__(self, game_time, penalty_score):
        MyPlayer.__init__(self, game_time, penalty_score)
        self.search_alg = AlphaBeta(None, None, None)
        self.time_factor = 0.3
        self.time_left = game_time

    def heuristic(self):
        result = self.get_game_score()
        if DEBUG_PRINT:
            print(f"value for compete heuristic is {result}")
        return result

    def make_move(self, time_limit, players_score):
        limit = None
        if self.search_alg is None:
            raise NotImplementedError("utils(make_move): self.search_alg is None!")
        time_start = t.time()
        only_move = self.check_one_move()
        if only_move is not None:
            max_move = only_move
        else:
            depth = 1
            max_move, max_val = self.search_alg.search(self, depth, True)
            last_iteration_time = t.time() - time_start
            next_iteration_max_time = 4 * last_iteration_time
            time_until_now = t.time() - time_start
            limit = self.time_left * self.time_factor
            while time_until_now + next_iteration_max_time < limit:
                depth += 1
                iteration_start_time = t.time()
                last_good_move = max_move
                max_move, val = self.search_alg.search(self, depth, True)
                if val == float('inf'):
                    break
                if val == float('-inf'):
                    max_move = last_good_move
                    break
                last_iteration_time = t.time() - iteration_start_time
                next_iteration_max_time = 4 * last_iteration_time
                time_until_now = t.time() - time_start
            self.time_left -= time_until_now
        self.perform_move(maximizing_player=True, move=max_move)
        if DEBUG_PRINT:
            print(f"move chosen is {max_move}\n"
                  f"time limit is {limit}\n"
                  f"time left is {self.time_left}")
        return max_move
