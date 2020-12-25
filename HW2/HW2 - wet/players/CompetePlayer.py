"""
Player for the competition
"""
from players.AbstractPlayer import AbstractPlayer
from SearchAlgos import AlphaBeta, State
import time as t
DEBUG = False
DEBUG_PRINT = False


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time, penalty_score)
        self.state = None
        self.time_left = game_time
        self.time_factor = 0.2

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        self.state = State(board, self.penalty_score)

    def choose_move(self, depth):
        max_value, max_value_move = float('-inf'), None
        alphabeta = AlphaBeta(None, None, None, None)
        for direction in self.directions:
            if self.state.valid_move(self.state.loc, direction):
                new_board = self.state.board.copy()
                new_state = State(new_board, self.penalty_score, self.state.score, self.state.opponent_score,
                                  self.state.fruits_timer, self.state.fruits_dict)
                new_state.make_move(1, direction)
                cur_minimax_val = alphabeta.search(new_state, depth - 1, True)
                if cur_minimax_val >= max_value:
                    max_value = cur_minimax_val
                    max_value_move = direction
        return max_value_move, max_value

    def check_one_move(self):
        count_moves = 0
        one_move = None
        for direction in self.directions:
            if self.state.valid_move(self.state.loc, direction):
                count_moves += 1
                one_move = direction
        if count_moves != 1:
            return None
        return one_move

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        time_start = t.time()
        only_move = self.check_one_move()
        if only_move is not None:
            max_move = only_move
        else:
            depth = 1
            max_move, max_val = self.choose_move(depth)
            last_iteration_time = t.time() - time_start
            next_iteration_max_time = 4 * last_iteration_time
            time_until_now = t.time() - time_start
            limit = self.time_left * self.time_factor
            while time_until_now + next_iteration_max_time < limit or (DEBUG and depth < 100):
                depth += 1
                iteration_start_time = t.time()
                max_move, val = self.choose_move(depth)
                last_iteration_time = t.time() - iteration_start_time
                next_iteration_max_time = 4 * last_iteration_time

                time_until_now = t.time() - time_start
            self.time_left -= time_until_now
            if DEBUG_PRINT:
                print(f"new location that was choosed is {self.state.loc}")
                print(f"move took: {time_until_now}\n"
                      f"time left: {self.time_left}\n"
                      f"time limit for move: {limit}")
        self.state.make_move(1, max_move)

        return max_move

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        self.state.set_rival_move(pos)

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        if self.state.fruits_dict is None:
            for (i, j), value in fruits_on_board_dict.items():
                self.state.board[i][j] = value
            self.state.fruits_dict = fruits_on_board_dict
        self.state.update_fruits_on_board()
