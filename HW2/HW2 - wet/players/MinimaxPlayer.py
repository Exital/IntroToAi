"""
MiniMax Player
"""
DEBUG = False
import time as t
from players.AbstractPlayer import AbstractPlayer
from SearchAlgos import MiniMax, State
#TODO: you can import more modules, if needed


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time, penalty_score)
        self.state = None

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
        minimax = MiniMax(None, None, None, None)
        for direction in self.directions:
            if self.state.valid_move(self.state.loc, direction):
                new_board = self.state.board.copy()
                new_state = State(new_board, self.penalty_score, self.state.score, self.state.opponent_score,
                                  self.state.fruits_timer, self.state.fruits_dict)
                new_state.make_move(1, direction)
                cur_minimax_val = minimax.search(new_state, depth - 1, False)
                if DEBUG:
                    print(f"The hueristic for {new_state.loc} is {cur_minimax_val}")
                if cur_minimax_val >= max_value:
                    max_value = cur_minimax_val
                    max_value_move = direction
        return max_value_move

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
            move = only_move
        else:
            depth = 1
            move = self.choose_move(depth)
            last_iteration_time = t.time() - time_start
            next_iteration_max_time = 4 * last_iteration_time
            time_until_now = t.time() - time_start
            # DEBUG = self.loc==(4,9)
            while time_until_now + next_iteration_max_time < time_limit or (DEBUG and depth < 100):
                depth += 1
                iteration_start_time = t.time()
                move = self.choose_move(depth)
                last_iteration_time = t.time() - iteration_start_time
                next_iteration_max_time = 4 * last_iteration_time
                time_until_now = t.time() - time_start
        self.state.make_move(1, move)
        return move

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        self.state.set_rival_move(pos)


    def update_fruits(self, fruits_on_board_dict: dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        # TODO move this to the state from the player.
        for (i, j), value in fruits_on_board_dict.items():
            self.state.board[i][j] = value
        self.state.fruits_dict = fruits_on_board_dict


    ########## helper functions in class ##########
    #TODO: add here helper functions in class, if needed


    ########## helper functions for MiniMax algorithm ##########
    #TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm