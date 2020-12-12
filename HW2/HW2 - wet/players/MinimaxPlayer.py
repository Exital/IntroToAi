"""
MiniMax Player
"""
import time as t
from players.AbstractPlayer import AbstractPlayer
CURRENT_AGENT = 1
#TODO: you can import more modules, if needed


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time, penalty_score)
        self.loc = None
        self.opponent_loc = None
        self.board = None
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        self.board = board
        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == 1:
                    self.loc = (i, j)
                elif val == 2:
                    self.opponent_loc = (i, j)

    def valid_move(self, loc, move=(0, 0)):
        i = loc[0] + move[0]
        j = loc[1] + move[1]
        in_board = 0 <= i < len(self.board) and 0 <= j < len(self.board[0])
        white_or_fruit = self.board[i][j] == 0 or (self.board[i][j] not in [-1, 1, 2])
        return in_board and white_or_fruit

    def player_got_only_one_move(self):
        count_moves=0
        one_move = None
        for direction in self.directions:
            if self.valid_move(self.loc, direction):
                count_moves += 1
                one_move = direction
        if count_moves != 1:
            return None
        return one_move

    def choose_move(self, depth):
        max_value, max_value_move = float('-inf'), None
        for direction in self.directions:
            if self.valid_move(self.loc, direction):
                new_i = self.loc[0] + direction[0]
                new_j = self.loc[1] + direction[1]
                self.board[new_i, new_j] = -1
                self.loc = new_i, new_j
                cur_minimax_val = self.minimax(depth-1, 3-CURRENT_AGENT)
                if cur_minimax_val >= max_value:
                    max_value = cur_minimax_val
                    max_value_move = direction
                # returning the board to the former state
                self.board[self.loc] = 0
                self.loc = self.loc[0] - direction[0], self.loc[1] - direction[1]
        return max_value_move

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        time_start = t.time()
        only_move = self.player_got_only_one_move()
        if only_move is not None:
            move = only_move
        else:
            depth = 1
            leafs_count = [0]
            move = self.choose_move(depth)
            last_iteration_time = t.time() - time_start
            next_iteration_max_time = 4 * last_iteration_time
            time_until_now = t.time() - time_start
            # DEBUG = self.loc==(4,9)
            DEBUG = False
            while time_until_now + next_iteration_max_time < time_limit or (DEBUG and depth < 100):
                depth += 1

                iteration_start_time = t.time()
                leafs_count[0] = 0
                move = self.choose_move(depth)
                last_iteration_time = t.time() - iteration_start_time
                next_iteration_max_time = 4 * last_iteration_time
                time_until_now = t.time() - time_start
            """
            if move is None:
                # print(self.board)
                exit()
            """
        self.loc = (self.loc[0] + move[0], self.loc[1] + move[1])
        self.board[self.loc] = -1
        return move


    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        #TODO: erase the following line and implement this function.
        raise NotImplementedError


    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        #TODO: erase the following line and implement this function. In case you choose not to use it, use 'pass' instead of the following line.
        raise NotImplementedError


    ########## helper functions in class ##########
    #TODO: add here helper functions in class, if needed


    ########## helper functions for MiniMax algorithm ##########
    #TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm