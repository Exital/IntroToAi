"""
MiniMax Player
"""
from players.AbstractPlayer import AbstractPlayer
from utils import tup_add
import numpy as np
#TODO: you can import more modules, if needed


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time, penalty_score) # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.board = None
        self.loc = None
        self.opponent_loc = None
        self.my_score = 0
        self.opponent_score = 0
        self.fruits_timer = None
        self.fruits_dict = {}
        self.last_move = None  # tuple - (move, player)


    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        self.board = board
        # setting fruit's timer to the shortest edge * 2.
        numrows = len(board)
        numcols = len(board[0])
        self.fruits_timer = min(numrows, numcols) * 2
        # setting the locations of the players
        self.update_players_locations()

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        #TODO: erase the following line and implement this function.
        raise NotImplementedError

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        old_i, old_j = self.opponent_loc
        new_i, new_j = pos
        self.board[old_i][old_j] = -1
        if self.board[new_i][new_j] == 0:
            self.board[new_i][new_j] = 2
        else:
            value = self.board[new_i][new_j]
            self.board[new_i][new_j] = 2
            self.opponent_score += value
        self.update_players_locations()
        self.update_fruits_on_board(forward=True)

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        self.fruits_dict = fruits_on_board_dict
        self.update_fruits_on_board(init=True)

    def find_value(self, value_to_find):
        pos = np.where(self.board == value_to_find)
        # convert pos to tuple of ints
        return tuple(ax[0] for ax in pos)

    def update_players_locations(self):
        self.loc = self.find_value(1)
        self.opponent_loc = self.find_value(2)

    def update_fruits_on_board(self, forward=True, init=False):
        """
        Updates the fruits on the board, deleting or applying them.
        forward - bool: true if we are moving forward, false for backwards.
        """
        if init:
            tick = 0
        else:
            tick = 1 if forward else -1
        self.fruits_timer += tick

        # deleting all of the fruits from the board
        if self.fruits_timer <= 0:
            for pos, value in self.fruits_dict.items():
                i, j = pos
                if self.board[i][j] == value:
                    self.board[i][j] = 0

        # returning the fruits to the board
        if self.fruits_timer > 0:
            for pos, value in self.fruits_dict.items():
                i, j = pos
                if self.board[i][j] == 0:
                    self.board[i][j] = value

    def steps_available(self, loc):
        """
        Steps_available will calculate the steps available from a location
        @type loc: tuple
        @return: list - list of available moves
        """
        number_steps_available = []
        for d in self.directions:
            if self.valid_move(loc, d):
                number_steps_available.append(d)
        return len(number_steps_available)

    def valid_move(self, loc, move):
        i, j = tup_add(loc, move)
        board = self.board
        in_board = 0 <= i < len(board) and 0 <= j < len(board[0])
        if in_board:
            white_or_fruit = self.board[i][j] == 0 or (self.board[i][j] not in [-1, 1, 2])
            return white_or_fruit
        else:
            return False

    def game_is_tied(self):
        """
        Checks if the current state is a tie
        @return: boolean
        """
        tie_score = False
        if self.my_score == self.opponent_score:
            tie_score = True
        my_moves = self.steps_available(self.loc)
        opponent_moves = self.steps_available(self.opponent_loc)
        if my_moves == 0 and opponent_moves == 0 and tie_score:
            return True
        else:
            penalty = self.penalty_score
            if my_moves == 0 and opponent_moves != 0:
                return (self.my_score - penalty) == self.opponent_score
            elif my_moves != 0 and opponent_moves == 0:
                return self.my_score == (self.opponent_score - penalty)
            else:
                return False

    def is_game_won(self):
        """
        Checks if that state is a win state.
        @rtype: bool
        """
        if self.game_is_tied():
            return False
        my_available_steps = self.steps_available(self.loc)
        opp_available_steps = self.steps_available(self.opponent_loc)
        if my_available_steps == 0 or opp_available_steps == 0:
            return True
        else:
            return False

    def is_end_game(self):
        """
        Checking if this plauer's board is end game board.
        """
        win = self.is_game_won()
        tie = self.game_is_tied()
        return win or tie
