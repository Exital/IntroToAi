"""
MiniMax Player with AlphaBeta pruning with light heuristic
"""
from SearchAlgos import State, AlphaBeta
from players.AbstractPlayer import AbstractPlayer
DEBUG_PRINT = False
DEBUG = False
DEPTH = 5


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
        alphabeta = AlphaBeta(self.light_heuristic, None, None, None)
        for direction in self.directions:
            if self.state.valid_move(self.state.loc, direction):
                new_board = self.state.board.copy()
                new_state = State(new_board, self.penalty_score, self.state.score, self.state.opponent_score,
                                  self.state.fruits_timer, self.state.fruits_dict)
                new_state.make_move(1, direction)
                cur_minimax_val = alphabeta.search(new_state, depth - 1, False)
                if DEBUG_PRINT:
                    print(f"The hueristic for {new_state.loc} is {cur_minimax_val} in depth: {depth}"
                          f""
                          f"The score is {new_state.score}")
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
        only_move = self.check_one_move()
        if only_move is not None:
            max_move = only_move
        else:
            max_move, val = self.choose_move(DEPTH)
            if DEBUG_PRINT:
                print(f"Best move till now is {max_move} with val {val}")
        self.state.make_move(1, max_move)
        if DEBUG_PRINT:
            print(f"new location that was choosed is {self.state.loc}")
        return max_move

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        self.state.set_rival_move(pos)


    def light_heuristic(self):
        game_score = (self.state.get_game_score_heuristic(), 0.8)
        moves_available = len(self.state.steps_available(self.state.loc))
        moves = (4 - moves_available) / 4
        moves_score = (moves, 0.2)
        heuristics = [game_score, moves_score]
        result = 0
        for score, weight in heuristics:
            result += score * weight
        # if DEBUG_PRINT:
        #     print(f"Heuristic for location - {self.state.loc} value is {result}\n"
        #            f"moves score is: {moves * moves_score[1]}\n"
        #           f"game score is: {game_score[0] * game_score[1]}")
        return result

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
