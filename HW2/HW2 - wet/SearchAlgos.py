"""Search Algos: MiniMax, AlphaBeta
"""
from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
PRINT_DEBUG = False

class SearchAlgos:
    def __init__(self, utility, succ, perform_move, goal=None):
        """The constructor for all the search algos.
        You can code these functions as you like to, 
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move

    def search(self, state, depth, maximizing_player):
        pass


class MiniMax(SearchAlgos):

    def search(self, state, depth, maximizing_player):
        """Start the MiniMax algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        max_value, max_value_move = float('-inf'), None
        for direction in state.directions:
            if state.valid_move(state.loc, direction):
                state.perform_move(maximizing_player, direction)
                cur_minimax_val = self.minimax_search(state, depth - 1, False)
                state.perform_move(maximizing_player, direction, reverse=True)
                if PRINT_DEBUG:
                    print(f"The heuristic for {direction} is {cur_minimax_val} in depth: {depth}")
                if cur_minimax_val >= max_value:
                    max_value = cur_minimax_val
                    max_value_move = direction
        if PRINT_DEBUG:
            print(f"chosen move is {max_value_move} and value is {max_value}")
        return max_value_move, max_value

    def minimax_search(self, state, depth, maximizing_player):
        if depth == 0:
            return state.heuristic()
        if state.is_end_game():
            if state.game_is_tied():
                return 0
            else:
                if state.get_game_score() > 0:
                    return float('inf')
                else:
                    return float('-inf')
        location = state.loc if maximizing_player else state.opponent_loc
        if maximizing_player:
            curr_max = float('-inf')
            for move in state.directions:
                if state.valid_move(location, move):
                    state.perform_move(maximizing_player, move)
                    value = self.minimax_search(state, depth - 1, False)
                    state.perform_move(maximizing_player, move, reverse=True)
                    curr_max = max(curr_max, value)
            return curr_max
        else:
            curr_min = float('inf')
            for move in state.directions:
                if state.valid_move(location, move):
                    state.perform_move(maximizing_player, move)
                    value = self.minimax_search(state, depth - 1, True)
                    state.perform_move(maximizing_player, move, reverse=True)
                    curr_min = min(curr_min, value)
            return curr_min


class AlphaBeta(SearchAlgos):

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        max_value, max_value_move = float('-inf'), None
        for direction in state.directions:
            if state.valid_move(state.loc, direction):
                state.perform_move(maximizing_player, direction)
                cur_minimax_val = self.alphabeta_search(state, depth - 1, False)
                state.perform_move(maximizing_player, direction, reverse=True)
                if PRINT_DEBUG:
                    print(f"The heuristic for {direction} is {cur_minimax_val} in depth: {depth}")
                if cur_minimax_val >= max_value:
                    max_value = cur_minimax_val
                    max_value_move = direction
        if PRINT_DEBUG:
            print(f"chosen move is {max_value_move} and value is {max_value}")
        return max_value_move, max_value

    def alphabeta_search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        if depth == 0:
            return state.heuristic()
        if state.is_end_game():
            if state.game_is_tied():
                return 0
            else:
                if state.get_game_score() > 0:
                    return float('inf')
                else:
                    return float('-inf')
        location = state.loc if maximizing_player else state.opponent_loc
        if maximizing_player:
            curr_max = float('-inf')
            for move in state.directions:
                if state.valid_move(location, move):
                    state.perform_move(maximizing_player, move)
                    value = self.alphabeta_search(state, depth - 1, False)
                    state.perform_move(maximizing_player, move, reverse=True)
                    curr_max = max(curr_max, value)
                    alpha = max(curr_max, alpha)
                    if curr_max >= beta:
                        return float('inf')
            return curr_max
        else:
            curr_min = float('inf')
            for move in state.directions:
                if state.valid_move(location, move):
                    state.perform_move(maximizing_player, move)
                    value = self.alphabeta_search(state, depth - 1, True)
                    state.perform_move(maximizing_player, move, reverse=True)
                    curr_min = min(curr_min, value)
                    beta = min(curr_min, beta)
                    if curr_min <= alpha:
                        return float('-inf')
            return curr_min
