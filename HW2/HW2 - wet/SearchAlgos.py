"""Search Algos: MiniMax, AlphaBeta
"""

from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
import numpy as np
DECIDING_AGENT = 1
OPPONENT = 2


class SearchAlgos:
    def __init__(self, utility, succ, perform_move, goal):
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
        self.goal_function = goal

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
        if depth == 0:
            return state.heuristic()
        if state.is_goal():
            if state.game_is_tied():
                return 0
            else:
                return state.game_score
        player = 1 if maximizing_player else 2
        if maximizing_player:
            curr_max = float('-inf')
            for child in state.succ(player):
                value = self.search(child, depth - 1, False)
                curr_max = max(value, curr_max)
            return curr_max
        else:
            curr_min = float('inf')
            for child in state.succ(player):
                value = self.search(child, depth - 1, True)
                curr_min = min(value, curr_min)
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
        if depth == 0:
            return state.heuristic()
        if state.is_goal():
            if state.game_is_tied():
                return 0
            else:
                return state.game_score
        player = 1 if maximizing_player else 2
        if maximizing_player:
            curr_max = float('-inf')
            for child in state.succ(player):
                value = self.search(child, depth - 1, False, alpha, beta)
                curr_max = max(value, curr_max)
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return float('inf')
            return curr_max
        else:
            curr_min = float('inf')
            for child in state.succ(player):
                value = self.search(child, depth - 1, True, alpha, beta)
                curr_min = min(value, curr_min)
                beta = min(curr_min, beta)
                if curr_min <= alpha:
                    return float('-inf')
            return curr_min


class State:
    def __init__(self, board: list, penalty, score=0, opponent_score=0, fruits_timer=None, fruits_dict:dict=None):
        self.board = board
        self.score = score
        self.opponent_score = opponent_score
        self.game_score = score - opponent_score
        self.loc = self.find_value(DECIDING_AGENT)
        self.opponent_loc = self.find_value(OPPONENT)
        self.penalty = penalty
        self.fruits_timer = fruits_timer
        self.fruits_dict = fruits_dict
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        # finding the fruits timer
        if fruits_timer is None:
            numrows = len(board)
            numcols = len(board[0])
            self.fruits_timer = min(numrows, numcols)

    def find_value(self, value_to_find):
        pos = np.where(self.board == value_to_find)
        # convert pos to tuple of ints
        return tuple(ax[0] for ax in pos)

    def valid_move(self, loc, move=(0, 0)):
        i = loc[0] + move[0]
        j = loc[1] + move[1]
        board = self.board
        in_board = 0 <= i < len(board) and 0 <= j < len(board[0])
        if in_board:
            white_or_fruit = self.board[i][j] == 0 or (self.board[i][j] not in [-1, 1, 2])
            return in_board and white_or_fruit
        else:
            return False

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
        return number_steps_available

    def game_is_tied(self):
        """
        Checks if the current state is a tie
        @return: boolean
        """
        tie_score = False
        if self.score == self.opponent_score:
            tie_score = True
        loc = self.loc
        opponent_loc = self.opponent_loc
        my_moves = len(self.steps_available(loc))
        opponent_moves = len(self.steps_available(opponent_loc))
        if my_moves == 0 and opponent_moves == 0 and tie_score:
            return True
        else:
            if my_moves == 0 and opponent_moves != 0:
                return (self.score - self.penalty) == self.opponent_score
            elif my_moves != 0 and opponent_moves == 0:
                return self.score == (self.opponent_score - self.penalty)
            else:
                return False

    def is_game_won(self):
        """
        Checks if that state is a win state.
        @return: bool, game score (positive if player 1 won)
        @rtype: tuple
        """
        if self.game_is_tied():
            return False, None
        my_available_steps = self.steps_available(self.loc)
        opp_available_steps = self.steps_available(self.opponent_loc)
        if len(my_available_steps) == 0 or len(opp_available_steps) == 0:
            return True, self.game_score
        else:
            return False, None

    def is_goal(self):
        won, _ = self.is_game_won()
        tie = self.game_is_tied()
        return won or tie

    def _update_locations(self):
        self.loc = self.find_value(DECIDING_AGENT)
        self.opponent_loc = self.find_value(OPPONENT)

    def make_move(self, player, direction):
        """

        @param player: 1 - for Deciding agent, 2 for the opponent
        @type player: int
        @param direction: a tuple from the directions list
        @type direction: tuple
        @return: The state updates itself
        @rtype:
        """
        self._update_locations()
        player_location = self.loc if player == DECIDING_AGENT else self.opponent_loc
        if not self.valid_move(player_location, direction):
            raise ValueError(f"The move {direction} for location {player_location} is not valid!")
        else:
            # make the location gray
            i, j = player_location
            self.board[i][j] = -1
            new_i, new_j = i + direction[0], j + direction[1]
            # check if there is a fruit in the new location
            if self.board[new_i][new_j] == 0:
                self.board[new_i][new_j] = player
            else:
                value = self.board[new_i][new_j]
                self.board[new_i][new_j] = player
                if player == DECIDING_AGENT:
                    self.score += value
                else:
                    self.opponent_score += value
                self.game_score = self.score - self.opponent_score
        self._update_locations()

    def succ(self, player):
        """
        A generator yielding all the successor states of a state
        @param player: 1 - Deciding agent, 2 for the opponent
        @type player: int
        @return: yielding the successors states
        @rtype:
        """
        location = self.find_value(player)
        for direction in self.directions:
            if self.valid_move(location, direction):
                new_board = self.board.copy()
                new_state = State(new_board, self.penalty, self.score, self.opponent_score,
                                  self.fruits_timer, self.fruits_dict)

                new_state.make_move(player, direction)
                yield new_state

    def count_zeroes(self):
        counter = 0
        for i, row in enumerate(self.board):
            for j, val in enumerate(row):
                if val == 0:
                    counter += 1
        return counter

    def number_of_reachable_nodes(self, loc, limit=float('inf')):
        queue = list()
        queue.append(loc)
        index = 0
        while index < len(queue) and index < limit:
            head_loc = queue[index]
            index += 1
            for direction in self.directions:
                i, j = head_loc[0] + direction[0], head_loc[1] + direction[1]
                if (i, j) not in queue and self.valid_move((i, j)):
                    queue.append((i, j))
        return index

    def longest_route_till_block(self, loc, limit=6):
        if limit == 0:
            return 0
        max_path_length = 0
        for direction in self.directions:
            if self.valid_move(loc, direction):
                new_i, new_j = loc[0] + direction[0], loc[1] + direction[1]
                self.board[new_i][new_j] = -1
                cur_max_length = self.longest_route_till_block((new_i, new_j), limit - 1) + 1
                self.board[new_i][new_j] = 0
                max_path_length = max(cur_max_length, max_path_length)
        return max_path_length

    def heuristic(self):
        count_zeroes = self.count_zeroes()
        factor = [count_zeroes, 9, count_zeroes, 1, 1]
        weights = [1, 1, 1, 2, 2]
        game_score = self.game_score
        fruits_score = self.fruits_game_score()
        hueristics = [self.number_of_reachable_nodes(self.loc) - self.number_of_reachable_nodes(self.opponent_loc),
                      2 * len(self.steps_available(self.loc)) - len(self.steps_available(self.opponent_loc)),
                      self.longest_route_till_block(self.loc)-self.longest_route_till_block(self.opponent_loc),
                      game_score,
                      fruits_score]
        result = 0
        for factor, weights, hueristic in zip(factor, weights, hueristics):
            result += (hueristic * weights) / factor
        return result

    def newheuristic(self):
        game_score = (self.get_game_score_heuristic(), 0.6)
        fruits_score = (self.get_fruit_score_heuristic(), 0.25)
        road_score = (self.get_longest_road_score(), 0.05)
        steps_score = (self.get_steps_available_score(), 0.05)
        reachable_nodes_score = (self.get_reachable_nodes_score(), 0.05)
        heuristics = [game_score, fruits_score, road_score, steps_score, reachable_nodes_score]
        result = 0
        for score, weight in heuristics:
            result = score * weight
        return result

    def set_rival_move(self, pos):
        old_i, old_j = self.find_value(2)
        new_i, new_j = pos
        self.board[old_i][old_j] = -1
        if self.board[new_i][new_j] == 0:
            self.board[new_i][new_j] = 2
        else:
            value = self.board[new_i][new_j]
            self.board[new_i][new_j] = 2
            self.opponent_score += value
            self.game_score = self.score - self.opponent_score
        self._update_locations()

    def find_minimum_path(self, loc, dst, visited=[]):
        if loc == dst:
            return 0
        else:
            visited.append(loc)
            curr_min = float('inf')
            for direction in self.directions:
                new_loc = loc[0] + direction[0], loc[1] + direction[1]
                if self.valid_move(loc, direction) and new_loc not in visited:
                    val = self.find_minimum_path(new_loc, dst, visited) + 1
                    curr_min = min(val, curr_min)
            visited.remove(loc)
            return curr_min

    def fruits_tick(self):
        if self.fruits_timer > 0:
            self.fruits_timer -= 1

    def fruit_score(self, loc, fruit_pos, value):
        distance = self.find_minimum_path(loc, fruit_pos)
        if distance == float('inf'):
            return 0
        else:
            if (distance * 2) <= self.fruits_timer:
                if distance == 0:
                    return value
                else:
                    return value / distance
            else:
                return 0

    def fruits_game_score(self):
        my_score = 0
        opponent_score = 0

        for pos, value in self.fruits_dict.items():
            my_score += self.fruit_score(self.loc, pos, value)
            opponent_score += self.fruit_score(self.opponent_loc, pos, value)
        max_score = max(my_score, opponent_score)
        if max_score == 0:
            return 0
        else:
            return my_score - opponent_score

    def update_fruits_on_board(self):
        # TODO move fruits update to the state from player
        if self.fruits_timer == 0:
            # deleting all of the fruits from the board
            for pos, value in self.fruits_dict.items():
                i, j = pos
                if self.board[i][j] == value:
                    self.board[i][j] = 0

    def get_game_score_heuristic(self):
        """
        get the heuristic for the game score
        @return: value at the range of [1,-1]
        @rtype: int
        """
        if self.score == self.opponent_score:
            return 0
        else:
            return (self.score - self.opponent_score) / (abs(self.score) + abs(self.opponent_score))

    def get_fruit_score_heuristic(self):

        def helper(loc):
            score = 0
            for direction in self.directions:
                if self.valid_move(self.loc, direction):
                    for pos, value in self.fruits_dict.items():
                        dist = self.find_minimum_path(self.loc, pos)
                        if dist != float('inf') and dist * 2 <= self.fruits_timer:
                            if dist == 0:
                                score += value
                            else:
                                score += value / dist
            return score

        my_score = helper(self.loc)
        opponent_score = helper(self.opponent_loc)
        factor = my_score + opponent_score
        if factor == 0:
            return 0
        else:
            return (my_score - opponent_score) / factor

    def get_reachable_nodes_score(self):
        my_reachable = self.number_of_reachable_nodes(self.loc)
        opponent_reachable = self.number_of_reachable_nodes(self.opponent_loc)
        zeros = self.count_zeroes()

        if zeros == 0:
            return 0
        else:
            return (my_reachable - opponent_reachable) / zeros

    def get_steps_available_score(self):
        my_steps = len(self.steps_available(self.loc))
        opponent_steps = len(self.steps_available(self.opponent_loc))
        both = my_steps + opponent_steps
        if both == 0:
            return 0
        else:
            return (my_steps - opponent_steps) / both

    def get_longest_road_score(self):
        my_road = self.longest_route_till_block(self.loc)
        opponent_road = self.longest_route_till_block(self.opponent_loc)
        both = my_road + opponent_road
        if both == 0:
            return 0
        else:
            return (my_road - opponent_road) / both
