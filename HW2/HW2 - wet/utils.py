import operator
import numpy as np
import os
import time as t
from players.AbstractPlayer import AbstractPlayer
DEBUG_PRINT = False
ALPHA_VALUE_INIT = float('-inf')
BETA_VALUE_INIT = float('inf')


def get_directions():
    """Returns all the possible directions of a player in the game as a list of tuples.
    """
    return [(1, 0), (0, 1), (-1, 0), (0, -1)]


def tup_add(t1, t2):
    """
    returns the sum of two tuples as tuple.
    """
    return tuple(map(operator.add, t1, t2))


def get_board_from_csv(board_file_name):
    """Returns the board data that is saved as a csv file in 'boards' folder.
    The board data is a list that contains: 
        [0] size of board
        [1] blocked poses on board
        [2] starts poses of the players
    """
    board_path = os.path.join('boards', board_file_name)
    board = np.loadtxt(open(board_path, "rb"), delimiter=" ")
    
    # mirror board
    board = np.flipud(board)
    i, j = len(board), len(board[0])
    blocks = np.where(board == -1)
    blocks = [(blocks[0][i], blocks[1][i]) for i in range(len(blocks[0]))]
    start_player_1 = np.where(board == 1)
    start_player_2 = np.where(board == 2)
    
    if len(start_player_1[0]) != 1 or len(start_player_2[0]) != 1:
        raise Exception('The given board is not legal - too many start locations.')
    
    start_player_1 = (start_player_1[0][0], start_player_1[1][0])
    start_player_2 = (start_player_2[0][0], start_player_2[1][0])

    return [(i, j), blocks, [start_player_1, start_player_2]]


class MyPlayer(AbstractPlayer):
    """
    MyPlayer class is the master player that all player inherits.
    It has many methods that used by all players.
    """
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time,penalty_score)
        self.board = None
        self.loc = None
        self.opponent_loc = None
        self.my_score = 0
        self.opponent_score = 0
        self.fruits_timer = None
        self.fruits_dict = {}
        self.last_move = None  # tuple - (move, player)
        self.search_alg = None

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
            while time_until_now + next_iteration_max_time < time_limit:
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
        self.perform_move(maximizing_player=True, move=max_move)
        return max_move

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        old_i, old_j = self.opponent_loc
        move = (pos[0] - old_i, pos[1] - old_j)
        if not self.valid_move(self.opponent_loc, move):
            raise ValueError(f"set_rival_move: move - {move} is not valid move.")
        else:
            self.perform_move(maximizing_player=False, move=move)

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
        """
        finds a value on the board
        :param value_to_find: The value to find on the board.
        :return: A tuple, the location of the value on board.
        """
        pos = np.where(self.board == value_to_find)
        # convert pos to tuple of ints
        return tuple(ax[0] for ax in pos)

    def update_players_locations(self):
        """
        Updates the player locations on the board.
        """
        self.loc = self.find_value(1)
        self.opponent_loc = self.find_value(2)

    def update_fruits_on_board(self, forward=True, init=False):
        """
        Updates the fruits on the board, deleting or applying them.
        :param forward: bool, True if time moves forward.
        :param init: bool, True if no turn needs to be done but just update fruits on board.
        """
        if init:
            tick = 0
        else:
            tick = -1 if forward else 1
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
        :param loc: tuple, the location of the player
        :return: int - number of available moves for the player.
        """
        number_steps_available = []
        for d in self.directions:
            if self.valid_move(loc, d):
                number_steps_available.append(d)
        return len(number_steps_available)

    def valid_move(self, loc, move):
        """
        :param loc: tuple, the location of the player.
        :param move: tuple, the direction we check if valid.
        :return: bool, whether that move is valid from this location.
        """
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
        :return: true if the game is tied.
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
        :return: true if game is won.
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
        Checking if this player's board is end game board.
        :return: true if the board is end game board.
        """
        win = self.is_game_won()
        tie = self.game_is_tied()
        return win or tie

    def get_game_score(self):
        """
        Checks the game score and calculates the penalty into it.
        :return: value between [-1,1] where 1 is the best to our player and -1 is the worst.
        """
        if self.game_is_tied():
            return 0
        elif self.is_game_won():
            my_available_steps = self.steps_available(self.loc)
            opp_available_steps = self.steps_available(self.opponent_loc)
            my_score = self.my_score - self.penalty_score if my_available_steps == 0 else self.my_score
            opp_score = self.opponent_score - self.penalty_score if opp_available_steps == 0 else self.opponent_score
            return (my_score - opp_score) / (abs(my_score) + abs(opp_score))
        else:
            if abs(self.my_score) + abs(self.opponent_score) == 0:
                return 0
            return (self.my_score - self.opponent_score) / (abs(self.my_score) + abs(self.opponent_score))

    def number_of_reachable_nodes(self, loc, limit=float('inf')):
        """
        :param loc: tuple, the location of the player.
        :param limit: float, a limit number to stop searching.
        :return: number of nodes reachable by the player on location loc.
        """
        queue = list()
        queue.append(loc)
        index = 0
        while index < len(queue) and index < limit:
            head_loc = queue[index]
            index += 1
            for direction in self.directions:
                i, j = tup_add(head_loc, direction)
                if (i, j) not in queue and self.valid_move(head_loc, direction):
                    queue.append((i, j))
        return index

    def longest_route_till_block(self, loc, limit=6):
        """
        :param loc: tuple, location of the player.
        :param limit: int, limit to stop searching.
        :return: int, the longest road a player can achieve.
        """

        def valid_move(loc, move, board):
            i, j = tup_add(loc, move)
            in_board = 0 <= i < len(board) and 0 <= j < len(board[0])
            if in_board:
                white_or_fruit = self.board[i][j] == 0 or (self.board[i][j] not in [-1, 1, 2])
                return in_board and white_or_fruit
            else:
                return False

        def longest_route_recursion(loc, limit, board):
            if limit == 0:
                return 0
            max_path_length = 0
            for direction in self.directions:
                if valid_move(loc, direction, board):
                    new_i, new_j = tup_add(loc, direction)
                    board[new_i][new_j] = -1
                    cur_max_length = longest_route_recursion((new_i, new_j), limit - 1, board) + 1
                    board[new_i][new_j] = 0
                    max_path_length = max(cur_max_length, max_path_length)
            return max_path_length

        board_copy = self.board.copy()
        return longest_route_recursion(loc, limit, board_copy)

    def find_minimum_path(self, loc, dst, visited=[]):
        """
        find manhattan distance on board between loc and dst.
        :param loc: tuple, location of player.
        :param dst: tuple, destination location.
        :param visited: list of nodes already visited on the path.
        :return: int, minimum distance between location to destination.
        """
        if loc == dst:
            return 0
        else:
            visited.append(loc)
            curr_min = float('inf')
            for direction in self.directions:
                new_loc = tup_add(loc, direction)
                if self.valid_move(loc, direction) and new_loc not in visited:
                    val = self.find_minimum_path(new_loc, dst, visited) + 1
                    curr_min = min(val, curr_min)
            visited.remove(loc)
            return curr_min

    def get_fruit_score_heuristic(self):
        """
        calculates fruits heuristic
        :return: float, [-1,1] where 1 is the best for our player.
        """

        def helper(loc):
            score = 0
            for direction in self.directions:
                if self.valid_move(loc, direction):
                    for pos, value in self.fruits_dict.items():
                        dist = self.find_minimum_path(loc, pos)
                        if dist != float('inf') and dist <= self.fruits_timer / 2:
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
        """
        Reachable nodes heuristic
        :return: float, [-1,1].
        """
        my_reachable = self.number_of_reachable_nodes(self.loc)
        opponent_reachable = self.number_of_reachable_nodes(self.opponent_loc)
        factor = my_reachable + opponent_reachable

        if factor == 0:
            return 0
        else:
            return (my_reachable - opponent_reachable) / factor

    def get_steps_available_score(self):
        """
        Steps heuristic
        :return: float, [-1,1].
        """
        my_steps = self.steps_available(self.loc)
        opponent_steps = self.steps_available(self.opponent_loc)
        both = my_steps + opponent_steps
        if both == 0:
            return 0
        else:
            return (my_steps - opponent_steps) / both

    def get_longest_road_score(self):
        """
        Longest road heuristic
        :return: float, [-1,1].
        """
        my_road = self.longest_route_till_block(self.loc)
        opponent_road = self.longest_route_till_block(self.opponent_loc)
        both = my_road + opponent_road
        if both == 0:
            return 0
        else:
            return (my_road - opponent_road) / both

    def heuristic(self):
        """
        The general heuristic for players.
        The heuristic utilizes game score, fruits, longest road, moves available, nodes reachable.
        :return: float between [-1,1] where 1 is the best for our player.
        """
        game_score = (self.get_game_score(), 0.85)
        road_score = (self.get_longest_road_score(), 0.05)
        steps_score = (self.get_steps_available_score(), 0.05)
        reachable_nodes_score = (self.get_reachable_nodes_score(), 0.05)
        heuristics = [game_score, road_score, steps_score, reachable_nodes_score]
        result = 0
        for score, weight in heuristics:
            result += score * weight
        if DEBUG_PRINT:
            print(f"Heuristic value for location {self.loc} is {result}")
            print(f"\treachable score: {reachable_nodes_score[0] * reachable_nodes_score[1]}")
            print(f"\tsteps score: {steps_score[0] * steps_score[1]}")
            print(f"\tlongest road score: {road_score[0] * road_score[1]}")
            print(f"\tgame score: {game_score[0] * game_score[1]}")
        return result

    def perform_move(self, maximizing_player: bool, move, reverse=False):
        """
        Moving the player on the board and updating fruits, score, etc.
        :param maximizing_player: bool, True if our player.
        :param move: Tuple, the move to perform.
        :param reverse: bool, True if we would like to reverse the move.
        """
        player = 1 if maximizing_player else 2
        location = self.loc if maximizing_player else self.opponent_loc
        old_i, old_j = location
        # checking if that move is valid
        if not reverse:
            if not self.valid_move(location, move):
                raise ValueError(f"The move {move} for location {location} is not valid!")
            else:
                new_i, new_j = tup_add(location, move)
                # gray the old location
                self.board[old_i][old_j] = -1
                # check if new location has fruit
                value = self.board[new_i][new_j]
                self.board[new_i][new_j] = player
                if maximizing_player:
                    self.my_score += value
                else:
                    self.opponent_score += value
                self.update_players_locations()
                self.update_fruits_on_board(forward=True)
        else:
            # calculate the reverse move
            reverse_move = (move[0] * -1, move[1] * -1)
            new_i, new_j = tup_add(location, reverse_move)
            # make old location white
            self.board[old_i][old_j] = 0
            # get the player to new location
            self.board[new_i][new_j] = player
            # check if there was a fruit
            self.update_fruits_on_board(forward=False)
            value = self.board[old_i][old_j]
            if maximizing_player:
                self.my_score -= value
            else:
                self.opponent_score -= value
            self.update_players_locations()

    def check_one_move(self):
        """
        Checks whether our player has only 1 valid move.
        :return: tuple, the valid move.
        """
        count_moves = 0
        one_move = None
        for direction in self.directions:
            if self.valid_move(self.loc, direction):
                count_moves += 1
                one_move = direction
        if count_moves != 1:
            return None
        return one_move
