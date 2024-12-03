# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("third_agent")
class ThirdAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(ThirdAgent, self).__init__()
    self.name = "SecondAgent"

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """
    start_time = time.time()
    time_limit = 1.95
    depth = 4
    best_move = None

    _, best_move = self.minimax(chess_board, player, opponent, depth, True, float("-inf"), float("inf"), start_time, time_limit)
    
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")
    return best_move
  
  def minimax(self, board, player, opponent, max_depth, isMaxPlayer, alpha, beta, start_time, time_limit):

    is_endgame, _, _ = check_endgame(board, player, opponent)
    if max_depth == 0 or is_endgame:
        return self.evaluate_board(board, player, opponent), None
    
    if isMaxPlayer:
        best_value = float("-inf")
        best_move = None
        valid_moves = get_valid_moves(board, player)

        #if not valid_moves:
            #return self.evaluate_board(board, player, opponent), None

        sorted_moves = sorted(valid_moves, key=lambda move: self.evaluate_move(board, move, player, opponent), reverse=True)
        
        for move in sorted_moves:

            if time.time() - start_time > time_limit:
               #print(best_move)
               break

            new_game = deepcopy(board)

            if move is not None:
                execute_move(new_game, move, player)

            value, _ = self.minimax(new_game, player, opponent, max_depth - 1, False, alpha, beta, start_time, time_limit)
             
            if value > best_value:
                best_value = value
                best_move = move
            #print("value: ", value)
            alpha = max(alpha, value)
            #print("alpha: ", alpha)
            if beta <= alpha:
                break
           
        return best_value, best_move
        
    else:
        min_value = float("inf")
        best_move = None
        valid_moves = get_valid_moves(board, opponent)

        #if not valid_moves:
            #return self.evaluate_board(board, player, opponent), None

        sorted_moves = sorted(valid_moves, key=lambda move: self.evaluate_move(board, move, player, opponent), reverse=False)

        for move in sorted_moves:
            new_game = deepcopy(board)

            if move is not None:
                execute_move(new_game, move, opponent)

            if time.time() - start_time > time_limit:
               #print(best_move)
               break

            value, _ = self.minimax(new_game, player, opponent, max_depth - 1, True, alpha, beta, start_time, time_limit)

            if value < min_value:
                min_value = value
                best_move = move
            #print("value: ", value)
            beta = min(beta, value)
            #print("beta: ", beta)
            if beta <= alpha:
               break
        
        return min_value, best_move
  
  def evaluate_board(self, board, player, opponent):

    # Coin parity (difference in disk count)
    player_count = np.sum(board == player)  # Count pieces for the player
    opponent_count = np.sum(board == opponent)  # Count pieces for the opponent
    
    pairty = player_count - opponent_count

    # Mobility (number of valid moves)
    player_moves = len(get_valid_moves(board, player))
    opponent_moves = len(get_valid_moves(board, opponent))
    mobility_score = player_moves -opponent_moves

    # Reward being in corners, punish opponent being in corners

    corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
    corner_score = sum(1 for corner in corners if board[corner] == player) * 10
    corner_penalty = sum(1 for corner in corners if board[corner] == opponent) * -10

    corners_score = corner_score + corner_penalty

    stability_score = self.get_stability(board, player)

    edge_score = sum(board[i][j] == player for i in [0, board.shape[0] - 1] for j in range(1, board.shape[1] - 1)) + \
                      sum(board[i][j] == player for i in range(1, board.shape[0] - 1) for j in [0, board.shape[1] - 1])

    return pairty + 2 * mobility_score + 5 * corners_score + 3 * stability_score + 2.5 * edge_score
    #return pairty
  
  def get_stability(self, board, player):
     
     def neighbors(row, col):
        # Define the 8 possible neighboring positions (excluding diagonals)
        return [
            (row + dr, col + dc)
            for dr in [-1, 0, 1]
            for dc in [-1, 0, 1]
            if (dr, dc) != (0, 0) and 0 <= row + dr < board.shape[0] and 0 <= col + dc < board.shape[1]
        ]
     
     # Define regions of the board
     corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
     edges = [(i, j) for i in [0, board.shape[0] - 1] for j in range(1, board.shape[1] - 1)] + [
        (i, j) for i in range(1, board.shape[0] - 1) for j in [0, board.shape[1] - 1]
    ]
     inner_region = [(i, j) for i in range(1, board.shape[0] - 1) for j in range(1, board.shape[1] - 1)]

     stable_count = 0

     def is_stable_disk(row, col):
        # Check if the disk is stable by analyzing its neighbors
        # A piece is stable if it's in the corners, edges, or if it's surrounded by its own pieces
        return (
            all(board[r][c] == player for r, c in neighbors(row, col))  # All neighboring squares are player's pieces
            or (row, col) in edges + corners  # The piece is in a corner or edge
        )
     
      # Loop through the regions and count stable pieces
     for region in [corners, edges, inner_region]:
        for row, col in region:
            if board[row, col] == player and is_stable_disk(row, col):
                stable_count += 1

     return stable_count

  def evaluate_move(self, board, move, player, opponent):

        """
        Evaluate a specific move for a given player. The higher the score, the better the move.
        """
        # Create a copy of the board and apply the move
        new_board = deepcopy(board)

        if move is not None:
            execute_move(new_board, move, player)    

        if move is not None:
            capture_count = count_capture(board, move, player)
        
        else:
            capture_count = 0

        # You can use any part of your evaluation function here, for example:
        return self.evaluate_board(new_board, player, opponent) + capture_count  
    
