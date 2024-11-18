# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """
  WEIGHTS_6x6 = [
    100, -18,  12,  12, -18,  100,
   -18, -24,  -6,  -6, -24, -18,
    12,  -6,   6,   6,  -6,  12,
    12,  -6,   6,   6,  -6,  12,
   -18, -24,  -6,  -6, -24, -18,
    100, -18,  12,  12, -18,  100
]
  WEIGHTS_8x8 = [
     100, -24,  16,  16,  16,  16, -24,  100,
    -24, -32,  -8,  -8,  -8,  -8, -32, -24,
     16,  -8,   8,   0,   0,   8,  -8,  16,
     16,  -8,   0,   8,   8,   0,  -8,  16,
     16,  -8,   0,   8,   8,   0,  -8,  16,
     16,  -8,   8,   0,   0,   8,  -8,  16,
    -24, -32,  -8,  -8,  -8,  -8, -32, -24,
     100, -24,  16,  16,  16,  16, -24,  100
]
  WEIGHTS_10x10 = [
    100, -30,  20,  20,  20,  20,  20,  20, -30,  100,
   -30, -40, -10, -10, -10, -10, -10, -10, -40, -30,
    20, -10,  10,   0,   0,   0,   0,  10, -10,  20,
    20, -10,   0,  10,  10,  10,  10,   0, -10,  20,
    20, -10,   0,  10,  20,  20,  10,   0, -10,  20,
    20, -10,   0,  10,  20,  20,  10,   0, -10,  20,
    20, -10,   0,  10,  10,  10,  10,   0, -10,  20,
    20, -10,  10,   0,   0,   0,   0,  10, -10,  20,
   -30, -40, -10, -10, -10, -10, -10, -10, -40, -30,
    100, -30,  20,  20,  20,  20,  20,  20, -30,  100
]
  WEIGHTS_12x12 = [
    100, -36,  24,  24,  24,  24,  24,  24,  24,  24, -36,  100,
   -36, -48, -12, -12, -12, -12, -12, -12, -12, -12, -48, -36,
    24, -12,  12,   0,   0,   0,   0,   0,   0,  12, -12,  24,
    24, -12,   0,  12,  12,  12,  12,  12,  12,   0, -12,  24,
    24, -12,   0,  12,  24,  24,  24,  24,  12,   0, -12,  24,
    24, -12,   0,  12,  24,  36,  36,  24,  12,   0, -12,  24,
    24, -12,   0,  12,  24,  36,  36,  24,  12,   0, -12,  24,
    24, -12,   0,  12,  24,  24,  24,  24,  12,   0, -12,  24,
    24, -12,  12,   0,   0,   0,   0,   0,   0,  12, -12,  24,
   -36, -48, -12, -12, -12, -12, -12, -12, -12, -12, -48, -36,
    100, -36,  24,  24,  24,  24,  24,  24,  24,  24, -36,  100
]


  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

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

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
        
    # Define the depth of the search
    max_depth = 5
    initial_weight = 0
    

    # Run Alpha-Beta Pruning to find the best move
    best_move, _ = self.alpha_beta_search(chess_board, player, opponent, max_depth)
      
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return random_move(chess_board,player)
  
  def alpha_beta_search(self, chess_board, player, opponent, max_depth):
      """
      Alpha-Beta Pruning search algorithm to find the best move.
      """
      best_move = None
      alpha = float("-inf")
      beta = float("inf")

      # Get all valid moves for the current player
      valid_moves = get_valid_moves(chess_board, player)

      if not valid_moves:
          return None, self.evaluate_board(chess_board, player, opponent, weight)

      best_value = float("-inf")

      for move in valid_moves:
          # Simulate the move
          board_copy = deepcopy(chess_board)
          execute_move(board_copy, move, player)

          # Call the minimizer
          value = self.min_value(board_copy, player, opponent, alpha, beta, max_depth - 1)

          if value > best_value:
              best_value = value
              best_move = move

          # Update alpha
          alpha = max(alpha, best_value)

      return best_move, best_value

  def max_value(self, chess_board, player, opponent, alpha, beta, depth):
      """
      Maximizer for Alpha-Beta Pruning.
      """
      is_endgame, player_score, opponent_score = check_endgame(chess_board, player, opponent)
      if depth == 0 or is_endgame:
          return self.evaluate_board(chess_board, player, opponent)

      value = float("-inf")

      for move in get_valid_moves(chess_board, player):
          board_copy = deepcopy(chess_board)
          execute_move(board_copy, move, player)

          value = max(value, self.min_value(board_copy, player, opponent, alpha, beta, depth - 1))

          if value >= beta:
              return value
          alpha = max(alpha, value)

      return value

  def min_value(self, chess_board, player, opponent, alpha, beta, depth):
      """
      Minimizer for Alpha-Beta Pruning.
      """
      is_endgame, player_score, opponent_score = check_endgame(chess_board, player, opponent)
      if depth == 0 or is_endgame:
          return self.evaluate_board(chess_board, player, opponent)

      value = float("inf")

      for move in get_valid_moves(chess_board, opponent):
          board_copy = deepcopy(chess_board)
          execute_move(board_copy, move, opponent)

          value = min(value, self.max_value(board_copy, player, opponent, alpha, beta, depth - 1))

          if value <= alpha:
              return value
          beta = min(beta, value)

      return value

#Current heuristic is based on the number of pieces- need to change that!!!
  def evaluate_board(self, chess_board, player, opponent):
    """
    Basic evaluation function: difference in number of pieces.
    """
    # Get the board size
    board_size = chess_board.shape[0]

    # Select the correct weight variable based on the board size
    if board_size == 6:
        weights = self.WEIGHTS_6x6
    elif board_size == 8:
        weights = self.WEIGHTS_8x8
    elif board_size == 10:
        weights = self.WEIGHTS_10x10
    elif board_size == 12:
        weights = self.WEIGHTS_12x12
    else:
        raise ValueError("Unsupported board size")

    # Calculate the score based on the weights
    player_score = sum(
        weights[i * board_size + j]
        for i in range(board_size)
        for j in range(board_size)
        if chess_board[i, j] == player
    )
    opponent_score = sum(
        weights[i * board_size + j]
        for i in range(board_size)
        for j in range(board_size)
        if chess_board[i, j] == opponent
    )
    opponen_moves = len(get_valid_moves(chess_board, opponent))
    mobility_score = -opponen_moves

    # Return the difference as the evaluation score
    return player_score - opponent_score + mobility_score

    # player_score = np.sum(chess_board == player)
    # opponent_score = np.sum(chess_board == opponent)
    # return player_score - opponent_score + 

