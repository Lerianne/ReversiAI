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
    max_depth = 4

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
          return None, self.evaluate_board(chess_board, player, opponent)

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
    player_score = np.sum(chess_board == player)
    opponent_score = np.sum(chess_board == opponent)
    return player_score - opponent_score

