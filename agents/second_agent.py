# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("second_agent")
class SecondAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """
  MOVE_COUNT =0

  def __init__(self):
    super(SecondAgent, self).__init__()
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

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    time_limit = 1.97
    depth = 1
    best_move = None

    max_depth = 1058

    while True:
        elapsed_time = time.time() - start_time
        
        if elapsed_time >= time_limit or (depth > max_depth):
            break  # Stop if time is over

        _, move = self.minimax(chess_board, player, opponent, depth, True, float("-inf"), float("inf"), start_time, time_limit)

        if move is not None:
           best_move = move
        
        depth += 1


    if best_move is None:
       return random_move(chess_board, player)

    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")
    self.MOVE_COUNT +=1

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return best_move
  
  def minimax(self, board, player, opponent, max_depth, isMaxPlayer, alpha, beta, start_time, time_limit):
    
    elapsed_time = time.time() - start_time

    if elapsed_time >= time_limit:
        return float("-inf") if isMaxPlayer else float("inf"), None

    is_endgame, _, _ = check_endgame(board, player, opponent)
    if max_depth == 0 or is_endgame:
        return self.evaluate_board(board, player, opponent), None
        
    valid_moves = get_valid_moves(board, player)
        
    #sorted_moves = sorted(valid_moves, key=lambda move: self.evaluate_move(board, move, player if isMaxPlayer else opponent, opponent if isMaxPlayer else player), reverse=isMaxPlayer)
    sorted_moves = sorted(valid_moves, key=lambda move: (
    5 * self.is_corner(move, board) - 
    3 * self.is_trap(move, board, opponent) +
    self.evaluate_board(deepcopy(board), player, opponent)
), reverse=isMaxPlayer)


    if isMaxPlayer:
        best_value = float("-inf")
        best_move_so_far = None
            
        for move in sorted_moves:
            new_game = deepcopy(board)
            execute_move(new_game, move, player)

            value, _ = self.minimax(new_game, player, opponent, max_depth - 1, False, alpha, beta, start_time, time_limit)
                
            if value > best_value:
                best_value = value
                best_move_so_far = move

            alpha = max(alpha, value)
            
            if beta <= alpha:
                break
            
        return best_value, best_move_so_far
            
    else:
        min_value = float("inf")
        best_move_so_far = None

        for move in sorted_moves:
            new_game = deepcopy(board)
            execute_move(new_game, move, opponent)

            value, _ = self.minimax(new_game, player, opponent, max_depth - 1, True, alpha, beta, start_time, time_limit)

            if value < min_value:
                min_value = value
                best_move_so_far = move

            beta = min(beta, value)
            if beta <= alpha:
                break
            
        return min_value, best_move_so_far

  
  def evaluate_board(self, board, player, opponent):
    # Coin parity (difference in disk count)
    player_count = np.sum(board == player)  # Count pieces for the player
    opponent_count = np.sum(board == opponent)  # Count pieces for the opponent
    
    parity = player_count - opponent_count

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

    return parity + 2 * mobility_score + 5 * corners_score + 3*stability_score + 2.5*edge_score
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
        if (row, col) in corners:
            return True
        elif (row, col) in edges:
            neighbors_on_edge = [
                (row + dr, col) for dr in [-1, 1] if 0 <= row + dr < board.shape[0]
            ] + [
                (row, col + dc) for dc in [-1, 1] if 0 <= col + dc < board.shape[1]
            ]
            return all(board[r][c] == player for r, c in neighbors_on_edge)
        else:
            return all(board[r][c] == player for r, c in neighbors(row, col))
     
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
    execute_move(new_board, move, player)

    # You can use any part of your evaluation function here, for example:
    return self.evaluate_board(new_board, player, opponent)      

  def early_game(self, chess_board):
        if self.MOVE_COUNT <= 2 & chess_board.shape[0] == 6:
            return True
        elif self.MOVE_COUNT <= 3 & chess_board.shape[0] == 8:
            return True
        elif self.MOVE_COUNT <= 4 & chess_board.shape[0] == 10:
            return True
        elif self.MOVE_COUNT <= 5 & chess_board.shape[0] == 12:
            return True
        else: 
          return False
        
    
  def is_corner(self, move, board):
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        return 1 if move in corners else 0
  
  def is_trap(self, move, board, opponent):
    """
    Evaluates if a move places the opponent in a disadvantageous position (a 'trap').
    """
    corners = [(0, 0), (0, len(board) - 1), (len(board) - 1, 0), (len(board) - 1, len(board) - 1)]
    move_x, move_y = move  # Assuming move is a tuple of (row, col)
    
    for corner in corners:
        corner_x, corner_y = corner
        # Check if corner is within the board and if it's occupied by the opponent
        if (
            0 <= corner_x < len(board) and
            0 <= corner_y < len(board[0]) and
            board[corner_x][corner_y] == opponent
        ):
            return True  # It's a trap
        
    return False
