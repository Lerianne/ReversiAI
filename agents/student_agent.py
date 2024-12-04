# Significant Input from:
# https://github.com/Roodaki/Minimax-Powered-Othello-Game/blob/master/src/ai_agent.py
# https://github.com/rohitv456/Othello-Game-alpha-Beta-pruning-/blob/master/MyBot.cpp
# and ChatGPT 

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
 
  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

  def step(self, chess_board, player, opponent):

    # Initializing start time
    start_time = time.time()

    # Setting a time limit of 1.95 seconds to give function time to return when limit is reached to not go over 2 sec
    time_limit = 1.95

    # Static search depth of 3
    depth = 3

    # Initalize best move
    best_move = None

    # Call minimax function with alpha = -inf and beta = inf, passing that we are starting as the max player
    # as well as the start time and time limit to check if we go over (and if so break and return)
    _, best_move = self.minimax(chess_board, player, opponent, depth, True, float("-inf"), float("inf"), start_time, time_limit)

    # Calcuating time taken to find best move
    time_taken = time.time() - start_time
    print("My AI's turn took ", time_taken, "seconds.")

    # Return best move found    
    return best_move


  # Alpha beta pruning algorithm adapted from:
  # https://github.com/Roodaki/Minimax-Powered-Othello-Game/blob/master/src/ai_agent.py
  # as well as use of ChatGPT
  # More information about sources can be found in the report
  def minimax(self, board, player, opponent, max_depth, isMaxPlayer, alpha, beta, start_time, time_limit):

    # Check if game is over
    is_endgame, _, _ = check_endgame(board, player, opponent)

    # Ff we hit a leaf or the game is over
    if max_depth == 0 or is_endgame:
        # Return value of evaulating the board and no best move
        return self.evaluate_board(board, player, opponent), None

    # If we are playing as the max player
    if isMaxPlayer:
        # Initalize best value
        best_value = float("-inf")
        # Initalize best move
        best_move = None
        # Call get_valid_moves to obtain a list of valid moves
        valid_moves = get_valid_moves(board, player)

        # Utilize move ordering to better prune branches using lambda function which iteratives over 
        # all the moves in valid moves, evaluates each of them, and sorts then in accending order (best move first)
        # Lambda function adapted from ChatGPT
        sorted_moves = sorted(valid_moves, key=lambda move: self.evaluate_move(board, move, player, opponent), reverse=True)

        for move in sorted_moves:

            # If we hit the time limit break out of the loop
            if time.time() - start_time > time_limit:
               break

            # Copy the board
            new_game = deepcopy(board)

            # Execute the move
            execute_move(new_game, move, player)

            # Call minimax, this time as min player, decrease depth of 1, pass alpha and beta values
            value, _ = self.minimax(new_game, player, opponent, max_depth - 1, False, alpha, beta, start_time, time_limit)

            # If value found from search is better than the best value so far
            if value > best_value:
                # set best move and best value as current move and value
                best_value = value
                best_move = move
              
            # Set alpha value as max between current alpha or new value
            alpha = max(alpha, value)
            
            # Pruning condition
            if beta <= alpha:
                break

        # Once loop is over, return best value and best move found
        return best_value, best_move

    # If playing as min player
    else:
        # Initalize min value
        min_value = float("inf")
        # Initalize best move 
        best_move = None

        # Get valid moves for opponent (as we are the min player now)    
        valid_moves = get_valid_moves(board, opponent)

        # Sort the moves as mentioned above
        sorted_moves = sorted(valid_moves, key=lambda move: self.evaluate_move(board, move, player, opponent), reverse=False)

        for move in sorted_moves:

            # Make copy of game board and execute the move
            new_game = deepcopy(board)
            execute_move(new_game, move, opponent)

            # If over the time limit, break the loop with current best results
            if time.time() - start_time > time_limit:
               break

            # Call minimax, this time as max player again, decrease depth, and pass current alpha/beta values
            value, _ = self.minimax(new_game, player, opponent, max_depth - 1, True, alpha, beta, start_time, time_limit)

            # If the value we found is lower than the most min value up to this point
            if value < min_value:
                # set new min value and best move 
                min_value = value
                best_move = move
            
            # Set beta as min between current beta or current value
            beta = min(beta, value)

            # Purning condition
            if beta <= alpha:
               break

        # Return the most min value and best move found from loop
        return min_value, best_move

  # Function used to evaluate the board at a given state
  def evaluate_board(self, board, player, opponent):

    # Coin parity (difference in disk count)
    # Adapted from ChatGPT
    player_count = np.sum(board == player)  
    opponent_count = np.sum(board == opponent) 
    parity = player_count - opponent_count

    # Mobility (number of valid moves)
    # Adapted from gpt_greedy_corners_agent with slight modification
    player_moves = len(get_valid_moves(board, player))
    opponent_moves = len(get_valid_moves(board, opponent))
    mobility_score = player_moves - opponent_moves

    # Reward being in corners, punish opponent being in corners
    # Adapted from gpt_greedy_corners_agent
    corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
    corner_score = sum(1 for corner in corners if board[corner] == player) * 10
    corner_penalty = sum(1 for corner in corners if board[corner] == opponent) * -10
    corners_score = corner_score + corner_penalty

    # Calculate stablility of pieces, adapted from
    # https://github.com/Roodaki/Minimax-Powered-Othello-Game/blob/master/src/ai_agent.py
    stability_score = self.get_stability(board, player)

    # Reward being on the edge of the board
    # Adapted from https://github.com/Roodaki/Minimax-Powered-Othello-Game/blob/master/src/ai_agent.py
    edge_score = sum(board[i][j] == player for i in [0, board.shape[0] - 1] for j in range(1, board.shape[1] - 1)) + \
                      sum(board[i][j] == player for i in range(1, board.shape[0] - 1) for j in [0, board.shape[1] - 1])

    # Return sum of each heuristic as overall evaluation 
    return parity + 2 * mobility_score + 5 * corners_score + 3 * stability_score + 2.5 * edge_score
    
  # Calculating stability of pieces 
  # Adapted from https://github.com/Roodaki/Minimax-Powered-Othello-Game/blob/master/src/ai_agent.py
  def get_stability(self, board, player):

    # Get the neighbours of a particular position (up, down, left, right)
     def neighbors(row, col):
        # Define the 8 possible neighboring positions (excluding diagonals)
        # Excludes the cell itself
        return [
            (row + dr, col + dc)
            for dr in [-1, 0, 1]
            for dc in [-1, 0, 1]
            # Checking boundary conditions
            if (dr, dc) != (0, 0) and 0 <= row + dr < board.shape[0] and 0 <= col + dc < board.shape[1]
        ]
     
     # Define regions of the board
     # Taken from eval function
     corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
     edges = [(i, j) for i in [0, board.shape[0] - 1] for j in range(1, board.shape[1] - 1)] + [
        (i, j) for i in range(1, board.shape[0] - 1) for j in [0, board.shape[1] - 1]
    ]

     # Define the inner reigon of the board 
     inner_region = [(i, j) for i in range(1, board.shape[0] - 1) for j in range(1, board.shape[1] - 1)]

    # Initalize stability count
     stable_count = 0

     # Returns boolea value depening on if the disk is condidered to be stable
     # Adapted from chatGPT
     def is_stable_disk(row, col):
        # Check if the disk is stable by checking its neighbors
        # A piece is stable if it's in the corners, edges, or if it's surrounded by its own pieces
        return (
            # All neighboring squares are player's pieces
            all(board[r][c] == player for r, c in neighbors(row, col)) 
            # The piece is in a corner or edge
            or (row, col) in edges + corners 
        )
     
     # Loop through the regions and count stable pieces
     for region in [corners, edges, inner_region]:
        for row, col in region:
            if board[row, col] == player and is_stable_disk(row, col):
                stable_count += 1

     return stable_count

  # Function that evaluates how 'good' a move was, used for move ordering in minimax
  def evaluate_move(self, board, move, player, opponent):

        # Create a copy of the board and execute the move
        new_board = deepcopy(board)
        execute_move(new_board, move, player)    

        # Ensure the move is None
        if move is not None:
            # count the number of pieces captured in that move
            capture_count = count_capture(board, move, player)
        
        else:
            capture_count = 0

        # Return both the evaluation of the board as well as how many pieces were captured
        return self.evaluate_board(new_board, player, opponent) + capture_count  
    
