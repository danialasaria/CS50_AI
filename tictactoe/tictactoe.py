"""
Tic Tac Toe Player
"""

import math
import copy
from typing import Tuple

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    counter = 0
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j]:
                counter += 1
    # X always moves first; when the number of filled cells is even, it's X's turn
    if counter % 2 == 0:
        return X
    else:
        return O

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == None:
                possible_actions.add((i,j))
    return possible_actions

def result(board, action: Tuple[int, int]):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i,j = action
    if not 0 <= i <=2 or not 0 <= j <=2 or board[i][j] is not None:
        raise ValueError('Invalid action')
    new_board = copy.deepcopy(board)
    new_board[i][j] = player(board)
    return new_board

def is_board_full(board):
    return all(cell != EMPTY for row in board for cell in row)
    
def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    #check rows and columns
    for i in range(3):
        #row i
        if board[i][0]==board[i][1] == board[i][2] and board[i][0] is not None:
            return board[i][0]
        #col i
        if board[0][i]==board[1][i] == board[2][i] and board[0][i] is not None:
            return board[0][i]
        
    #check diagonals        
    if board[0][0]==board[1][1] == board[2][2] and board[0][0] is not None:
        return board[0][0]
    
    if board[0][2]==board[1][1] == board[2][0] and board[0][2] is not None:
        return board[0][2]
    
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    return is_board_full(board) or winner(board) is not None
    

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    game_winner = winner(board)
    if game_winner == X:
        return 1
    elif game_winner == O:
        return -1
    else: 
        return 0

def max_value(board, alpha, beta):
    if terminal(board):
        return utility(board), None
    
    best_move = None
    best_score = -math.inf
    for move in actions(board):
        new_board = result(board, move)
        score, _ = min_value(new_board, alpha, beta)
        
        if score > best_score:
            best_score = score
            best_move = move
            # Correct alpha update: compare against current alpha, not beta
            alpha = max(alpha, best_score)
        #O would never allow as they can enforce a better outcome with beta
        if best_score >= beta:
            break #prune branches left
    
    return best_score, best_move

def min_value(board, alpha, beta):
    if terminal(board):
        return utility(board), None
    
    best_move = None
    best_score = math.inf
    for move in actions(board):
        new_board = result(board, move)
        score, _ = max_value(new_board, alpha, beta)
        
        if score < best_score:
            best_score = score
            best_move = move
            beta = min(beta, best_score)
        if best_score <= alpha:
            break #prune branches left
    
    return best_score, best_move
        
def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    curr_player = player(board)
    
    #maximize
    if curr_player == X:
        _, move = max_value(board, -math.inf, math.inf)
        return move
    elif curr_player == O:
        _, move = min_value(board, -math.inf, math.inf)
        return move
