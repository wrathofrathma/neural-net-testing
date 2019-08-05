from Model import Model
import numpy as np
import tensorflow as tf
def print_ttt_board(state, move=None):
    board = ['.','.','.',
             '.','.','.',
             '.','.','.']
    for i in range(9):
        if(state[i]==1):
            # x exists
            board[i] = 'X'
        elif(state[i+9]==1):
            # o exists
            board[i] = 'O'
        else:
            # nothing there
            pass
    if(move!=None):
        print("applying move to board: " + str(move))
        if(move<9):
            board[move]='O'
        else:
            print("Invalid move")
    board = np.array(board)
    board=board.reshape(3,3)
    print(board[0])
    print(board[1])
    print(board[2])

def ttt_valid_move(game_state, move):
    if(move<0 or move>17):
        return False
    if(move<=8):
        if(game_state[move]==0 and game_state[move+9]==0):
            return True
        else:
            return False
    else:
        if(game_state[move]==0 and game_state[move-9]==0):
            return True
        else:
            return False

# Returns the sorted list of moves suggested by the engine.
def ttt_order_prediction(prediction):
    moves = []
    prediction = prediction[0]
    while(len(moves)<9):
        p = np.argmax(prediction)
        prediction[p] = 0
        moves.append(p)
    return moves

# Returns the first valid move given a game state and a list of moves
def get_first_valid_move(game_state, moves):
    print("Fetching valid move ")
    print(game_state)
    print(moves)
    for move in moves:
        if(ttt_valid_move(game_state, move)):
            return move
def int_ttt(filename):
    model = Model.load_from_file(filename)

    game_state = np.zeros(18)
    print_ttt_board(game_state)
    game_state = game_state.reshape(1,18)
    model.init_graph()
    with tf.Session() as sess:
        sess.run(model.init_op)
        print_ttt_board(game_state[0])

        # prediction = model.predict(game_state, sess)
        # p = ttt_order_prediction(prediction)
        # computer_move = get_first_valid_move(game_state[0], p)
        # game_state[0][computer_move+9]=1
        # print_ttt_board(game_state[0])
        while(True):
            # Getting user input
            move = input("Type a number 1-9 for the index to place your X: ")
            move = int(move)
            while(ttt_valid_move(game_state[0], move)==False):
                print("Invalid move input")
                move = input("Type a number 1-9 for the index to place your X: ")
                move = int(move)
            game_state[0][move]=1
            print_ttt_board(game_state[0])

            np.set_printoptions(suppress=True)
            print("Computer move")
            prediction = model.predict(game_state, sess)
            print("Prediction vector")
            print(prediction)
            p = ttt_order_prediction(prediction)
            print("Prediction array: " + str(p))
            c_move = get_first_valid_move(game_state[0],p)
            game_state[0][c_move+9]=1
            print_ttt_board(game_state[0])
