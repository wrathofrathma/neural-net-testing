#!/usr/bin/env python
import numpy as np
import random 
# Env class handles the state of the individual game
# defined methods
# step - takes in the action vector.
# Returns the next state, reward, whether we're finished, and info
# reset - resets the environment to clean
# randomize - randomizes the environment(useful in training models that compete from scratch)

class TTT_Env:
    def __init__(self):
        self.state = np.zeros(18)

    # No validation. Just assume the move is valid and apply it. Return the resulting state & possible winner.
    def step(self, action):
        # Minor bandaid to stop applying moves on top of one another
        move = np.argmax(action>0)
        if(self.state[move]==1):
            return (self.state, False, None)
        self.state += action
        cw = self.check_win()
        if(cw=='X'):
            reward = -1
            done=True
        elif(cw=='O'):
            reward = 1
            done=True
        elif(cw=='D'):
            reward = 0.5
            done=True
        else:
            reward=0
            done=False
        return (self.state, done, reward)
        
    def check_win(self):
        X = self.state[:9]
        X = X.reshape(3,3)
        O = self.state[9:]
        O = O.reshape(3,3)

        for i in range(3):
            # Check columns
            if(X[i][0]==1 and X[i][1]==1 and X[i][2]==1):
                return 'X'
            if(O[i][0]==1 and O[i][1]==1 and O[i][2]==1):
                return 'O'
            # Check rows
            if(X[0][i]==1 and X[1][i]==1 and X[2][i]==1):
                return 'X'
            if(O[0][i]==1 and O[1][i]==1 and O[2][i]==1):
                return 'O'
        if(X[0][0]==1 and X[1][1]==1 and X[2][2]==1):
            return 'X'
        if(X[2][0]==1 and X[1][1]==1 and X[0][2]==1):
            return 'X'
        if(O[0][0]==1 and O[1][1]==1 and O[2][2]==1):
            return 'O'
        if(O[2][0]==1 and O[1][1]==1 and O[0][2]==1):
            return 'O'
        if(np.count_nonzero(self.state)>=9):
            return 'D'
        return 'N'

    def reset(self):
        self.state = np.zeros(18)
    # Returns a random valid move in the current boardstate
    def random_move(self):
        # We need to know who we're making a move for.
        # We can determine this by counting the number of X/Os
        X = self.state[:9]
        X = X.reshape(3,3)
        O = self.state[9:]
        O = O.reshape(3,3)
        taken = O+X
        nz = np.count_nonzero(taken)
        xnz = np.count_nonzero(X)
        onz = np.count_nonzero(O)

        # Randomly arrange a move list and then see which fits.
        if(xnz<=onz):
            # Move as X
            moves = np.arange(9)
        else:
            # Move as O
            moves = np.arange(9,18)

        np.random.shuffle(moves)

        for move in moves:
            if(self.is_valid_move(move)):
                mv = np.zeros(18)
                mv[move]=1
                return mv

    # Generates a randomized board state with 0-4 moves already input. 
    def randomize(self):
        n_moves = random.randint(0,4) # number of moves into a game we are
        self.state = np.zeros(18)
        if(n_moves==0):
            return self.state
        for i in range(n_moves):
            move = self.random_move()
            self.state+=move
        return self.state
    # Displays the board
    def display(self):
        X = self.state[:9]
        print(X)
        X = X.reshape(3,3)
        O = self.state[9:]
        print(O)
        O = O.reshape(3,3)
        board = ['.','.','.']
        board = [board.copy(),board.copy(),board.copy()]
        for i in range(9):
            x = int(i/3)
            y = i%3
            if(self.state[i]==1):
                board[x][y] = 'X'
            elif(self.state[i+9]==1):
                board[x][y] = 'O'

        for line in board:
            print(line)
    def is_valid_move(self,move):
        if(move<0 or move>17):
            return False
        if(move<=8):
            if(self.state[move]==0 and self.state[move+9]==0):
                return True
            else:
                return False
        else:
            if(self.state[move]==0 and self.state[move-9]==0):
               return True
            else:
               return False

    def _order_prediction(self,prediction):
        moves = []
        while(len(moves)<9):
            p = np.argmax(prediction)
            prediction[p] = 0
            # prediction = np.delete(prediction, p) # remove the index at p
            moves.append(p)
        return moves

    def get_first_valid_move(self,prediction):
        moves = self._order_prediction(prediction)
        # print("Moves")
        # print(moves)
        for move in moves:
            if(self.is_valid_move(move)):
                return move
            
class Chess_Env:
    def __init__(self):
        pass
