from numpy import array
import numpy as np
# Parses the game data from 3 lines of text. 
# Input 3 lines of strings that make up every game state. ex
# X..    XO.
# ...    ...
# ...    ... ...and so on
# An array of the game notation
def parse_game_data(data):
    lines = []
    lines.append(data[0].split(' '))
    lines.append(data[1].split(' '))
    lines.append(data[2].split(' '))

    # Remove white space & new lines
    for line in lines:
        line[:] = [value for value in line if value != '']
        line[:] = [value for value in line if value !='\n']
    state_count = len(lines[0]) # Should theoretically be the number of array elements.

    game_notation = []
    # For each state, flatten the state into 1 string
    # then extract notation
    # Append to game notation
    p_state = ""
    for i in range(0,state_count):
        n_state = flatten([lines[0][i], lines[1][i], lines[2][i]])
#        move = state_diff(p_state, n_state)
 #       move = n_state
        move = gen_notation(n_state)
        p_state = n_state
        game_notation+=move
    game_notation = game_notation
    return game_notation

def gen_notation(c_state):
    xarr = [0,0,0,
            0,0,0,
            0,0,0]
    oarr = [0,0,0,
            0,0,0,
            0,0,0]
    for i in range(0,9):
        if(c_state[i]=='X'):
            xarr[i]=1
        elif(c_state[i]=='O'):
            oarr[i]=1
    return xarr + oarr

# Takes in two game states, and figures out the difference between the two. Returns the notational difference.
# p_state = previous state
# n_state = new state
# Input notation
# X..
# ...
# ..O
# Outputs string notation of the move difference
def state_diff(p_state, n_state):
    # For each character in the new state, check if it's equivalent to the previous state. If it's not, extract notation
    n_size = len(n_state)
    p_size = len(p_state)
    pos = (0,0)
    piece = ""
    for i in range(0, n_size):
        if(p_size==n_size):
            if(n_state[i]!=p_state[i]):
              x = i // 3
              y = i % 3
              pos=(x,y)
              piece = n_state[i]
              break
        else:
            if(n_state[i]!='.'):
              x = i // 3
              y = i % 3
              pos=(x,y)
              piece = n_state[i]
              break
    row = ""
    col = ""
    move = [ ]
    if(piece=="X"):
        move.append(1)
    else:
        move.append(0)
    if(pos[0]==0):
        row = 1
    elif(pos[0]==1):
        row = 0.5 # prev 0
    elif(pos[0]==2):
        row = 0 # prev -1
    else:
        print("This should never happen, if it does, something got fucked on import")
    if(pos[1]==0):
        col = 0 # prev -1
    elif(pos[1]==1):
        col = 0.5 # prev 0
    elif(pos[1]==2):
        col = 1
    else:
        print("This should never happen, if it does, something got fucked on import")
    move.append(row)
    move.append(col)
    return move
    # if(pos[0]==0):
    #     row = "T"
    # elif(pos[0]==1):
    #     row = "C"
    # elif(pos[0]==2):
    #     row = "B"
    # else:
    #     print("This should never happen, if it does, something got fucked on import")
    # if(pos[1]==0):
    #     col = "L"
    # elif(pos[1]==1):
    #     col = "C"
    # elif(pos[1]==2):
    #     col = "R"
    # else:
    #     print("This should never happen, if it does, something got fucked on import")
    # move += row
    # move += col
    # return move
# Loads the tic tac toe database file into a useable format.
# Input - Filename
# Output - Array of games, which are arrays of game states.
# Output - Dictionary of games.
# Key is the winner "Nought" or "Cross"
# # The data inside the dict is 
# STATE = 0 searching for a game
# STATE = 1 reading game data

def load_ttt_db(filename):
    file = open(filename,"r")
    lines = file.readlines()

    STATE = 0 # State of the parser
    ln = 0 # Line number
    games = {5:([],[]), 6:([],[]), 7:([],[]), 8:([],[]), 9:([],[])}

    while(ln!=len(lines)):
        if(STATE==0):
            if(lines[ln][0].isdigit()):
                STATE = 1 # Change state to parser for the game input.
            ln+=1
        if(STATE==1):
            game_info = lines[ln-1].split(' ')
            game_winner = game_info[1]
            game_data = [ lines[ln+1], lines[ln+2], lines[ln+3]]
            game_notation = parse_game_data(game_data)
            state_num = len(game_notation)//18
            games[state_num][0].append(game_winner)
            games[state_num][1].append(game_notation)
            STATE = 0
    return games

def load_one_hot_db(filename):
    file = open(filename,"r")
    lines = file.readlines()

    STATE = 0 # State of the parser
    ln = 0 # Line number
    o_states = [ ] # Game state db for nought wins
    o_labels = [ ] # Labels for nought wins
    x_states = [ ]
    x_labels = [ ] 

    while(ln!=len(lines)):
        if(STATE==0):
            if(lines[ln][0].isdigit()):
                STATE = 1 # Change state to parser for the game input.
            ln+=1
        if(STATE==1):
            game_info = lines[ln-1].split(' ')
            game_winner = game_info[1]
            game_data = [ lines[ln+1], lines[ln+2], lines[ln+3]]
            game_states, game_moves = parse_to_onehot(game_data)
            if(game_winner=='Noughts'):
                o_states+=(game_states)
                o_labels+=(game_moves)
            elif(game_winner=='Crosses'):
                x_states+=(game_states)
                x_labels+=(game_moves)
            # else:
            #     # draw. Add to both
            #     o_states+=(game_states)
            #     o_labels+=(game_moves)
            #     x_states+=(game_states)
            #     x_labels+=(game_moves)

            STATE = 0
    return (x_states, x_labels), (o_states, o_labels)

# Takes in game notation data in string format, returns a list of vectorized game states and the next moves.
def parse_to_onehot(data):
    lines = []
    lines.append(data[0].split(' '))
    lines.append(data[1].split(' '))
    lines.append(data[2].split(' '))

    # Remove white space & new lines
    for line in lines:
        line[:] = [value for value in line if value != '']
        line[:] = [value for value in line if value !='\n']
    state_count = len(lines[0]) # Should theoretically be the number of array elements.

    game_states = []
    moves = []
    # For each state, flatten the state into 1 string
    # Extract the notational vector form of the game state
    # Compare the state to previous state to create the one-hot label associated with the next move.

    # Our initial state/previous state
    p_state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(0,state_count):
        n_state = flatten([lines[0][i], lines[1][i], lines[2][i]])
        n_state = gen_notation(n_state)
        move = one_hot_diff(p_state,n_state)
        # Add our move and the previous state to our game state & move lists. Indexes will keep them in order.
        if(i%2==0):
            move=move[9:]
            moves.append(move)
            game_states.append(p_state)
        p_state = n_state

    return game_states, moves

def one_hot_diff(p_state, n_state):
    new_list = [ 0 if p_state[i]==n_state[i] else 1 for i in range(18)]
    return new_list


# Flattens a game state to a single string
# Reads in a game state with 3 rows of strings and condenses to 1 string.
# outputs a flattened string
def flatten(game_state):
    s = ""
    for row in game_state:
        s+=row
    return s

def save_ttt_db(games, fmt):
    pass
