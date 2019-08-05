from Model import *
import tensorflow as tf
import random
# The way this should work is
# We play a BUNCH of games and store the game state sequences
# We determine the maxQ value of every game state played
# Then set the label of each game state to be the next state with the highest Q value
# Then train normally

# Deciding how to train.
# We have a collection of every TTT game. We could easily train using just those and run the games through the same process.
# Or we can create something to play the max value based on teh games.
# Or we can have two neural networks fight one another for true dynamic gameplay.

# Opting for option 3 if we have time for it.

# We need a memory class to store our Q values.
# Yoinked this idea & class from http://adventuresinmachinelearning.com/reinforcement-learning-tensorflow/
class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self.samples = []

    def add_sample(self, sample):
        self.samples.append(sample)
        if(len(self.samples) > self._max_memory):
                self.samples.pop(0)
    def sample(self, num):
        if(num > len(self.samples)):
            return random.sample(self.samples, len(self.samples))
        else:
            return random.sample(self.samples, num)
    def clear(self):
        self.samples = []
# So here we're going to have a neural network play versus an environment that randomly samples the next move. 
# Originally I was going to go with two neural networks to play versus one another, but I ran into an issue where I had to decide how to deal with repeating games(since a neural network will always spit out the same output for a given input...so it'd repeat the same game states over and over). 
# We'll store the gamestates of each series of battles in our memory class
# Then we play back through the memory class later and do the actual training. 

# TODO - Later let's add an option to pass in a memory object stuffed with the actual game database samples. This would allow for us to perform a smarter form of supervised learning.
class ReinforcementTrainer:
    def __init__(self, env, model, memory, max_memory):
        self._env = env
        self._model = model
        self._max_memory = max_memory
        self._memory = memory 

    def batch_games(self, sess, count, randomize=False):
        self._memory.clear()
        wins = 0
        losses = 0
        draws = 0
        for i in range(count):
            reward = self.simulate_game(sess, True)
            if(reward==1):
                wins+=1
            elif(reward==0):
                draws+=1
            elif(reward==-1):
                losses+=1
            reward = self.simulate_game(sess, False)
            if(reward==1):
                wins+=1
            elif(reward==0.5):
                draws+=1
            elif(reward==-1):
                losses+=1

        print("Memory Usage: " + str(len(self._memory.samples) / self._max_memory) + "%")
        win_perc = wins/(count*2)
        draw_perc = draws/(count*2)
        loss_perc = losses/(count*2)
        print("Win %: " + str(win_perc))
        print("Draw %: " + str(draw_perc))
        print("Loss %: " + str(loss_perc))
        return win_perc
    # Runs through a random game
    def simulate_game(self,sess, randomize=False):
        if(randomize==True):
            self._env.randomize()
        else:
            self._env.reset()
        # Make 1 random beginning move
        state = self._env.state
        
        # self._env.display()
        # while the game is running, take actions, store the game states. 
        states = []
        # states.append(state.copy())
        next_states = []
        actions = []
        reward = 0
        while True:
            # Choose action & apply it to the current environment state
            next_state, done, reward = self._env.step(self._env.random_move())
            # print("")
            state = next_state
            next_states.append(next_state.copy())
            states.append(state.copy())
            # self._env.display()
            if(done):
                # self._memory.add_sample((states, reward, actions))
                # By adding the samples in this way, we assign reward values to sequence itself. Later we can maximize our winrate by finding sequences that have the highest reward and using those labels.
                for i in range(len(actions)):
                    self._memory.add_sample((states[i], reward, actions[i]))
                break
            
            action = self._choose_action(state,sess)
            actions.append(action[9:])
            next_state, done, reward = self._env.step(action)
            next_states.append(next_state.copy())
            # self._env.display()
            if(done):
                # We can loop over the range of actions since the states will always be equal to or 1 greater. In the case of 1 greater, we don't want the value anyways since it's an ending loss value.
                for i in range(len(actions)):
                    self._memory.add_sample((states[i], reward, actions[i]))
                break
        return reward
    def _choose_action(self,state,sess):
        # print("Choosing an action...")
        data = state.reshape(1,18)
        action = self._model.predict(data,sess)[0]
        # print("Action: " + str(action))
        move = self._env.get_first_valid_move(action)
        # print("Suggested move: " + str(move))
        mv = np.zeros(18)
        mv[move+9] = 1
        return mv
    # This function sorts the batch into a dictionary of states and the rewards of each action coming out.
    # Then finds the max reward and turns that into the singular label coming out of the state.
    # We return a list of condensed states(no dupes) & the most rewarding label coming out
    def generate_reward_labels(self, batch):
        state_dict = {}
        # Create the dictionary of rewards. This would be better as a graph tbh...but running out of time and this is fast enough
        for i, b in enumerate(batch):
            state, reward, action = b[0], b[1], b[2]
            # Need to convert to a list then to a tuple. Since lists and numpy arrays aren't hashable without overriding the class.
            state = tuple(state.tolist())
            if(state in state_dict.keys()):
                found = False
                for act in state_dict[state]:
                    # print(act)
                    # print(action)
                    if((act[0]==action).all()):
                        # print("Compared")
                        # print(act[0])
                        # print(action)
                        found = True
                        act[1]+=reward
                if(found==False):
                    state_dict[state].append([action,reward])
            else:
                state_dict[state] = [[action, reward]]
        
        # Now we're going to create the actual states and labels.
        keys = state_dict.keys()
        states = []
        labels = []
        for key in keys:
            m = max(state_dict[key], key=lambda item: item[1])
            # print("State: " + str(key))
            # print("Number of actions: " + str(len(state_dict[key])))
            # print("Max value action/reward: " + str(m))
            if(m[1]<=0):
                continue
            states.append(list(key))
            labels.append(m[0])
        return states, labels, state_dict
    def train(self):
        batch_size = 1000
        batch = self._memory.sample(batch_size)
        states = np.array([val[0] for val in batch])
        rewards = np.array([val[1] for val in batch])
        actions = np.array([(np.zeros(18) if val[2] is None else val[2]) for val in batch])
        states, labels = self.generate_reward_labels(batch)
        states = np.array(states)
        labels = np.array(labels)
        epochs = 100
        self._model.train(states, labels, epochs=epochs)
