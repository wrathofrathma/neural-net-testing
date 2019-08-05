#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from Model import *
from Environments import TTT_Env
from ttt_db_reader import *
from reinforce import *
from interactive_ttt import *
from supervised import *
import pickle

# Setting up the initial model
model = Model(iosize=[18,9])
model.add_layer([18,72], activation='tanh')
model.add_layer([72,72], activation='tanh')
model.add_layer([72,9], activation='softmax')
model.set_learn_rate(0.03)
model.init_graph()

# Using a model we've already trained
# model = Model.load_from_file("ttt_model")
# model = Model.load_from_file("test_model")
# model.init_graph()
# Sprinkle in randomized states

# Tic Tac Toe Environment
env = TTT_Env()

# memory = load_one_hot_rewards("ttt.db")
# pickle.dump(memory, open("memory.p","wb"))

memory = pickle.load(open("memory.p","rb"))
# memory = Memory(100000)

trainer = ReinforcementTrainer(env=env, model=model, max_memory=100000, memory=memory)
training_states, training_labels, sdict = trainer.generate_reward_labels(memory.sample(100000))
# training_states = np.array(training_states)
# training_labels = np.array(training_labels)
# model.train(training_states, training_labels, epochs=10000, verbose=True)
# auto_bulk_ttt("models/ttt",4, training_states, training_labels)

# model.save_model("test_model2")

# episodes = 10
# with tf.Session() as sess:
#     sess.run(model.init_op)
#     for i in range(episodes):
#         trainer.batch_games(sess,1000, True)
#         trainer.train()
        # model.save_model("test_model2")
def test_memory_agent(state_dict, env):
    games = 1000
    wins = 0
    loss = 0
    draws = 0
    for i in range(games):
        env.reset()
        state = env.state

        while(True):
            next_state, done, reward = env.step(env.random_move())

            state = next_state
            if(done):
                if(reward==1):
                    wins+=1
                elif(reward==0.5):
                    draws+=1
                else:
                    loss=+1
                break
            # choose action based on state
            tstate = state.astype(int).tolist()
            tstate = tuple(tstate)
            for key in state_dict.keys():
                print(tstate)
                print(key)
            if(tstate in state_dict.keys()):
                move = max(state_dict[tstate], key=lambda item: item[1])
                print("found")
            else:
                print("Rand")
                move = env.random_move()

            print(move)
test_memory_agent(sdict, env)
# int_ttt("test_model2")

