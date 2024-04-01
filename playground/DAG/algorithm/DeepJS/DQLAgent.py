import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from collections import deque
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class DQLAgent:
    def __init__(self, state_size, action_size, gamma, name, sequence_length, model=None):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length  # Length of the input sequence
        self.sequence_buffer = deque([], maxlen=2)  # Buffer to hold last 2 state-action pairs
        self.memory = deque(maxlen=2000)  # Replay buffer
        self.gamma = gamma  # Discount rate
        self.epsilon = 0.4  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.name = name
        self.update_frequency = 5  # Train the model every 10 timesteps
        self.timestep_since_last_update = 0  # Counter for timesteps since last training
        # self.summary_path = summary_path if summary_path is not None else './tensorboard/%s--%s' % (
        #     name, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
        # self.summary_writer = tf.contrib.summary.create_file_writer(self.summary_path)
        # self.brain = brain
        # self.checkpoint = tf.train.Checkpoint(brain=self.brain)
        # self.model_save_path = model_save_path
        if model == None:
            self.model = self._build_model()
        else:
            self.model = model

    def _build_model(self):
        """Builds a deep neural network model with LSTM layers."""
        model = Sequential()
        # Assuming state and action are concatenated in the input sequence
        # Input shape [sequence_length, state_size + action_size]
        model.add(LSTM(64, input_shape=(self.sequence_length, self.state_size + self.action_size), return_sequences=False))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # 'linear' for continuous actions or Q-values
        model.compile(loss=MeanSquaredError(), optimizer='adam')
        return model

    def remember(self, sequence, next_state, reward, done):
        self.memory.append((sequence, next_state, reward, done))

    def act(self, current_state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            print("I chode randomly with epsilon value", self.epsilon)
            return np.random.randint(0, self.action_size)

        # Initialize the sequence with zeros
        current_sequence = np.zeros((1, self.sequence_length, self.state_size + self.action_size))
        
        # Fill in the sequence with data from the sequence buffer
        for i, (state, action) in enumerate(self.sequence_buffer):
            current_sequence[0, i, :self.state_size] = state
            current_sequence[0, i, self.state_size:] = action
        
        # Add the current state as the last element in the sequence with a dummy action
        current_sequence[0, -1, :self.state_size] = current_state

        # Predict the action based on the input sequence
        act_values = self.model.predict(current_sequence)
        return np.argmax(act_values[0])

    def update_sequence_buffer(self, state, action):
        # Ensure 'action' is one-hot encoded if it isn't already
        action_vector = np.zeros(self.action_size)
        action_vector[action] = 1
        # Add the state-action pair to the sequence buffer
        self.sequence_buffer.append((state, action_vector))
    
    def prepare_sequence(self, sequence):
        # Assuming 'sequence' is a list of (state, action) tuples
        # Concatenate state and action for each pair and prepare for LSTM input
        formatted_sequence = np.zeros((1, self.sequence_length, self.state_size + self.action_size))
        for i, (state, action) in enumerate(sequence):
            formatted_sequence[0, i, :self.state_size] = state  # Assign state part
            formatted_sequence[0, i, self.state_size:] = action  # Assign action part
        return formatted_sequence

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for sequence, next_state, reward, done in minibatch:
            input_sequence = self.prepare_sequence(sequence)
            target_f = self.model.predict(input_sequence)
            
            if not done:
                next_state_input = np.reshape(next_state, (1, 1, self.state_size))  # Single timestep sequence for next state
                next_state_input = np.concatenate([next_state_input, np.zeros((1, 1, self.action_size))], axis=-1)  # Append dummy action
                next_q_value = np.amax(self.model.predict(next_state_input)[0])
                update_target = reward + self.gamma * next_q_value
            else:
                update_target = reward
            
            # Assuming the last action in the sequence is what we want to update
            action_index = np.argmax(target_f[0])  # This might need adjustment based on how you track actions
            target_f[0][action_index] = update_target

            self.model.fit(input_sequence, target_f, epochs=1, verbose=0)

    def save_model(self):
        model_dir = 'DAG/algorithm/DeepJS/agents/%s' % self.name
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'model.h5')
        self.model.save(model_path)
        # self.model.save(model_dir)

# Example usage within SimPy environment
# agent = DQLAgent(state_size=12, action_size=21)  # Define appropriate sizes
# During each pause:
#   action = agent.act(current_state)
#   apply_action_to_simulation(action)
#   agent.remember(state, action, reward, next_state, done)
#   if len(agent.memory) > batch_size:
#       agent.replay(batch_size)
