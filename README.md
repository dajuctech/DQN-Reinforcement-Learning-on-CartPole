# DQN Reinforcement Learning on CartPole

This project implements a reinforcement learning solution using a Deep Q-Network (DQN) to solve the CartPole environment. The code covers environment setup, hyperparameter configuration, agent creation, data collection with a replay buffer, training, and evaluation. It leverages TensorFlow, tf-agents, and Reverb to build and train the agent.

---

## 1. Environment and Setup

- **System and Library Installation:**  
  The script installs system dependencies (e.g., `xvfb`, `ffmpeg`) and Python packages including `tf-agents[reverb]`, `pyglet`, and `dm-reverb[tensorflow]` to prepare the environment for reinforcement learning.

- **Configuration:**  
  Sets a higher recursion limit and uses a virtual display to support rendering of the Gym environment. The code also specifies a consistent random seed for reproducibility.

- **Hyperparameters:**  
  Key hyperparameters are defined such as:
  - Learning rate: `1e-3`
  - Number of training iterations: `3000`
  - Batch size: `264`
  - Replay buffer maximum length: `100000`
  - Logging and evaluation intervals

---

## 2. Environment and Agent Initialization

- **Environment Loading:**  
  The CartPole environment is loaded using OpenAI Gym and then wrapped into TensorFlow-friendly environments for training and evaluation.

- **Agent Setup:**  
  A Q-network is built using several dense layers. The DQN agent is initialized with this network, an Adam optimizer, and a squared loss function for temporal difference errors.

---

## 3. Replay Buffer and Data Collection

- **Replay Buffer Configuration:**  
  A Reverb replay buffer is set up with a uniform sampling strategy and FIFO removal policy. This buffer stores the agent’s experiences for training.

- **Data Collection:**  
  The agent collects initial experience from the environment using a driver with a random policy to populate the replay buffer before training begins.

---

## 4. Training Loop

- **Training Process:**  
  The agent is trained over multiple iterations. In each iteration:
  - New experiences are collected using the agent’s policy.
  - A batch of experiences is sampled from the replay buffer.
  - The agent is trained using these experiences.
  - Training loss and performance metrics are logged at regular intervals.

- **Performance Evaluation:**  
  Periodically, the agent’s average return over several evaluation episodes is computed and printed. This helps track the learning progress.

- **Visualization:**  
  The training progress is visualized by plotting the average return against training iterations. Additionally, a video of the trained agent’s performance is embedded for qualitative evaluation.

---

## 5. Results

- **Return Analysis:**  
  The code includes logic to determine when the agent consistently achieves high returns, indicating successful learning.

- **Video Embedding:**  
  A helper function creates an MP4 video of the agent’s behavior during evaluation, which is embedded directly in the notebook for easy review.

---

## Conclusion

This project demonstrates a complete reinforcement learning pipeline using a DQN agent to balance a CartPole. The approach integrates environment configuration, replay buffer management, data collection, agent training, and performance evaluation. Its modular design allows for straightforward hyperparameter tuning and further experimentation with different RL strategies.

