# ADAS_using_DQN
# Creating an AI Agent with Level 2 ADAS Integrated Using CARLA Simulator, DQN, LSTM, and Replay Buffer

## 🟣 Project Overview

This project aims to develop an **AI agent** that can perform **Level 2 ADAS** functionality — including lane-keeping, obstacle avoidance, and adaptive cruise control — within a realistic, scalable, and risk-free **CARLA** simulation environment.

To achieve this, we employ:
- **Deep Q-Network (DQN)** for policy learning.
- **LSTM (Long Short-Term Memory)** to handle temporal dependency in sensor data.
- **Replay Buffer** to stabilize training by sampling past experiences.

---

## 🟣 Features

- **CARLA Simulator:** Provide realistic driving scenarios for training and testing.
- **DQN Algorithm:** Enables the agent to learn an optimal policy from trial-and-error.
- **LSTM Layers:** Provide memory of past states and actions.
- **Experience Replay:** Stores and reuses past transitions to improve sample efficiency.

---

## ⚠ Disclaimer

➥ The **CARLA Simulator must be installed and running on your workstation** for this code to execute.  
➥ Download and install CARLA from the official repository: [https://github.com/carla-simulator/carla](https://github.com/carla-simulator/carla)  
➥ Make sure you **launch the CARLA server first** (using `CarlaUE4.sh`) **before running any code**.

---

## 🟣 Installation

1. **Clone this repository:**
```bash
git clone https://github.com/your-username/adas-carla-dqn.git
cd adas-carla-dqn


## 🟣 Installation

1. **Clone this repository:**
```bash
git clone https://github.com/your-username/adas-carla-dqn.git
cd adas-carla-dqn
