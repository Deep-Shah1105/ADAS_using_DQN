import carla
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import logging
import pickle
import json
from collections import deque
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# DQN Network
class DQN(nn.Module):
    def __init__(self, action_size, sequence_length=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(sequence_length, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.lstm = nn.LSTM(512, 256, batch_first=True)
        self.fc2 = nn.Linear(256, action_size)
        self.sequence_length = sequence_length
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        x = x.squeeze(2)
        x = x.view(batch_size, self.sequence_length, 84, 84)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        x = torch.relu(self.fc1(x))
        x = x.view(batch_size, 1, 512)
        x, hidden = self.lstm(x, hidden)
        x = x[:, -1, :]
        x = self.fc2(x)
        return x, hidden

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done
    
    def __len__(self):
        return len(self.buffer)

# CARLA Environment with Updated Reward Function
class CarlaEnv:
    def __init__(self, host='localhost', port=2000, map_name='Town04'):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(map_name)
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None
        self.camera = None
        self.lidar = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.image_queue = deque(maxlen=4)
        self.lidar_data = None
        self.lane_invasions = []
        self.junction_lane_invasions = []
        self.collision = False
        self.sequence_length = 4
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        self.episode_start_time = None
        self.max_episode_duration = 50.0
        self.was_in_junction = False
    
    def reset(self):
        if self.vehicle:
            self.vehicle.destroy()
        if self.camera:
            self.camera.destroy()
        if self.lidar:
            self.lidar.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()
        self.image_queue.clear()
        self.lidar_data = None
        self.lane_invasions = []
        self.junction_lane_invasions = []
        self.collision = False
        self.episode_start_time = time.time()
        self.was_in_junction = False
        
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_point = random.choice(self.spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '84')
        camera_bp.set_attribute('image_size_y', '84')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda image: self.process_image(image))
        
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('channels', '32')
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.0))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.lidar.listen(lambda data: self.process_lidar(data))
        
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.on_collision(event))
        
        lane_invasion_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(lambda event: self.on_lane_invasion(event))
        
        for _ in range(self.sequence_length):
            self.image_queue.append(np.zeros((84, 84), dtype=np.float32))
        
        self.world.tick()
        
        state = np.array(list(self.image_queue)[:self.sequence_length], dtype=np.float32)[:, np.newaxis, :, :]
        assert state.shape == (4, 1, 84, 84), f"Invalid reset state shape: {state.shape}"
        return state
    
    def process_image(self, image):
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            array = array[:, :, :3]
            gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            gray = gray / 255.0
            self.image_queue.append(gray)
        except Exception as e:
            pass
    
    def process_lidar(self, data):
        try:
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            if points.shape[1] >= 3:
                self.lidar_data = points.copy()
        except Exception as e:
            pass
    
    def on_collision(self, event):
        self.collision = True
    
    def on_lane_invasion(self, event):
        lane_types = event.crossed_lane_markings
        invasion = [str(lane_type) for lane_type in lane_types]
        self.lane_invasions.append(invasion)
        if self.world.get_map().get_waypoint(self.vehicle.get_location()).is_junction:
            self.junction_lane_invasions.append(invasion)
    
    def step(self, action, steps):
        steering_idx = action // 3
        throttle_idx = action % 3
        steering = (steering_idx - 15) * 0.5 / 15.0
        if throttle_idx == 0:
            throttle, brake = 0.5, 0.0
        elif throttle_idx == 1:
            throttle, brake = 0.0, 0.5
        else:
            throttle, brake = 0.0, 0.0
        
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steering, brake=brake))
        
        self.world.tick()
        
        #if not self.vehicle.get_location().is_valid():
          #  logging.error("Invalid vehicle location, terminating episode")
         #   return None, 0.0, True, {"error": "Invalid vehicle location"}
        
        while len(self.image_queue) < self.sequence_length:
            self.image_queue.append(np.zeros((84, 84), dtype=np.float32))
        
        state = np.array(list(self.image_queue)[:self.sequence_length], dtype=np.float32)[:, np.newaxis, :, :]
        assert state.shape == (4, 1, 84, 84), f"Invalid step state shape: {state.shape}"
        
        reward, done, envdata = self.compute_reward(steering, throttle, steps)
        self.prev_steering = steering
        self.prev_throttle = throttle
        
        if time.time() - self.episode_start_time > self.max_episode_duration:
            done = True
        
        return state, reward, done, envdata
    
    def compute_reward(self, steering, throttle, steps):
        reward = 0.0
        done = False
        
        # Get waypoint and road context
        waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        is_junction = waypoint.is_junction
        next_waypoints = waypoint.next(10.0)
        curvature = 0.0
        if len(next_waypoints) > 1:
            v1 = np.array([next_waypoints[0].transform.location.x, next_waypoints[0].transform.location.y])
            v2 = np.array([next_waypoints[-1].transform.location.x, next_waypoints[-1].transform.location.y])
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 > 0.1 and norm_v2 > 0.1:
                angle = np.arccos(np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0))
                curvature = angle / 10.0
            else:
                tempval = 0
                logging.debug("Waypoints too close, defaulting curvature to 0.0")
        
        # Dynamic weights (normalized to sum to ~1.0)
        if is_junction:
            w_lane, w_safe, w_speed, w_progress, w_smooth = 0.4, 0.48, 0.04, 0.04, 0.04
        elif curvature > 0.2:
            w_lane, w_safe, w_speed, w_progress, w_smooth = 0.385, 0.462, 0.077, 0.038, 0.038
        else:
            w_lane, w_safe, w_speed, w_progress, w_smooth = 0.296, 0.222, 0.222, 0.037, 0.037
        
        # Lane discipline
        distance_to_center = np.linalg.norm(
            np.array([waypoint.transform.location.x, waypoint.transform.location.y]) -
            np.array([self.vehicle.get_location().x, self.vehicle.get_location().y])
        )
        r_lane = 1.0 - distance_to_center / 2.0
        r_lane = max(-1.0, min(1.0, r_lane))
        if waypoint.lane_type != carla.LaneType.Driving:
            r_lane = -2.0
        lane_penalty = 0.0
        if self.lane_invasions:
            for lane_type in self.lane_invasions[-1]:
                if 'Solid' in lane_type or 'SolidSolid' in lane_type:
                    lane_penalty -= 5.0
                elif 'Broken' in lane_type:
                    lane_penalty -= 2.0
            self.lane_invasions = []
        r_lane += lane_penalty
        
        # Safety
        r_safe = -1.0
        min_distance = float('inf')
        if self.lidar_data is not None and self.lidar_data.shape[0] > 0 and self.lidar_data.shape[1] >= 3:
            try:
                distances = np.sqrt(self.lidar_data[:, 0]**2 + self.lidar_data[:, 1]**2)
                min_distance = np.min(distances)
                r_safe = 3.0 / min_distance if min_distance > 0.1 else -1.0
                r_safe = max(-1.0, min(1.0, r_safe))
            except Exception as e:
                tempval = 0
                #logging.warning(f"LIDAR processing error: {e}, defaulting r_safe to -1.0")
        else:
            tempval = 0
            #logging.warning("LIDAR data unavailable, defaulting r_safe to -1.0")
        
        # Speed
        speed = np.linalg.norm([self.vehicle.get_velocity().x, self.vehicle.get_velocity().y])
        target_speed = 3.0 if is_junction else (5.0 if curvature > 0.2 else 8.0)
        r_speed = (2.0 - abs((speed - target_speed)*2)*2)
        if speed > 12.0:
            r_speed -= 2.0
        
        # Progress
        r_progress = 0.05 * speed
        r_progress = min(r_progress, 0.5)
        
        # Smoothness
        r_smooth = -0.1 * abs(steering - self.prev_steering) - 0.05 * abs(throttle - self.prev_throttle)
        
        # Collision
        r_collision = 0.0
        if self.collision:
            r_collision = -400.0
            done = True
        
        # Crossroad navigation with lane discipline
        r_crossroad = 0.0
        exploration_factor = min(1.0, steps / 10000.0)  # Soften penalties early
        if is_junction:
            if speed < 2.0 or speed > 4.0:  # Penalize unsafe speed
                r_crossroad = -0.5 * exploration_factor
            if distance_to_center > 1.5:  # Penalize lane deviation
                r_crossroad = -1.0 * exploration_factor
            if self.junction_lane_invasions:  # Penalize lane invasions
                for lane_type in self.junction_lane_invasions[-1]:
                    if 'Solid' in lane_type or 'SolidSolid' in lane_type:
                        r_crossroad = -5.0 * exploration_factor
                    elif 'Broken' in lane_type:
                        r_crossroad = -2.0 * exploration_factor
                self.junction_lane_invasions = []
            self.was_in_junction = True
        elif self.was_in_junction and not self.collision:  # Exited junction
            if distance_to_center < 1.5 and not any(self.junction_lane_invasions):  # Good lane discipline
                r_crossroad = 0.5
            self.was_in_junction = False
            self.junction_lane_invasions = []
        
        reward = (w_lane * r_lane + w_safe * r_safe + w_speed * r_speed +
                  w_progress * r_progress + w_smooth * r_smooth + r_collision + r_crossroad)
        if steps % 1000 == 0:
            logging.debug(f"Rewards: r_safe={r_safe:.2f}, r_lane={r_lane:.2f}, r_speed={r_speed:.2f}, "
                          f"r_progress={r_progress:.2f}, r_smooth={r_smooth:.2f}, "
                          f"r_collision={r_collision:.2f}, r_crossroad={r_crossroad:.2f}, total={reward:.2f}")
        
        # Return envdata for logging
        envdata = {
            'step': steps,
            'pos_x': self.vehicle.get_location().x,
            'pos_y': self.vehicle.get_location().y,
            'speed': speed,
            'steering': steering,
            'throttle': throttle,
            'distance_to_center': distance_to_center,
            'is_junction': is_junction,
            'junction_lane_invasions': str(self.junction_lane_invasions),
            'min_distance': min_distance,
            'reward': reward
        }
        return reward, done, envdata

# Training Function
def train_dqn():
    hyperparams = {
        'action_size': 31 * 3,  # 93 actions
        'sequence_length': 4,
        'buffer_size': 50000,
        'batch_size': 128,
        'gamma': 0.99,
        'tau': 0.01,
        'lr': 0.0005,
        'update_freq': 2,
        'max_episodes': 2500,
        'save_freq': 200,
        'warmup_steps': 5000,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 80000
    }
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = CarlaEnv(map_name='Town04')
    settings = env.world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    env.world.apply_settings(settings)
    
    policy_net = DQN(hyperparams['action_size'], hyperparams['sequence_length'])
    target_net = DQN(hyperparams['action_size'], hyperparams['sequence_length'])
    
    run_id = "20250502_110644"
    os.makedirs(f"output/{run_id}/checkpoints", exist_ok=True)
    os.makedirs(f"output/{run_id}/metrics", exist_ok=True)
    os.makedirs(f"output/{run_id}/replay_buffer", exist_ok=True)
    os.makedirs(f"output/{run_id}/envdata", exist_ok=True)
    
    # Save hyperparameters
    with open(f"output/{run_id}/hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=4)
    
    checkpoint_path = None
    start_episode = 0
    if os.path.exists(f"output/{run_id}/checkpoints"):
        checkpoints = sorted([f for f in os.listdir(f"output/{run_id}/checkpoints") if f.startswith("policy_net_ep")])
        if checkpoints:
            checkpoint_path = f"output/{run_id}/checkpoints/{checkpoints[-5]}"
            start_episode = int(checkpoint_path.split("ep")[-1].split(".")[0])
            policy_net.load_state_dict(torch.load(checkpoint_path))
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Resumed from checkpoint: {checkpoint_path}, starting at episode {start_episode}")
    else:
        target_net.load_state_dict(policy_net.state_dict())
    
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=hyperparams['lr'])
    memory = ReplayBuffer(hyperparams['buffer_size'])
    
    metrics = {
        "episode": [],
        "reward": [],
        "lane_deviation": [],
        "speed": [],
        "collisions": [],
        "crossroad_success": [],
        "lane_invasions": []
    }
    
    q_values_log = {
        "step": [],
        "action": [],
        "q_value": []
    }
    
    steps = 0
    epsilon = hyperparams['epsilon_start']
    
    for episode in range(start_episode, hyperparams['max_episodes']):
        state = env.reset()
        state = torch.FloatTensor(state)
        hidden = None
        episode_reward = 0
        collisions = 0
        lane_deviations = []
        speeds = []
        crossroad_success = 0
        lane_invasion_count = 0
        envdata_log = []
        
        while True:
            if random.random() < epsilon:
                action = random.randrange(hyperparams['action_size'])
                q_value = None
            else:
                with torch.no_grad():
                    q_values, hidden = policy_net(state.unsqueeze(0), hidden)
                    action = q_values.max(1)[1].item()
                    q_value = q_values[0, action].item()
            
            if steps % 100 == 0 and q_value is not None:
                logging.debug(f"Step {steps}, Action {action}, Q-value {q_value:.2f}")
                q_values_log["step"].append(steps)
                q_values_log["action"].append(action)
                q_values_log["q_value"].append(q_value)
            
            next_state, reward, done, envdata = env.step(action, steps)
            if next_state is None:  # Handle invalid state
                break
            next_state = torch.FloatTensor(next_state)
            episode_reward += reward
            
            memory.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
            
            waypoint = env.world.get_map().get_waypoint(env.vehicle.get_location())
            distance_to_center = np.linalg.norm(
                np.array([waypoint.transform.location.x, waypoint.transform.location.y]) -
                np.array([env.vehicle.get_location().x, env.vehicle.get_location().y])
            )
            lane_deviations.append(distance_to_center)
            speed = np.linalg.norm([env.vehicle.get_velocity().x, env.vehicle.get_velocity().y])
            speeds.append(speed)
            if env.collision:
                collisions += 1
            if waypoint.is_junction and not env.collision and distance_to_center < 1.5:
                crossroad_success += 1
            if env.lane_invasions:
                lane_invasion_count += len(env.lane_invasions)
                env.lane_invasions = []
            
            envdata_log.append(envdata)
            
            if len(memory) > hyperparams['batch_size'] and steps > hyperparams['warmup_steps']:
                states, actions, rewards, next_states, dones = memory.sample(hyperparams['batch_size'])
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)
                
                q_values, _ = policy_net(states, None)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                
                next_q_values, _ = target_net(next_states, None)
                target_q_values = rewards + (1 - dones) * hyperparams['gamma'] * next_q_values.max(1)[0]
                
                loss = nn.MSELoss()(q_values, target_q_values.detach())
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()
                
                if steps % hyperparams['update_freq'] == 0:
                    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                        target_param.data.copy_(hyperparams['tau'] * policy_param.data + (1.0 - hyperparams['tau']) * target_param.data)
            
            state = next_state
            steps += 1
            epsilon = hyperparams['epsilon_end'] + (hyperparams['epsilon_start'] - hyperparams['epsilon_end']) * np.exp(-steps / hyperparams['epsilon_decay'])
            
            if done:
                if env.collision:
                    print(f"Episode {episode} ended due to collision")
                elif time.time() - env.episode_start_time > env.max_episode_duration:
                    print(f"Episode {episode} ended due to max duration ({env.max_episode_duration} seconds)")
                else:
                    print(f"Episode {episode} ended unexpectedly: {envdata.get('error', 'Unknown reason')}")
                break
        
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Lane Deviation: {np.mean(lane_deviations):.2f}, "
              f"Speed: {np.mean(speeds):.2f}, Collisions: {collisions}, Crossroad Success: {crossroad_success}, "
              f"Lane Invasions: {lane_invasion_count}, epsilon: {epsilon:.2f}")
        
        metrics["episode"].append(episode)
        metrics["reward"].append(episode_reward)
        metrics["lane_deviation"].append(np.mean(lane_deviations))
        metrics["speed"].append(np.mean(speeds))
        metrics["collisions"].append(collisions)
        metrics["crossroad_success"].append(crossroad_success)
        metrics["lane_invasions"].append(lane_invasion_count)
        
        if episode % hyperparams['save_freq'] == 0 or episode == hyperparams['max_episodes'] - 1:
            torch.save(policy_net.state_dict(), f"output/{run_id}/checkpoints/policy_net_ep{episode}.pth")
            pd.DataFrame(metrics).to_csv(f"output/{run_id}/metrics/metrics_ep{episode}.csv", index=False)
            pd.DataFrame(q_values_log).to_csv(f"output/{run_id}/metrics/q_values_ep{episode}.csv", index=False)
            pd.DataFrame(envdata_log).to_csv(f"output/{run_id}/envdata/envdata_ep{episode}.csv", index=False)
            with open(f"output/{run_id}/replay_buffer/buffer_ep{episode}.pkl", "wb") as f:
                pickle.dump(list(memory.buffer), f)
            plt.figure(figsize=(10, 5))
            plt.plot(metrics["episode"], metrics["reward"])
            plt.title("Episode Reward")
            plt.savefig(f"output/{run_id}/metrics/reward_ep{episode}.png")
            plt.close()

if __name__ == "__main__":
    train_dqn()