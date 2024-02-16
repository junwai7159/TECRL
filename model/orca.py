import torch
import numpy as np
import rvo2

class ORCA(torch.nn.Module):
  def __init__(self, env, ARGS):
    super(ORCA, self).__init__()
    self.env = env
    self.ARGS = ARGS
    self.N = self.env.num_pedestrians
    self.M = self.env.num_obstacles
    self.simulator = self.init_simulator()
    self.rvo_obstacles = self.init_obstacle()
    self.rvo_agents = self.init_state()  

  def init_state(self):
    rvo_agents = [None] * self.N
    
    for agent_id in range(self.N):
      p = self.env.position[agent_id, -1, :]
      g = self.env.destination[agent_id, :]
      v_p = (g - p) / torch.norm(g - p)

      rvo_agents[agent_id] = self.simulator.addAgent(tuple(p.tolist()))
      self.simulator.setAgentPrefVelocity(rvo_agents[agent_id], tuple(v_p.tolist()))

    return rvo_agents

  def init_obstacle(self):
    rvo_obstacles = [None] * self.M
    half_side = self.env.obstacle_radius * np.sqrt(2) / 2

    for obs_id in range(self.M):
      x, y = self.env.obstacle[obs_id].tolist()

      top_left_vertex = (x - half_side, y + half_side)
      top_right_vertex = (x + half_side, y + half_side)
      bottom_left_vertex = (x - half_side, y - half_side)
      bottom_right_vertex = (x + half_side, y - half_side)
      
      rvo_obstacles[obs_id] = self.simulator.addObstacle([top_left_vertex, top_right_vertex, bottom_left_vertex, bottom_right_vertex])

    self.simulator.processObstacles()

    return rvo_obstacles

  def init_simulator(self):
    simulator = rvo2.PyRVOSimulator(timeStep=0.15, neighborDist=1.5, maxNeighbors=self.ARGS.NUM_PED,
                                    timeHorizon=1.5, timeHorizonObst=10, radius=self.env.ped_radius, maxSpeed=2.0) 
    return simulator
    
  def forward(self):
    self.simulator.doStep()
    
    position_, velocity_ = torch.zeros(self.N, 2), torch.zeros(self.N, 2)
    for agent_id in range(self.N):
      p = torch.tensor(self.simulator.getAgentPosition(agent_id))
      g = self.env.destination[agent_id, :]
      v = torch.tensor(self.simulator.getAgentVelocity(agent_id))
      position_[agent_id, :] = p
      velocity_[agent_id, :] = v

      v_p = ((g - p) / torch.norm(g - p) if torch.norm(g - p) > 1.0 else (g - p)) * 1.33
      self.simulator.setAgentPrefVelocity(self.rvo_agents[agent_id], tuple(v_p.tolist()))

    arrive_flag_ = torch.where(self.env.mask[:, -1], torch.norm(self.env.position[:, -1, :] - self.env.destination, dim=-1) 
                               < self.env.ped_radius, self.env.arrive_flag[:, -1])
    mask_ = ~arrive_flag_
    
    position_[~mask_] = float('nan')
    velocity_[~mask_] = float('nan')
    direction_ = torch.atan2(velocity_[:, 1], velocity_[:, 0]).unsqueeze(1)  # (N, 1)

    self.env.position = torch.cat([self.env.position, position_.unsqueeze(1)], dim=1) # (N, T, 2)
    self.env.velocity = torch.cat([self.env.velocity, velocity_.unsqueeze(1)], dim=1) # (N, T, 2)
    self.env.arrive_flag = torch.cat([self.env.arrive_flag, arrive_flag_.unsqueeze(1)], dim=1)  # (N,T)
    self.env.mask = torch.cat([self.env.mask, mask_.unsqueeze(1)], dim=1) # (N, T)
    self.env.direction = torch.cat([self.env.direction, direction_.unsqueeze(1)], dim=1)  # (N, T)
    
    self.env.num_steps += 1

    return None
