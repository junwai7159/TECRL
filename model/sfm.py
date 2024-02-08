import torch
import pysocialforce as psf
from pathlib import Path

class SFM(torch.nn.Module):
  def __init__(self, env, ARGS):
    super(SFM, self).__init__()
    self.env = env
    self.ARGS = ARGS
    self.initial_state = self.init_state()  # (N, 6)
    self.simulator = self.init_simulator()

  def init_state(self):
    N = self.env.position.shape[0]
    # (px, py, vx, vy, gx, gy)
    initial_state = torch.zeros((N, 6))
    initial_state[:, 0:2] = self.env.position[:, -1, :]
    initial_state[:, 2:4] = torch.ones((N, 2))
    initial_state[:, 4:6] = self.env.destination

    return initial_state
  
  def init_simulator(self):
    simulator = psf.Simulator(self.initial_state,
                              groups=None,
                              obstacles=None,
                              config_file=Path(__file__).resolve().parent.joinpath("sfm.toml"))
    
    return simulator
  
  def forward(self, index=-1):
    # (x, y, v_x, v_y, d_x, d_y, [tau])
    self.simulator.step_once()

    state = torch.tensor(self.simulator.get_states()[0], dtype=torch.float32).permute(1, 0, 2)  # (N, T, 7)
    position_ = state[:, -1, 0:2] # (N, 2)
    velocity_ = state[:, -1, 2:4] # (N, 2)
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
    return state
  
  def plot(self):
    with psf.plot.SceneVisualizer(self.simulator, "images") as sv:
      # sv.animate()
      sv.plot()