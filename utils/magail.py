import torch
import multiprocessing
from torch.utils.data import Dataset, DataLoader
import numpy as np

from envs.pedsim import Pedsim
from utils.utils import get_args, set_seed, pack_state

class ExpertDataset(Dataset):
  def __init__(self, state, action):
    self.state = state
    self.action = action

def load_expert_data():
  ARGS = get_args(
    ENV=dict(type=str, default='./dataset/GC/GC_Dataset_ped1-12685_time1560-1620_interp9_xrange5-25_yrange15-35.npy'),
    # ENV=dict(type=str, default='./dataset/UCY/UCY_Dataset_time162-216_timeunit0.08.npy'),
  )
    
  set_seed(ARGS.SEED)

  # load dataset to env_real
  meta_data, trajectoris, des, obs = np.load(ARGS.ENV, allow_pickle=True)
  N = len(trajectoris)
  T = np.max([t for traj in trajectoris for x, y, t in traj]) + 1
  position = torch.full((N, T, 2), float('nan'), device=ARGS.DEVICE)  # (N, T, 2)
  for p, traj in enumerate(trajectoris):
      for x, y, t in traj:
          position[p, t, 0] = x
          position[p, t, 1] = y
  nan_flag = position.isnan().any(dim=-1)  # (N, T)
  into_env = (~nan_flag) & (nan_flag.roll(shifts=1, dims=1))  # (N, T)
  exit_env = (nan_flag) & (~nan_flag.roll(shifts=1, dims=1))  # (N, T)
  into_env[nan_flag.logical_not().all(dim=-1), 0] = True
  exit_env[nan_flag.logical_not().all(dim=-1), 0] = True
  assert (into_env.sum(dim=1) == 1).all() and (exit_env.sum(dim=1) == 1).all(), "A pedestrian enter the env for more/less than 1 times!"
  time = torch.arange(T, device=ARGS.DEVICE)
  into_time = torch.masked_select(time, into_env)  # (N,)
  exit_time = torch.masked_select(time, exit_env)  # (N,)
  exit_time[exit_time == 0] = T

  velocity = position.diff(dim=1, prepend=position[:, (0,), :]) / meta_data['time_unit']  # (N, T, 2)
  velocity[:, into_time, :] = velocity.roll(shifts=-1, dims=1)[:, into_time, :]
  one_step_flag = (into_time + 1 == exit_time)
  velocity[one_step_flag, into_time[one_step_flag], :] = 0.

  destination = torch.FloatTensor(des)[:, 0, :2]  # (N, 2)
  obstacle = torch.FloatTensor(obs)   # (M, 2)
  env_real = Pedsim(ARGS)
  env_real.init_ped(position, velocity, destination)

  # load to expert_dataset  
  expert_state = torch.zeros((N, T, 169))
  expert_action = torch.zeros((N, T, 2))

  for i in range(T):
    s_self, s_int, s_ext = env_real.get_state(index=i)
    state_t = pack_state(s_self, s_int, s_ext)  # (N, 169)
    expert_state[:, i, :] = state_t

    w0 = env_real.direction[:, i, 0]
    v = torch.norm(env_real.velocity[:, i, :], dim=-1)

  
  expert_dataset = ExpertDataset(state=expert_state, action=expert_action)
  expert_dataset_loader = DataLoader(dataset=expert_dataset,
                                     batch_size=4,
                                     shuffle=False,
                                     num_workers=multiprocessing.cpu_count() // 2)
  