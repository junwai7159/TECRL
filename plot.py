import torch
import numpy as np
from tqdm import tqdm
from model.ppo import PPO
from envs.pedsim import Pedsim
from utils.utils import get_args, set_seed, pack_state, mod2pi
from utils.metrics import calc_TEC, find_CAP


if __name__ == '__main__':
    ARGS = get_args(
        # ENV=dict(type=str, default='./dataset/GC/GC_Dataset_ped1-12685_time1560-1620_interp9_xrange5-25_yrange15-35.npy'),
        ENV=dict(type=str, default='./dataset/UCY/UCY_Dataset_time162-216_timeunit0.08.npy'),
    )
    set_seed(ARGS.SEED)

    # init model
    model = PPO(ARGS).to(ARGS.DEVICE)
    # model.load_state_dict(torch.load(ARGS.LOAD_MODEL, map_location=torch.device(ARGS.DEVICE)))

    # load dataset
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

    velocity = np.diff(position, axis=1, prepend=position[:, (0,), :]) / meta_data['time_unit']
    velocity[:, into_time, :] = np.roll(velocity, shift=-1, axis=1)[:, into_time, :]
    velocity = torch.tensor(velocity)
    # velocity = position.diff(dim=1, prepend=position[:, (0,), :]) / meta_data['time_unit']  # (N, T, 2)
    # velocity[:, into_time, :] = velocity.roll(shifts=-1, dims=1)[:, into_time, :]
    one_step_flag = (into_time + 1 == exit_time)
    velocity[one_step_flag, into_time[one_step_flag], :] = 0.

    destination = torch.FloatTensor(des)[:, 0, :2]  # (N, 2)
    obstacle = torch.FloatTensor(obs)   # (M, 2)
    env_real = Pedsim(ARGS)
    env_real.init_ped(position, velocity, destination)

    
    # Plot dataset
    # import matplotlib.pyplot as plt
    # import matplotlib.animation as animation
    
    # plt.figure()
    # for i in range(N):
    #     plt.plot(position[i, :, 0], position[i, :, 1])

    # plt.show()

    # KDE
    from matplotlib import pyplot as plt
    from sklearn.neighbors import KernelDensity

    v = velocity.reshape(-1, 2)
    mask = ~torch.any(torch.isnan(v), dim=1)
    v = v[mask]
    speed = torch.norm(v, dim=1).unsqueeze(dim=1)
    X_plot = np.linspace(-3, 3, 1000)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(speed)
    log_dens = kde.score_samples(X_plot)

    plt.figure()
    plt.fill(X_plot[:, 0], np.exp(log_dens))
    plt.show()
