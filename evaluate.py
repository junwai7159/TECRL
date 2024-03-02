import torch
import numpy as np
from tqdm import tqdm
import math

from model.ppo import PPO
from model.sfm import SFM
from model.orca import ORCA
from envs.pedsim import Pedsim
from utils.utils import get_args, set_seed, pack_state, mod2pi
from utils.metrics import *
from utils.visualization_cv import generate_gif

if __name__ == '__main__':
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

    # init RL model
    if ARGS.MODEL == 'TECRL':
        model = PPO(ARGS).to(ARGS.DEVICE)
        model.load_state_dict(torch.load(ARGS.LOAD_MODEL, map_location=torch.device(ARGS.DEVICE)))

    # collision avoidance behavior evaluation
    CAP_flag = find_CAP(env_real)   # (N, N, T)
    CAP_start = CAP_flag.clone(); CAP_start[:, :, 1:] &= ~CAP_start[:, :, :-1]
    CAP_final = CAP_flag.clone(); CAP_final[:, :, :-1] &= ~CAP_final[:, :, 1:]
    
    # metrics
    Collision_rates = []
    Displacements = []
    V_locomotions = []
    A_locomotions = []
    Perp_dev_distances = []
    Energys = []
    Steer_energys = []
    ADEs = []
    FDEs = []

    # simulate from the state at time t        
    pbar = tqdm(range(T))
    for t in pbar:
        if not CAP_start[:, :, t].any(): continue
        
        # filter agent from env_real to env_imit
        imit_agent_mask = (~torch.isnan(env_real.position[:, t, :]).any(dim=1)).nonzero().squeeze() # (N_imit,)
        # remove agents who are near the destination
        remove_agent_mask = torch.norm(env_real.position[imit_agent_mask, t, :] - env_real.destination[imit_agent_mask, :], dim=-1) < env_real.ped_radius
        imit_agent_mask = imit_agent_mask[~remove_agent_mask]

        # init env_imit
        env_imit = Pedsim(env_real.args)
        env_imit.meta_data = env_real.meta_data
        env_imit.add_pedestrian(env_real.position[imit_agent_mask, t, :], env_real.velocity[imit_agent_mask, t, :], env_real.destination[imit_agent_mask, :], init=True)
        
        # init sfm/orca model
        if ARGS.MODEL == 'SFM':
            model = SFM(env_imit, ARGS)
        elif ARGS.MODEL == 'ORCA':
            model = ORCA(env_imit, ARGS)
        
        # real_mean_speed = []
        # imit_mean_speed = []
        # simulate from t to t+101 or T
        for s in range(t + 1, min(t + 101, T)):
            mask = env_imit.mask[:, -1] & ~env_imit.arrive_flag[:, -1] 
            if ARGS.MODEL == 'TECRL':
                action = torch.full((env_imit.num_pedestrians, 2), float('nan'), device=env_imit.device)
                if mask.any():
                    action[mask, :], _ = model(pack_state(*env_imit.get_state())[mask])
                env_imit.action(action[:, 0], action[:, 1], enable_nan_action=True)
            elif ARGS.MODEL == 'SFM' or 'ORCA':
                if mask.any():  
                    model()
            
        
        # Plot and compare real and imit env
        plot = False
        if plot:
            # generate_gif(env_real, 'GC.gif')
            generate_gif(env_real, save_path=f'./result/GC/real_{t}.gif', start_time=t, final_time=min(t + 101, T), xrange=(5, 25), yrange=(15, 35))
            generate_gif(env_imit, save_path=f'./result/GC/imit_{t}_{ARGS.MODEL}.gif', xrange=(5, 25), yrange=(15, 35), imit_agent_mask=imit_agent_mask)
        
        ##### METRICS #####
        ## Macroscopic ##
        # real_mean_speeds.append(np.mean(real_mean_speed))
        # imit_mean_speeds.append(np.mean(imit_mean_speed))
        
        ## Microscopic ##
        Collision_rates.append(calc_collision_rate(env_imit))
        Displacements.append(calc_displacement(env_real, env_imit, imit_agent_mask, t, T))
        V_locomotions.append(calc_v_locomotion(env_real, env_imit, imit_agent_mask, t, T))
        A_locomotions.append(calc_a_locomotion(env_real, env_imit, imit_agent_mask, t, T))
        Perp_dev_distances.append(calc_perp_dev_distance(env_real, env_imit, imit_agent_mask, t, T))
        Energys.append(calc_energy(env_real, env_imit, imit_agent_mask, t, T))
        Steer_energys.append(calc_steer_energy(env_real, env_imit, imit_agent_mask, t, T))

        # # spatiotemporal similarity metrics
        ADE = calc_ade(env_real, env_imit, imit_agent_mask, t, T)
        ADEs.append(ADE)
        FDE = calc_fde(env_real, env_imit, imit_agent_mask, t, T)
        FDEs.append(FDE)
        pbar.set_postfix(dict(Col=(np.mean(Collision_rates)),
                              Dis=(np.mean(Displacements)), 
                              P_d_dist=(np.mean(Perp_dev_distances)),
                              V_loc=(np.mean(V_locomotions)),
                              A_loc=(np.mean(A_locomotions)),
                              Energy=(np.mean(Energys)),
                              S_energy=(np.mean(Steer_energys)),
                              ADE=(np.mean(ADEs)),
                              FDE=(np.mean(FDEs))
                        ))
        # break
    # print(f'ADE = {np.mean(ADEs)}, FDE = {np.mean(FDEs)}')
