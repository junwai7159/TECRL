import torch
import numpy as np
import os
import math
import pickle
from tqdm import tqdm

from envs.pedsim import Pedsim
from utils.utils import mod2pi, get_ttcmd

# find collision avoidance behavior
def find_CAP(env, D_max=0.45, T_max=3.5, T_reac=0.3):
    # D_max: social distancing, the distance people tolerate others encroaching on their side
    # T_max: avoidance lead time, temporary response range of pedestrians to future collisions
    # T_reac: human's reaction time
    """
    find CAP in environment env, return CAP_flag
    - CAP_flag: (N, N, T), bool, (i, j, t)=1 means pedestrian i is avoiding pedestrian j at time t
    """
    md, ttc, msk = get_ttcmd(env)
    N, T, dt = env.num_pedestrians, env.num_steps, env.meta_data['time_unit']
    F_reac = round(T_reac / dt)
    
    start = (md < D_max) & (ttc < T_max) & (msk) & (env.velocity[:, None, :, :].norm(dim=-1) > 0.5) # (N, N, T)
    start = start.int().cumsum(dim=-1)
    start_ = start.clone()
    start_[:, :, F_reac:] -= start[:, :, :-F_reac]
    start_flag = (start_ == F_reac)  # (N, N, T)

    final = ((ttc < 1e-8) & (md > 1e-8)) | (~msk) # (N, N, T)
    final = final.int().cumsum(dim=-1)
    final_ = final.clone()
    final_[:, :, F_reac:] -= final[:, :, :-F_reac]
    final_flag = (final_ == F_reac)  # (N, N, T)
    final_flag[:, :, -1] = True

    CAP_flag = torch.full((N, N, T), False, device=env.device)
    for i in range(N):
        for j in range(N):
            while start_flag[i, j].any():
                start_idx = start_flag[i, j].nonzero()[0, 0]
                final_idx = start_idx + final_flag[i, j, start_idx:].nonzero()[0, 0]
                start_flag[i, j, start_idx:final_idx + 1] = False
                if (final_idx - start_idx + 1) * dt > 0.4:  # ignore too short CAP
                    CAP_flag[i, j, start_idx:final_idx + 1] = True
    CAP_flag[~env.mask[:, None, :] | ~env.mask[None, :, :]]  = False

    return CAP_flag

##### Microscopic Metrics #####
def calc_collision_rate(env_imit):
    N, T, _ = env_imit.position.shape
    num_collided_agents = []

    for t in range(T):
        collision_agent_table = torch.full((N,), False)
        for i in range(N):
            for j in range(i + 1, N):
                if env_imit.mask[i, t] and env_imit.mask[j, t]:
                    if torch.norm(env_imit.position[i, t, :] - env_imit.position[j, t, :], dim=-1) <= (2 * env_imit.ped_radius) ** 2:
                        collision_agent_table[i] = True
                        collision_agent_table[j] = True
        num_collided_agents.append(torch.sum(collision_agent_table).item())

    return torch.mean(torch.tensor(num_collided_agents) / N)

def calc_displacement(env_real=None, env_imit=None, imit_agent_mask=None, t=None, T=None):
    real_displacement = []
    imit_displacement = []
    
    if env_real is not None:
        for id, pos in enumerate(env_real.position[imit_agent_mask, t:min(t + 101, T), :]):
            real_displacement.append(torch.sum(torch.norm(torch.diff(pos[env_real.mask[imit_agent_mask[id], t:min(t + 101, T)], :], dim=0), dim=1)).item())
    for id, pos in enumerate(env_imit.position):
        imit_displacement.append(torch.sum(torch.norm(torch.diff(pos[env_imit.mask[id, :], :], dim=0), dim=1)).item())
        
    if env_real is not None:
        displacement = torch.mean(torch.abs(torch.tensor(real_displacement) - torch.tensor(imit_displacement)))
    else:
        displacement = torch.mean(torch.tensor(imit_displacement))
    return displacement

def calc_perp_dev_distance(env_real=None, env_imit=None, imit_agent_mask=None, t=None, T=None):
    real_perp_dev_distance = []
    imit_perp_dev_distance = []

    # dist = |Ax_0 + By_0 + C| / sqrt(A^2 + B^2)
    # A = y_2 - y_1; B = x_1 - x_2; C = x-2 * y_1 - x_1 * y_2
    if env_real is not None:
        for id, pos in enumerate(env_real.position[imit_agent_mask, t:min(t + 101, T), :]):
            p0 = pos[env_real.mask[imit_agent_mask[id], t:min(t + 101, T)], :]
            p1 = p0[0]
            p2 = env_real.destination[imit_agent_mask[id]]

            A = p2[1] - p1[1]
            B = p1[0] - p2[1]
            C = p2[0] * p1[1] - p1[0] * p2[1]
            real_perp_dev_distance.append(torch.sum(torch.abs(A * p0[:, 0] + B * p0[:, 1] + C) / torch.sqrt(A**2 + B**2)).item())

    for id, pos in enumerate(env_imit.position):
        p0 = pos[env_imit.mask[id, :], :]
        p1 = pos[0, :]
        p1 = p0[0]
        p2 = env_imit.destination[id]
        A = p2[1] - p1[1]
        B = p1[0] - p2[1]
        C = p2[0] * p1[1] - p1[0] * p2[1]
        imit_perp_dev_distance.append(torch.sum(torch.abs(A * p0[:, 0] + B * p0[:, 1] + C) / torch.sqrt(A**2 + B**2)).item())        
    
    if env_real is not None:
        perp_dev_distance = torch.mean(torch.abs(torch.tensor(real_perp_dev_distance) - torch.tensor(imit_perp_dev_distance)))
    else:
        perp_dev_distance = torch.mean(torch.tensor(imit_perp_dev_distance))
    return perp_dev_distance

def calc_v_locomotion(env_real=None, env_imit=None, imit_agent_mask=None, t=None, T=None):
    real_v_locomotion = []
    imit_v_locomotion = []

    if env_real is not None:
        for id, vel in enumerate(env_real.velocity[imit_agent_mask, t:min(t + 101, T), :]):
            real_v_locomotion.append(torch.sum(torch.abs(torch.diff(torch.norm(vel[env_real.mask[imit_agent_mask[id], t:min(t + 101, T)], :], dim=1)))).item())
    for id, vel in enumerate(env_imit.velocity):
        imit_v_locomotion.append(torch.sum(torch.abs(torch.diff(torch.norm(vel[env_imit.mask[id, :], :], dim=1)))).item())
        
    if env_real is not None:
        v_locomotion = torch.mean(torch.abs(torch.tensor(real_v_locomotion) - torch.tensor(imit_v_locomotion)))
    else:
        v_locomotion = torch.mean(torch.tensor(imit_v_locomotion))
    return v_locomotion

def calc_a_locomotion(env_real=None, env_imit=None, imit_agent_mask=None, t=None, T=None):
    real_a_locomotion = []
    imit_a_locomotion = []
    
    # angle_diff = (angle1 - angle2 + pi) % 2pi - pi
    if env_real is not None:
        for id, ang in enumerate(env_real.direction[imit_agent_mask, t:min(t + 101, T), :]):
            real_a_locomotion.append(torch.sum(torch.abs((torch.diff(ang[env_real.mask[imit_agent_mask[id], t:min(t + 101, T)]].squeeze(dim=1)) + np.pi) % (2 * np.pi) - np.pi)).item())
    for id, ang in enumerate(env_imit.direction):
        imit_a_locomotion.append(torch.sum(torch.abs((torch.diff(ang[env_imit.mask[id, :]].squeeze(dim=1)) + np.pi) % (2 * np.pi) - np.pi)).item())
    
    if env_real is not None:
        a_locomotion = torch.mean(torch.abs(torch.tensor(real_a_locomotion) - torch.tensor(imit_a_locomotion)))
    else:
        a_locomotion = torch.mean(torch.tensor(imit_a_locomotion))
    return a_locomotion

def calc_energy(env_real=None, env_imit=None, imit_agent_mask=None, t=None, T=None):
    real_energy = []
    imit_energy = []

    dt = env_imit.meta_data['time_unit']
    mass = 60
    e_s = 2.23
    e_w = 1.26

    if env_real is not None:
        for id, vel in enumerate(env_real.velocity[imit_agent_mask, t:min(t + 101, T), :]):
            real_energy.append(torch.sum(mass * (e_s + e_w * torch.norm(vel[env_real.mask[imit_agent_mask[id], t:min(t + 101, T)], :], dim=1) ** 2) * dt).item())
    for id, vel in enumerate(env_imit.velocity):
        imit_energy.append(torch.sum(mass * (e_s + e_w * torch.norm(vel[env_imit.mask[id, :], :], dim=1) ** 2) * dt).item())

    if env_real is not None:
        energy = torch.mean(torch.abs(torch.tensor(real_energy) - torch.tensor(imit_energy)))
    else:
        energy = torch.mean(torch.tensor(imit_energy))
    return energy

def calc_steer_energy(env_real=None, env_imit=None, imit_agent_mask=None, t=None, T=None):
    real_steer_energy = []
    imit_steer_energy = []

    dt = env_imit.meta_data['time_unit']
    mass = 60
    e_s = 2.23
    e_w = 1.26

    if env_real is not None:
        for id, vel in enumerate(env_real.velocity[imit_agent_mask, t:min(t + 101, T), :]):
            real_steer_energy.append(torch.sum(mass * (e_s + e_w * torch.norm(torch.diff(vel[env_real.mask[imit_agent_mask[id], t:min(t + 101, T)], :], dim=1), dim=1) ** 2) * dt).item())
    for id, vel in enumerate(env_imit.velocity):
        imit_steer_energy.append(torch.sum(mass * (e_s + e_w * torch.norm(torch.diff(vel[env_imit.mask[id, :], :], dim=1), dim=1) ** 2) * dt).item())

    if env_real is not None:
        steer_energy = torch.mean(torch.abs(torch.tensor(real_steer_energy) - torch.tensor(imit_steer_energy)))
    else:
        steer_energy = torch.mean(torch.tensor(imit_steer_energy))
    return steer_energy


# TEC: total effort consumption
def calc_TEC(dt, position, destination, mass=60, radius=0.3, lambda_E=1e-3, lambda_W=1e-2, lambda_M=1.0):
    """
    calculate Total Effort Consumption (TEC) of a trajectory
    - dt: length of simulation time step
    - position: (T, 2)
    - destination: (1, 2)
    """
    velocity = position.diff(dim=0, prepend=position[(0,), :]) / dt  # (T, 2)
    direction = torch.atan2(velocity[:, (1,)], velocity[:, (0,)])  # (T, 1)
    
    speed = velocity.norm(dim=-1, keepdim=True)  # (T, 1)
    yaw_rate = mod2pi(direction.diff(dim=0, prepend=direction[(0,), :])) / dt # (T, 1)
    EC = mass * (2.23 + 1.25 * speed.square() + 2.0 * (.5 * radius**2) * yaw_rate.square()) * dt  # (T, 1)
    EC = EC.sum()

    velocity_ = velocity.roll(shifts=1, dims=0)  # (T, 2)
    velocity_diff = velocity - velocity_  # (T, 2)
    AW = .5 * mass * (velocity_diff * velocity_).sum(dim=-1, keepdim=True).abs() + .5 * mass * (velocity_diff * velocity).sum(dim=-1, keepdim=True).abs()  # (T, 1)
    AW = AW.sum()

    vec2des = destination.view(-1, 2) - position  # (T, 2)
    ang2des = torch.atan2(vec2des[:, (1,)], vec2des[:, (0,)])  # (T, 1)
    ori2des = (mod2pi(ang2des - direction)).abs() # (T, 1)
    ME = ori2des * speed * dt  # (T, 1)
    ME = ME.sum()
    
    TEC = lambda_E * EC + lambda_W * AW + lambda_M * ME
    return TEC

# ADE: average displacement error
def calc_ade(env_real, env_imit, agent_mask, t, T):
    real_position = env_real.position[agent_mask, t + 1:min(t + 101, T)]
    imit_position = env_imit.position[:, 1:]
    N = env_imit.num_pedestrians

    ADEs = []
    for i in range(N):
        real_pos = real_position[i]
        imit_pos = imit_position[i]

        real_mask = ~torch.isnan(real_pos).any(dim=1)
        imit_mask = ~torch.isnan(imit_pos).any(dim=1)
        mask = imit_mask if torch.sum(imit_mask) < torch.sum(real_mask) else real_mask  # truncate

        # weird bug, align length
        min_t = min(mask.shape[0], real_pos.shape[0], imit_pos.shape[0])
        if not (mask.shape[0] == real_pos.shape[0] == imit_pos.shape[0]):
            mask = mask[:min_t]
            real_pos = real_pos[:min_t]
            imit_pos = imit_pos[:min_t]

        ade = torch.mean(torch.sqrt(torch.sum((real_pos[mask] - imit_pos[mask])**2, dim=1)))
        if torch.isnan(ade):
            continue

        ADEs.append(ade.item())

    return np.mean(ADEs)

# FDE: final displacement error
def calc_fde(env_real, env_imit, agent_mask, t, T):
    real_position = env_real.position[agent_mask, t + 1:min(t + 101, T)]
    imit_position = env_imit.position[:, 1:]
    N = env_imit.num_pedestrians

    fde = []
    for i in range(N):
        real_pos = real_position[i]
        imit_pos = imit_position[i]

        real_mask = ~torch.isnan(real_pos).any(dim=1)
        imit_mask = ~torch.isnan(imit_pos).any(dim=1)
        
        real_pos = real_pos[real_mask]
        imit_pos = imit_pos[imit_mask]
        
        # if real_pos.shape[0] == 0 or imit_pos.shape[0] == 0:
        #     continue

        fde.append(torch.sqrt(torch.sum((real_pos[-1] - imit_pos[-1])**2, dim=-1)).item())

    return np.mean(fde)


##### misc. #####
def imitate_env(env_real, action_func, start_index=0, end_index=-1, show_tqdm=True, drop_nan=False, exit_with_real=False, cheat=False):
    """
    模仿 env_real, 从 start_index 帧开始到 end_index 帧 (含) 结束
    :param env_real: 被模仿的场景
    :param action_func: 给定当前场景 env 和需要获得动作的行人掩模 mask, 返回 (mask.sum(), 2) 维的动作向量
    :param start_index: 开始模仿的时刻
    :param end_index: 结束模仿的时刻 (含), 注意必须小于 env_real.num_steps
    :param show_tqdm: 是否显示进度条
    :param drop_nan: 是否扔掉模仿后 env 中全程未出现的行人, 若设置为 True, 则会一并返回一个 pedmap_r2i
    :param exit_with_real: 以真实场景 mask=0 为退出标记, 还是以模拟场景到达目的地为退出标记
    :param cheat: 是否将下一时刻的真实位置传入 action_func
    :return env: 模仿后的场景
    :return pedmap_r2i: 若 drop_nan 为 True, 则为从 env_real 中的每个人到 env 中每个人的编号映射, 否则不返回此值
    """
    env = Pedsim(env_real.args)
    env.meta_data = env_real.meta_data
    env.add_pedestrian(env_real.position[:, start_index, :], env_real.velocity[:, start_index, :], env_real.destination, init=True)
    end_index += env_real.num_steps if end_index < 0 else 0
    assert end_index in range(start_index, env_real.num_steps), f"Wrong index: {start_index} ~ {end_index}"
    for t in tqdm(range(start_index + 1, end_index + 1)) if show_tqdm else range(start_index + 1, end_index + 1):
        if exit_with_real:
            exit_flag = (env_real.mask[:, t - 1] == 1) & (env_real.mask[:, t] == 0)
            mask = env.mask[:, -1] & ~exit_flag
        else:
            mask = env.mask[:, -1] & ~env.arrive_flag[:, -1]
        action = torch.full((env.num_pedestrians, 2), float('nan'), device=env.device)
        if mask.any():
            if cheat: action[mask, :] = action_func(env, mask, env_real.position[:, t, :])
            else:     action[mask, :] = action_func(env, mask)
        env.action(action[:, 0], action[:, 1], enable_nan_action=True)
        into_flag = (env_real.mask[:, t] == 1) & (env_real.mask[:, t - 1] == 0)
        if into_flag.any():
            env.position[into_flag, -1, :] = env_real.position[into_flag, t, :]
            env.velocity[into_flag, -1, :] = env_real.velocity[into_flag, t, :]
            env.direction[into_flag, -1, :] = env_real.direction[into_flag, t, :]
            env.destination[into_flag, :] = env_real.destination[into_flag, :]
            env.arrive_flag[into_flag, -1] = ((env_real.position[into_flag, t, :] - env_real.destination[into_flag, :]).norm(dim=-1) < env_real.ped_radius)
            env.mask[into_flag, -1] = True
    if drop_nan:
        mask_ped = env.mask.any(dim=-1)
        env.num_pedestrians = mask_ped.sum().int().item()
        env.position = env.position[mask_ped]
        env.velocity = env.velocity[mask_ped]
        env.destination = env.destination[mask_ped]
        env.direction = env.direction[mask_ped]
        env.mask = env.mask[mask_ped]
        env.arrive_flag = env.arrive_flag[mask_ped]
        env.raw_velocity = env.raw_velocity[mask_ped]
        pedmap_r2i = mask_ped.cumsum(dim=0).int().sub(1).masked_fill(~mask_ped, -1)  # (N,)
        return env, pedmap_r2i
    else:
        return env

