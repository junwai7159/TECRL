import random
import numpy as np
import torch
import argparse
import setproctitle
import os
import sys

########## STATE ##########
def pack_state(s_self, s_int, s_ext):  # (N, 1) & (N, 8) & (N, 20, 8) -> (N, 169)
    s = torch.cat([s_self, s_int, s_ext.view(s_ext.shape[0], -1)], dim=-1)
    return s

def unpack_state(s):  # (N, 169) -> (N, 1) & (N, 8) & (N, 20, 8)
    s_self, s_int, s_ext = s.split((1, 8, s.shape[1] - 1 - 8), dim=-1)
    s_ext = s_ext.view(s_ext.shape[0], -1, 8)
    return s_self, s_int, s_ext


########## ENVS ##########
# 定义行人、障碍物、地图
scenario_list = ['CIRCLE', 'CORRIDOR', 'CROSSING', 'RANDOM']

def init_ped_circle(ARGS):
    angles = np.linspace(0, 2*np.pi, ARGS.NUM_PED, endpoint=False)
    positions = [[round(ARGS.SIZE_ENV * np.cos(theta), 4), round(ARGS.SIZE_ENV * np.sin(theta), 4)] for theta in angles]
    destinations = [[-pos[0], -pos[1]] for pos in positions]
    return torch.tensor(positions, dtype=torch.float32), torch.tensor(destinations, dtype=torch.float32)

def init_ped_corridor(env, ARGS, vertical=None, horizontal=None):
    positions, destinations = list(), list()
    rng = np.random.RandomState()
    r2 = env.ped_radius ** 2
    i, placeable = 0, None
    len_ratio = 3
    
    # Positions
    while len(positions) < ARGS.NUM_PED:
        if vertical:
            x = rng.rand() - 0.5
            if horizontal: x *= 0.5
            y = (rng.random() - 0.5) * 0.5
            if y < 0: 
                y -= 0.25
            else: 
                y += 0.25
        elif horizontal:
            x = (rng.random() - 0.5) * 0.5
            if x < 0:
                x -= 0.25
            else:
                x += 0.25
            y = rng.random() - 0.5
            if vertical: y *= 0.5
        
        if vertical:
            ped_pos0 = [round(x * ARGS.SIZE_ENV * 1.5, 4), round(y * ARGS.SIZE_ENV * 2 * len_ratio, 4)]
        elif horizontal:
            ped_pos0 = [round(x * ARGS.SIZE_ENV * 2 * len_ratio, 4), round(y * ARGS.SIZE_ENV * 1.5, 4)]

        for ped_pos1 in positions:
            dist2 = (ped_pos0[0] - ped_pos1[0])**2 + (ped_pos0[1] - ped_pos1[1])**2
            if dist2 <= (env.ped_radius * 2)**2: # Collision
                placeable = False
            placeable = True
        if placeable or not positions:
            positions.append(ped_pos0)
    
    # Destinations
    while len(destinations) < ARGS.NUM_PED:
        if vertical:
            x = rng.random() - 0.5
            if horizontal: x *= 0.5
            y = (rng.random() - 0.5) * 0.5 
            if y < 0:
                y -= 0.25
            else:
                y += 0.25
            if ((positions[i][1] > 0 and y > 0) or (positions[i][1] < 0 and y < 0)):
                y = -y
        elif horizontal:
            x = (rng.random() - 0.5) * 0.5
            if x < 0:
                x -= 0.25
            else:
                x += 0.25
            if (positions[i][0] > 0 and x > 0) or (positions[i][0] < 0 and x < 0):
                x = -x
            y = rng.random()-0.5
            if vertical: y *= 0.5
        
        if vertical:
            x *= ARGS.SIZE_ENV 
            y *= ARGS.SIZE_ENV * 2 * len_ratio       
        elif horizontal: 
            x *= ARGS.SIZE_ENV * 2 * len_ratio
            y *= ARGS.SIZE_ENV

        if (positions[i][0] - x)**2 + (positions[i][1] - y)**2 <= r2:
            continue
        placeable = True
        for gx, gy in destinations:
            if (gx-x)**2 + (gy-y)**2 <= r2:
                placeable = False
        if placeable:
            destinations.append([x, y])
            i+=1

    return torch.tensor(positions, dtype=torch.float32), torch.tensor(destinations, dtype=torch.float32)

def init_ped_crossing(env, ARGS):
    positions, destinations = list(), list()
    rng = np.random.RandomState()
    r2 = env.ped_radius ** 2
    i, placeable = 0, None
    len_ratio = 3
    vertical, horizontal = None, None
    
    # Positions
    while len(positions) < ARGS.NUM_PED:
        if rng.random() > 0.5:
            vertical, horizontal = True, False
        else:
            vertical, horizontal = False, True

        if vertical:
            x = rng.rand() - 0.5
            if horizontal: x *= 0.5
            y = (rng.random() - 0.5) * 0.5
            if y < 0: 
                y -= 0.25
            else: 
                y += 0.25
        elif horizontal:
            x = (rng.random() - 0.5) * 0.5
            if x < 0:
                x -= 0.25
            else:
                x += 0.25
            y = rng.random() - 0.5
            if vertical: y *= 0.5
        
        if vertical:
            ped_pos0 = [round(x * ARGS.SIZE_ENV * 1.5, 4), round(y * ARGS.SIZE_ENV * 2 * len_ratio, 4)]
        elif horizontal:
            ped_pos0 = [round(x * ARGS.SIZE_ENV * 2 * len_ratio, 4), round(y * ARGS.SIZE_ENV * 1.5, 4)]

        for ped_pos1 in positions:
            dist2 = (ped_pos0[0] - ped_pos1[0])**2 + (ped_pos0[1] - ped_pos1[1])**2
            if dist2 <= (env.ped_radius * 2)**2: # Collision
                placeable = False
            placeable = True
        if placeable or not positions:
            positions.append(ped_pos0)
    
    # Destinations
    while len(destinations) < ARGS.NUM_PED:
        if rng.random() > 0.5:
            vertical, horizontal = True, False
        else:
            vertical, horizontal = False, True
    
        if vertical:
            x = rng.random() - 0.5
            if horizontal: x *= 0.5
            y = (rng.random() - 0.5) * 0.5 
            if y < 0:
                y -= 0.25
            else:
                y += 0.25
            if ((positions[i][1] > 0 and y > 0) or (positions[i][1] < 0 and y < 0)):
                y = -y
        else:
            x = (rng.random() - 0.5) * 0.5
            if x < 0:
                x -= 0.25
            else:
                x += 0.25
            if (positions[i][0] > 0 and x > 0) or (positions[i][0] < 0 and x < 0):
                x = -x
            y = rng.random()-0.5
            if vertical: y *= 0.5

        if vertical:
            x *= ARGS.SIZE_ENV 
            y *= ARGS.SIZE_ENV * 2 * len_ratio       
        elif horizontal: 
            x *= ARGS.SIZE_ENV * 2 * len_ratio
            y *= ARGS.SIZE_ENV

        if (positions[i][0] - x)**2 + (positions[i][1] - y)**2 <= r2:
            continue
        placeable = True
        for gx, gy in destinations:
            if (gx-x)**2 + (gy-y)**2 <= r2:
                placeable = False
        if placeable:
            destinations.append([x, y])
            i+=1

    return torch.tensor(positions, dtype=torch.float32), torch.tensor(destinations, dtype=torch.float32)

def init_env(env, ARGS):
    n1, n2, size = ARGS.NUM_PED, ARGS.NUM_OBS, ARGS.SIZE_ENV
    ########## Pedestrians ##########
    velocity = 0.0 * torch.rand((n1, 2))
    if ARGS.SCENARIO == 'CIRCLE':
        positions, destinations = init_ped_circle(ARGS)
    elif ARGS.SCENARIO == 'CORRIDOR':
        positions, destinations = init_ped_corridor(env, ARGS, vertical=True, horizontal=False)
    elif ARGS.SCENARIO == 'CROSSING':
        positions, destinations = init_ped_crossing(env, ARGS)
    elif ARGS.SCENARIO == 'RANDOM' or 'CROSSING':
        positions = torch.distributions.Uniform(-size, size).sample([n1, 2])
        destinations = torch.distributions.Uniform(-size, size).sample([n1, 2])

    env.add_pedestrian(positions, velocity, destinations, init=True)
    
    ########## Obstacles ##########
    n2 = torch.distributions.Poisson(-n2).sample().to(int).item() if n2 < 0 else n2
    len_ratio = 3

    if n2 > 0:
        if ARGS.SCENARIO == 'CIRCLE' or ARGS.SCENARIO == 'RANDOM':
            obstacles = torch.distributions.Uniform(-1.5 * size, 1.5 * size).sample([n2, 2])
            
        elif ARGS.SCENARIO == 'CORRIDOR':
            vertical, horizontal = True, False
            if horizontal:
                start_top, end_top = np.array([-ARGS.SIZE_ENV * len_ratio, ARGS.SIZE_ENV]), np.array([ARGS.SIZE_ENV * len_ratio, ARGS.SIZE_ENV])
                start_bottom, end_bottom = np.array([-ARGS.SIZE_ENV * len_ratio, -ARGS.SIZE_ENV]), np.array([ARGS.SIZE_ENV * len_ratio, -ARGS.SIZE_ENV])
                n_top = int(np.linalg.norm(end_top - start_top) / (env.obstacle_radius * 2))
                n_bottom = int(np.linalg.norm(end_bottom - start_bottom) / (env.obstacle_radius * 2))
                obstacles_top = np.linspace(start_top, end_top, n_top)
                obstacles_bottom = np.linspace(start_bottom, end_bottom, n_bottom)
                obstacles = torch.tensor(np.concatenate((obstacles_top, obstacles_bottom)), dtype=torch.float32)
            elif vertical:
                start_top, end_top = np.array([-ARGS.SIZE_ENV, ARGS.SIZE_ENV * len_ratio]), np.array([-ARGS.SIZE_ENV, -ARGS.SIZE_ENV * len_ratio])
                start_bottom, end_bottom = np.array([ARGS.SIZE_ENV, ARGS.SIZE_ENV * len_ratio]), np.array([ARGS.SIZE_ENV, -ARGS.SIZE_ENV * len_ratio])
                n_top = int(np.linalg.norm(end_top-start_top) / (env.obstacle_radius * 2))
                n_bottom = int(np.linalg.norm(end_bottom-start_bottom) / (env.obstacle_radius * 2))
                obstacles_top = np.linspace(start_top, end_top, n_top)
                obstacles_bottom = np.linspace(start_bottom, end_bottom, n_bottom)
                obstacles = torch.tensor(np.concatenate((obstacles_top, obstacles_bottom)), dtype=torch.float32)

        elif ARGS.SCENARIO == 'CROSSING':
            quadrant1 = np.array([[ARGS.SIZE_ENV, ARGS.SIZE_ENV * len_ratio], [ARGS.SIZE_ENV , ARGS.SIZE_ENV], [ARGS.SIZE_ENV * len_ratio, ARGS.SIZE_ENV]])
            quadrant2 = np.array([[-ARGS.SIZE_ENV, ARGS.SIZE_ENV * len_ratio], [-ARGS.SIZE_ENV, ARGS.SIZE_ENV], [-ARGS.SIZE_ENV * len_ratio, ARGS.SIZE_ENV]])
            quadrant3 = np.array([[-ARGS.SIZE_ENV, -ARGS.SIZE_ENV * len_ratio], [-ARGS.SIZE_ENV, -ARGS.SIZE_ENV], [-ARGS.SIZE_ENV * len_ratio, -ARGS.SIZE_ENV]])
            quadrant4 = np.array([[ARGS.SIZE_ENV, -ARGS.SIZE_ENV * len_ratio], [ARGS.SIZE_ENV, -ARGS.SIZE_ENV], [ARGS.SIZE_ENV * len_ratio, -ARGS.SIZE_ENV]])

            def init_quad_crossing(quadrant):
                for vertex in quadrant:
                    n_line1 = int(np.linalg.norm(quadrant[0] - quadrant[1]) / (env.obstacle_radius * 2))
                    n_line2 = int(np.linalg.norm(quadrant[1] - quadrant[2]) / (env.obstacle_radius * 2)) - 1
                    obstacles_line1 = np.linspace(quadrant[0], quadrant[1], n_line1)
                    obstacles_line2 = np.linspace(quadrant[1], quadrant[2], n_line2)[1:]
                    obstacles = torch.tensor(np.concatenate((obstacles_line1, obstacles_line2)), dtype=torch.float32)
                return obstacles
            
            obstacles = torch.cat((init_quad_crossing(quadrant1), init_quad_crossing(quadrant2), init_quad_crossing(quadrant3), init_quad_crossing(quadrant4)))
            
        env.add_obstacle(obstacles, init=True)


########## CONFIG ########## 
def get_args(**kwargs):
    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument("--UUID", type=str, default="testproj", help='custom model name')
    parser.add_argument("--SAVE_DIRECTORY", type=str, default="checkpoint", help='save directory')
    parser.add_argument("--LOAD_MODEL", type=str, default=None, help='pretrained model to load')
    parser.add_argument("--SEED", type=int, default=42, help='random seed')
    parser.add_argument("--DEVICE", type=str, default="cpu", help='cpu or cuda')
    parser.add_argument("--MODEL", type=str, default="TECRL", help="SFM or ORCA or TECRL")

    # parameters in environments
    parser.add_argument("--NO_COLLISION_DETECTION", action="store_true", help='disable the collision detection mechanism')
    parser.add_argument("--NUM_PED", type=int, default=10, help='the number of pedestrians in the environment')
    parser.add_argument("--NUM_OBS", type=int, default=10, help='the number of obstacles in the environment')
    parser.add_argument("--SIZE_ENV", type=float, default=10.0, help='half the length of the side of the environment')
    parser.add_argument("--RW_ARRIVE", type=float, default=0.0, help='reward for arrived pedestrian')
    parser.add_argument("--RW_WORK", type=float, default=-0.01, help='reward: process work')
    parser.add_argument("--RW_ENERGY", type=float, default=-0.0015, help='reward: energy consumption')
    parser.add_argument("--RW_MENTAL", type=float, default=6.3, help='reward: mental effort')
    parser.add_argument("--SCENARIO", type=str, default="RANDOM", help='scenario of the environment')

    # parameters in agents
    parser.add_argument("--MEMORY_CAPACITY", type=int, default=6000)
    parser.add_argument("--MAX_EPISODES", type=int, default=50000)
    parser.add_argument("--MAX_EP_STEPS", type=int, default=200)
    parser.add_argument("--K_EPOCH", type=int, default=128)
    parser.add_argument("--GAMMA", type=float, default=0.95)
    parser.add_argument('--LAMBDA', type=float, default=0.9)
    parser.add_argument("--EPSILON", type=float, default=0.2)
    parser.add_argument('--ENTROPY', type=float, default=0.01)    
    parser.add_argument("--LR_0", type=float, default=1e-4, help='learning rate: the feature extractor')
    parser.add_argument("--LR_A", type=float, default=1e-4, help='learning rate: actor')
    parser.add_argument("--LR_C", type=float, default=3e-4, help='learning rate: critic')
    parser.add_argument("--H_FEATURE", type=str, default="(128,)")
    parser.add_argument("--H_ATTENTION", type=str, default="(128,)")
    parser.add_argument("--H_SEQ", type=str, default="(128, 128)")

    # custom parameters
    for key, value in kwargs.items():
        if type(value) is dict:  # add a new parameter, such as `get_args(NEW_ARG=dict(type=str, default='new_arg_value'))`
            parser.add_argument(f"--{key}", **value)
        else:  # reset default value of a existing parameter, `such as get_args(DEVICE='cuda')`
            parser.set_defaults(**{key: value})
    args, unknown = parser.parse_known_args()
    if len(unknown):
        print(f'[Warning] Unknown Arguments: {unknown}')
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


########## ACTION ########## 
def rotate(vec, ang):
    """
    rotate a vector by a angle
    :param vec: (N, 2)
    :param ang: (N,)
    """
    c, s = ang.cos(), ang.sin()
    mat = torch.stack([c, -s, s, c], dim=0).view(2, 2, *vec.shape[:-1])  # (2, 2, N)
    vec_ = torch.einsum('ji...,...i->...j', mat, vec)  # (N, 2)
    return vec_

def xy2ra(xy):
    """
    :param xy: [N, 2]
    :return: ra: [N, 2]
    """
    r = torch.norm(xy, dim=-1)
    a = torch.atan2(xy[:, 1], xy[:, 0])
    ra = torch.stack([r, a], dim=-1)
    return ra

def ra2xy(ra):
    """
    :param ra: [N, 2]
    :return: xy: [N, 2]
    """
    x = ra[:, 0] * torch.cos(ra[:, 1])
    y = ra[:, 0] * torch.sin(ra[:, 1])
    xy = torch.stack([x, y], dim=-1)
    return xy

def xy2rscnt(pos, vel, dir=0):
    """
    get observation states
    :param pos: [N, 2]
    :param vel: [N, 2]
    :return rscnt: [N, 8]
        r: distance
        s: sin(orientation), left(+) right(-)
        c: cos(orientation), front(+) back(-)
        n: departure speed, departure(+) approach(-)
        t: circular velocity, anticlockwise(+), clockwise(-)
        a: orientation
        x: r * c
        y: r * s
    """
    r = pos.norm(dim=-1, keepdim=True)
    a = mod2pi(torch.atan2(pos[:, 1], pos[:, 0]).unsqueeze(dim=-1) - dir)
    s = a.sin()
    c = a.cos()
    x = r * c
    y = r * s
    _r = 1. / (r + 1e-8)
    n = (pos * vel).sum(dim=-1, keepdim=True) * _r
    t = torch.diff((pos.flip(-1) * vel), dim=-1) * _r
    rscnt = torch.cat([r, s, c, n, t, a, x, y], dim=-1)
    return rscnt

def mod2pi(delta_angle):
    """
    map a angle in (-2pi, 2pi) into (-pi, pi), used to deal with angle differences
    - -2pi < x < -pi: return x + 2pi
    - -pi < x < +pi: return x
    - +pi < x < +2pi: return x - 2pi
    """
    return torch.remainder(delta_angle + np.pi, 2 * np.pi) - np.pi


def get_ttcmd(env, TTC_MAX=20.0, FRIEND_DIS=5.0, FRIEND_RATIO=0.7):
    """
    calculate the TTC and MD between N*N pairs of pedestrians at T steps, return the MD, TTC, and MASK
    - MD: (N, N, T), the closest distance between two pedestrians within future TTC_MAX time
    - TTC: (N, N, T), the time they take to reach their closest distance
    - MASK: (N, N, T), bool, whether the corresponding MD & TTC is valid.
        - when i cannot see j at time t, (i, j, t) is invalid
        - when i and j are friends, (i, j, :) is invalid
    """
    # MD: DCA (Distance of Closest Approach)
    # TTC: TTCA (Time to Closest Approach)
    # TTC_MAX: T_clip

    N, T = env.num_pedestrians, env.num_steps
    time_idx = torch.arange(T, device=env.device) # (t,)
    self_idx = torch.arange(N, device=env.device) # (n,)
    peds_idx = torch.arange(N, device=env.device) # (N,)
    pair_idx = torch.stack(torch.meshgrid(self_idx, peds_idx), dim=-1)  # (n, N, 2)
    pos = env.position[:, time_idx, :][pair_idx, :, :]  # (n, N, 2, t, 2)
    vel = env.velocity[:, time_idx, :][pair_idx, :, :]  # (n, N, 2, t, 2)
    drc = env.direction[:, time_idx, :][pair_idx, :, :]  # (n, N, 2, t, 1)
    xx = (pos[:, :, 0] - pos[:, :, 1]).square().sum(dim=-1)  # (n, N, t)
    xv = ((pos[:, :, 0] - pos[:, :, 1]) * (vel[:, :, 0] - vel[:, :, 1])).sum(dim=-1)  # (n, N, t)
    vv = (vel[:, :, 0] - vel[:, :, 1]).square().sum(dim=-1)  # (n, N, t)

    # calculate MD & TTC
    r2 = 2 * env.ped_radius
    d_now = xx.sqrt() - r2
    d_max = (vv * TTC_MAX**2 + 2 * xv * TTC_MAX + xx).clamp(1e-8).sqrt() - r2
    d_min = (xx - xv ** 2 / vv.clamp(1e-8)).clamp(1e-8).sqrt() - r2
    t_min = -xv / vv.clamp(1e-8)    # ?
    t_col = (-xv - (xv ** 2 - (xx - r2**2) * vv).sqrt()) / vv.clamp(1e-8)   # ?
    md = d_min.clamp(0).where(t_min <= TTC_MAX, d_max.clamp(0)).where(t_min >= 0, d_now.clamp(0.))
    ttc = t_min.clamp(0, TTC_MAX).where(t_col.isnan() | (t_min <= 0), t_col.clamp(0., TTC_MAX))

    # calculate MASK
    dp = pos[:, :, 1] - pos[:, :, 0]  # (n, N, t, 2)
    view = mod2pi(torch.atan2(dp[:, :, :, 1], dp[:, :, :, 0]) - drc[:, :, 0, :, 0]).abs() < np.pi / 2  # (n, N, t)
    # zone = (xx.sqrt() - r2 < 0.45)
    # view |= zone & ((view & zone).cumsum(dim=-1) > 0)
    friend = ((xx < FRIEND_DIS ** 2).float().sum(dim=-1, keepdim=True) / pos.isnan().any(dim=-1).all(dim=2).logical_not().sum(dim=-1, keepdim=True) > FRIEND_RATIO)  # (N, N, 1)
    # into_env = env.mask.clone(); into_env[:, 1:] &= ~into_env[:, :-1]  # (N, T)
    # exit_env = env.mask.clone(); exit_env[:, :-1] &= ~exit_env[:, 1:]  # (N, T)
    # into_env[env.mask.any(dim=-1).logical_not(), -1] = True
    # exit_env[env.mask.any(dim=-1).logical_not(), 0] = True
    # assert (into_env.sum(dim=1) == 1).all() and (exit_env.sum(dim=1) == 1).all(), "A pedestrian enter the environment for more than 1 times!"
    # src = env.position[into_env, :]  # (N, 2)
    # tgt = env.position[exit_env, :]  # (N, 2)
    # friend &= (src[pair_idx, :].diff(dim=2).norm(dim=-1) < FRIEND_DIS) & (tgt[pair_idx, :].diff(dim=2).norm(dim=-1) < FRIEND_DIS)  # (N, 2) -> (N, N, 2, 2) -> (N, N, 1, 2) -> (N, N, 1)
    msk = (view) & (~friend)  # (n, N, t)

    assert (md[msk] >= 0).all() and (ttc[msk] >= 0).all() and (ttc[msk] <= TTC_MAX).all(), "there must be something wrong!"
    return md, ttc, msk

