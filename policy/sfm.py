import numpy as np
from envs.pedsim import Pedsim
from utils.utils import init_env, get_args, set_seed

v_pref = 1.33
A = 2
B = 1
KI = 1
timestep = 0.08

# initial_state : px,py,vx,vy,gx,gy

if __name__ == '__main__':
  ARGS = get_args()
  set_seed(ARGS.SEED)

  env = Pedsim(ARGS)
  init_env(env, ARGS)

  for step in range(3): # ARGS.MAX_EP_STEPS
    # Pull force to goal
    delta_x =  env.destination[:, 0] - env.position[:, -1, 0]
    delta_y =  env.destination[:, 1] - env.position[:, -1, 1]
    dist_to_goal = np.sqrt(delta_x**2 + delta_y**2)
    desired_vx = (delta_x / dist_to_goal) * v_pref
    desired_vy = (delta_y / dist_to_goal) * v_pref
    curr_delta_vx = KI * (desired_vx - env.velocity[:, -1, 0])
    curr_delta_vy = KI * (desired_vy - env.velocity[:, -1, 1])

    # Push force(s) from other agents
    interaction_vx = 0
    interaction_vy = 0
    # ...

    # Sum of push & pull forces
    total_delta_vx = (curr_delta_vx + interaction_vx) * timestep
    total_delta_vy = (curr_delta_vy + interaction_vy) * timestep

    # clip the speed
    new_vx = env.velocity[:, -1, 0] + total_delta_vx
    new_vy = env.velocity[:, -1, 1] + total_delta_vy

    print(new_vx)
    