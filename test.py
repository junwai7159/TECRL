import numpy as np
import torch





def init_ped_circle(r, n1):
    angles = np.linspace(0, 2*np.pi, n1, endpoint=False)
    positions = [[round(r*np.cos(theta), 4), round(r*np.sin(theta), 4)] for theta in angles]
    destinations = [[round(r*np.cos(theta) + np.pi, 4), round(r*np.sin(theta) + np.pi, 4)] for theta in angles]
    return torch.tensor(positions), torch.tensor(destinations)

p = init_ped_circle(40, 8)
print(p)