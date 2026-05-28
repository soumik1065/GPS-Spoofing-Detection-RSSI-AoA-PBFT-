import random

def generate_uavs(n):
    return [
        (i,
         random.uniform(0, 25),   # x
         random.uniform(0, 25),   # y
         random.uniform(2, 20),   # z
         random.uniform(-2, 2),   # vx
         random.uniform(-2, 2),   # vy
         random.uniform(0.5, 4))  # vz 
        for i in range(n)
    ]


uav_states = generate_uavs(1000)
