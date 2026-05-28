import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from uav_init import uav_states
import time
import numpy as np


pbft_prepare_votes = defaultdict(list)
pbft_commit_votes = defaultdict(list)
pbft_decisions = {}

PRINT_ENABLED = True  
FALSE_POSITIVE_MODE = True


# UAV CLASS (DYNAMIC MOTION)
class UAV:
    def __init__(self, uid, x, y, z, vx, vy, vz):
        self.uid = uid
        self.tx, self.ty, self.tz = x, y, z
        self.x, self.y, self.z = x, y, z
        self.vx, self.vy, self.vz = vx, vy, vz
        self.ax, self.ay, self.az = 0.0, 0.0, 0.0
        self.neighbors = []
        self.true_hist = [(x, y, z)]
        self.gps_hist = [(x, y, z)]

    def update(self, dt):
        self.ax = random.uniform(-0.5, 0.5)
        self.ay = random.uniform(-0.5, 0.5)
        self.az = random.uniform(-0.2, 0.2)
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self.vz += self.az * dt
        self.tx += self.vx * dt
        self.ty += self.vy * dt
        self.tz += self.vz * dt
        self.x, self.y, self.z = self.tx, self.ty, self.tz
        self.true_hist.append((self.tx, self.ty, self.tz))
        self.gps_hist.append((self.x, self.y, self.z))

    def true_pos(self):
        return (self.tx, self.ty, self.tz)

    def gps_pos(self):
        return (self.x, self.y, self.z)


# CONE MODEL 
def is_inside_cone(uav_pos, apex, direction, angle, height):
    px, py, pz = uav_pos
    ax, ay, az = apex
    dx, dy, dz = direction

    vx, vy, vz = px-ax, py-ay, pz-az
    dist = math.sqrt(vx*vx + vy*vy + vz*vz)

    if dist == 0 or dist > height:
        return False

    dot = vx*dx + vy*dy + vz*dz
    cos_theta = dot / dist

    return cos_theta >= math.cos(angle)


# UTILITY FUNCTIONS
def euclidean(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def trilateration_check(ui, uk, RSSI0, n, eps_d):
    d_true = euclidean(ui.true_pos(), uk.true_pos())
    RSSI = RSSI0 - 10 * n * math.log10(d_true + 1e-6)

    noise = random.gauss(0, 3) if FALSE_POSITIVE_MODE else 0
    RSSI += noise

    d_rssi = 10 ** ((RSSI0 - RSSI) / (10 * n))
    d_gps = euclidean(ui.gps_pos(), uk.gps_pos())
    return abs(d_rssi - d_gps) > eps_d


def aoa_from_pdoa(ui, uk, d_ant, lamb, sigma_phi=0.05):
    dx = uk.tx - ui.tx
    dy = uk.ty - ui.ty
    theta_true = math.atan2(dy, dx)

    phi_meas = (2 * math.pi * d_ant / lamb) * math.sin(theta_true)
    phi_meas += random.gauss(0, sigma_phi)

    arg = (phi_meas * lamb) / (2 * math.pi * d_ant)
    arg = max(-1.0, min(1.0, arg))

    return math.asin(arg)


def triangulation_check(ui, uj, uk, d_ant, lamb, eps_theta):
    alpha = aoa_from_pdoa(ui, uk, d_ant, lamb)
    beta  = aoa_from_pdoa(uj, uk, d_ant, lamb)

    gamma = math.pi - (alpha + beta)
    if abs(math.sin(gamma)) < 1e-3:
        return False

    AB = euclidean(ui.true_pos(), uj.true_pos())
    AC = AB * math.sin(beta) / math.sin(gamma)
    BC = AB * math.sin(alpha) / math.sin(gamma)

    d_gps_i = euclidean(ui.gps_pos(), uk.gps_pos())
    d_gps_j = euclidean(uj.gps_pos(), uk.gps_pos())

    return (abs(AC - d_gps_i) > eps_theta or
            abs(BC - d_gps_j) > eps_theta)


# PBFT CONSENSUS
def pbft_consensus(uavs, suspected_uid, leader_uid):
    neighbors = [u for u in uavs if u.uid != suspected_uid]
    Nk = len(neighbors)

    for u in neighbors:
        pbft_prepare_votes[suspected_uid].append(u.uid)

    if len(pbft_prepare_votes[suspected_uid]) / Nk >= 2 / 3:
        for u in neighbors:
            pbft_commit_votes[suspected_uid].append(u.uid)
        pbft_decisions[suspected_uid] = "MALICIOUS"
    else:
        pbft_decisions[suspected_uid] = "NORMAL"

    return pbft_decisions[suspected_uid]


# DYNAMIC LEADER ELECTION
def elect_new_leader(uavs):
    return random.choice([u.uid for u in uavs])


# MAIN SIMULATION
def main():
    total_start_time = time.perf_counter()

    print("=" * 60)
    print("UAV GPS SPOOFING DETECTION SIMULATOR (DYNAMIC LEADER)")
    print("=" * 60)

    n_uav = int(input("  • Enter number of UAVs to simulate (>3): "))
    states = uav_states[:n_uav]

    rounds = 30
    dt = 1

    RSSI0 = 46.6777
    n = 3
    eps_d = 2

    d_ant = 0.6
    lamb = 1.235
    eps_theta = 3.5

    uavs = [UAV(*state) for state in states]
    for u in uavs:
        u.neighbors = [x for x in uavs if x.uid != u.uid]

    spoof_round = random.randint(2, rounds - 1)

    spoofer_pos = (
        random.uniform(0, 40),
        random.uniform(0, 40),
        0.0
    )

    cone_angle = math.radians(25)
    cone_height = 50
    cone_dir = (0, 0, 1)

    print("Spoofer position:", spoofer_pos)

    attack_start_time = None
    detection_time = None
    leader_uid = None
    total_round_time = 0.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        nonlocal attack_start_time, detection_time, leader_uid, total_round_time 

        round_start_time = time.perf_counter()   # ADDED

        pbft_prepare_votes.clear()
        pbft_commit_votes.clear()
        pbft_decisions.clear()

        leader_uid = elect_new_leader(uavs)
        print(f"\n>>> ROUND {frame+1} LEADER : UAV {leader_uid}")

        ax.clear()

        if frame > 0:
            for u in uavs:
                u.update(dt)

        spoofed_uavs = []

        if frame >= spoof_round:
            for u in uavs:
                if is_inside_cone(u.true_pos(), spoofer_pos, cone_dir, cone_angle, cone_height):
                    spoofed_uavs.append(u.uid)
                    u.x = u.tx + random.uniform(5, 10)
                    u.y = u.ty + random.uniform(5, 10)
                    u.z = u.tz + random.uniform(5, 10)
                    u.gps_hist[-1] = (u.x, u.y, u.z)

            if len(spoofed_uavs) > 0 and attack_start_time is None:
                attack_start_time = time.perf_counter()

        if frame >= spoof_round and len(spoofed_uavs) > 0:
            suspected = spoofed_uavs[0]
            print(f"[Round {frame+1}] Spoofed UAVs:", spoofed_uavs)

            for u in uavs:
                if u.uid == suspected:
                    continue

                helper = random.choice([x for x in uavs if x.uid not in [u.uid, suspected]])

                suspicious = (
                    trilateration_check(u, uavs[suspected], RSSI0, n, eps_d) or
                    triangulation_check(u, helper, uavs[suspected], d_ant, lamb, eps_theta)
                )

                print(f"  UAV {u.uid} → {'MALICIOUS' if suspicious else 'NORMAL'}")

            decision = pbft_consensus(uavs, suspected, leader_uid)
            print(f"  PBFT DECISION → UAV {suspected} is {decision}")

            if decision == "MALICIOUS" and detection_time is None:
                detection_time = time.perf_counter() - attack_start_time
                print("\n>>> SPOOFING DETECTED <<<")
                print(f">>> TIME TO DETECTION = {detection_time:.6f} seconds\n")

        for u in uavs:
            gx, gy, gz = zip(*u.gps_hist)
            tx, ty, tz = zip(*u.true_hist)

            ax.plot(tx, ty, tz, "k--", linewidth=1)

            if u.uid == leader_uid:
                color = "#8A2BE2"
            elif u.uid in spoofed_uavs:
                color = "red"
            else:
                color = "blue"

            ax.plot(gx, gy, gz, color, linewidth=2)
            ax.scatter(gx[-1], gy[-1], gz[-1], c=color, s=60)
            ax.text(gx[-1], gy[-1], gz[-1], f"UAV{u.uid}")

        ax.scatter(*spoofer_pos, c="green", s=120)
        ax.text(*spoofer_pos, "Spoofer", color="green")

        theta = np.linspace(0, 2*np.pi, 40)
        h = np.linspace(0, cone_height, 40)
        Theta, H = np.meshgrid(theta, h)

        R = H * np.tan(cone_angle)
        X = spoofer_pos[0] + R * np.cos(Theta)
        Y = spoofer_pos[1] + R * np.sin(Theta)
        Z = spoofer_pos[2] + H

        ax.plot_surface(X, Y, Z, alpha=0.25, color="green")

        ax.set_title(f"Dynamic UAV Network – Round {frame+1}")
        ax.set_xlim(-20, 60)
        ax.set_ylim(-20, 60)
        ax.set_zlim(0, 60)

        ax.legend(handles=[
            Line2D([0], [0], marker='o', label='Leader UAV', markerfacecolor='#8A2BE2', markersize=8),
            Line2D([0], [0], marker='o', label='Honest UAV', markerfacecolor='blue', markersize=8),
            Line2D([0], [0], marker='o', label='Spoofed UAV', markerfacecolor='red', markersize=8),
        ])
        round_end_time = time.perf_counter()   
        round_runtime = round_end_time - round_start_time
        total_round_time += round_runtime

        print(f">>> ROUND {frame+1} EXECUTION TIME = {round_runtime:.6f} seconds")


    ani = FuncAnimation(fig, update, frames=rounds, interval=1000, repeat=False)
    ani.save("uav_dynamic_leader_spoofing.gif", writer="pillow", fps=1)
    plt.show()

    total_end_time = time.perf_counter()
    total_runtime = total_end_time - total_start_time

    print("\n" + "=" * 60)
    print("TOTAL ALGORITHM RUNTIME")
    print(f"Summation of All Round Execution Times = {total_round_time:.6f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()