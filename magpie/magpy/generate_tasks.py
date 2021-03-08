# maybe at some coord add trajectory controller
# for now just randomly sample some tasks
# save each task in a npy file inside task folder

import os
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import pylab


def sample(var_name, precision, length=1):
    value = np.random.randint(
        start[var_name]["min"] * precision, start[var_name]["max"] * precision, length
    ) / float(precision)
    return value[0] if length == 1 else value


def calc_coord(dist, alpha, unit_dir, previous_coord, precision):
    # get random point on circle area that is cutting the ball
    coord = []

    ball_radius = dist / np.cos(np.deg2rad(alpha))
    circle_radius = np.sqrt(ball_radius ** 2 - dist ** 2)
    circle_middle = dist * unit_dir + previous_coord
    circle_vec = np.array([unit_dir[2], 0, unit_dir[0]])
    theta = np.random.randint(0, 360 * precision, 1)[0] / float(
        precision
    )  # random angle
    theta = np.deg2rad(theta)
    mag = np.random.rand(1)[0] * circle_radius  # random magnitude

    # Rodrigues formula
    vec = (
        circle_vec * np.cos(theta)
        + np.cross(circle_vec, unit_dir) * np.sin(theta)
        + unit_dir * (np.dot(unit_dir, circle_vec)) * (1 - np.cos(theta))
    )
    vec = mag * vec
    coord = circle_middle + vec
    return coord


def calc_vel(unit_dir, vel_range, fix_velocity):
    # get velocities from direction vector --> |vel| = 10
    # scale velocity to lie in range 5 to 20 m/s
    vel = unit_dir * vel_range

    if fix_velocity:
        vel_scale = 1
    else:
        # scale velocity vector by 0.9 to 1.1 to have small changes between points
        vel_scale = np.random.randint(90, 110, 1)[0] / float(100)
    vel *= vel_scale
    return vel


def calc_angle(v):
    # https://www.codeproject.com/Questions/324240/Determining-yaw-pitch-and-roll
    angle = []

    roll = 0
    angle.append(roll)

    vp = np.copy(v)
    vp[2] = 0

    num = np.dot(v, vp)
    den = np.linalg.norm(v) * np.linalg.norm(vp)
    if den == 0:
        yaw = 0
    else:
        yaw = np.arccos(num / den)

    angle.append(yaw)

    vpp = np.copy(vp)
    vpp[0] = 0

    num = np.dot(vp, vpp)
    den = np.linalg.norm(vp) * np.linalg.norm(vpp)
    if den == 0:
        pitch = 0
    else:
        pitch = np.arccos(num / den)

    angle.append(pitch)
    angle = np.asarray(angle)

    return angle


def calc_wind(wind, use_wind, fix_wind):
    if not use_wind:
        return [0, 0, 0]
    else:
        if fix_wind:
            wind_scale = 1
        else:
            wind_scale = np.random.randint(90, 110, 1)[0] / float(100)

    wind *= wind_scale

    return wind


def asDict(point, var):
    names = [
        "position_n",
        "position_e",
        "position_d",
        "roll",
        "pitch",
        "yaw",
        "velocity_u",
        "velocity_v",
        "velocity_w",
        "wind_n",
        "wind_e",
        "wind_d",
    ]
    d = {}
    for i in range(len(names)):
        d[names[i]] = point[i]

    unwanted_vars = set(names) - set(var)
    for unwanted_key in unwanted_vars:
        del d[unwanted_key]
    return d


def save_task(dir, difficulty, name, task):
    name = difficulty + "_task_" + name
    path = os.path.join(dir, difficulty)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, name), task)


def plot_task(coords, dist, alpha):

    ball_radius = dist / np.cos(np.deg2rad(alpha))
    fig = plt.figure(figsize=(10, 10))
    # data = [self.res_n, self.res_e, self.res_d]
    xs = np.take(coords, 0, axis=1)
    ys = np.take(coords, 1, axis=1)
    zs = np.take(coords, 2, axis=1)

    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_xlim3d([np.min(xs) - 15, np.max(xs) + 15])
    ax.set_xlabel("X")
    ax.set_ylim3d([np.min(ys) - 15, np.max(ys) + 15])
    ax.set_ylabel("Y")
    ax.set_zlim3d([np.min(zs) - 15, np.max(zs) + 15])
    ax.set_zlabel("Z")

    ax.set_title("")

    ax.plot(xs, ys, zs, c="green")
    ax.scatter3D(xs[1:-1], ys[1:-1], zs[1:-1], cmap="Greens")
    ax.scatter3D(xs[0], ys[0], zs[0], c="r")
    ax.scatter3D(xs[-1], ys[-1], zs[-1], c="b")

    ax.text(xs[0], ys[0], zs[0], "%s" % ("Start"), size=20, zorder=1, color="k")
    ax.text(xs[-1], ys[-1], zs[-1], "%s" % ("Goal"), size=20, zorder=1, color="k")

    for i in range(len(xs)):
        if i % 2 == 0:
            continue
        else:
            r = ball_radius
            alpha = 0.1
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = r * np.outer(np.cos(u), np.sin(v)) + xs[i]
            y = r * np.outer(np.sin(u), np.sin(v)) + ys[i]
            z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + zs[i]
            ax.plot_surface(x, y, z, color="m", alpha=alpha)

    plt.show()
    return True


startingDir = os.path.dirname(os.path.realpath(__file__))
print(startingDir)
os.chdir(startingDir)
taskDir = os.path.join(startingDir, "tasks")

np.random.seed(10)
num_tasks = 200  # number of tasks to generate

var = [
    "position_n",
    "position_e",
    "position_d",
    "roll",
    "pitch",
    "yaw",
    "velocity_u",
    "velocity_v",
    "velocity_w",
    "wind_n",
    "wind_e",
    "wind_d",
]  # variables to keep

# set starting direction
fix_start_dir = True
start_dir = [1, 0, 0]

fix_velocity = True  # if magnitude of velocity should be kept
use_wind = False
fix_wind = True

precision = 10  # precision for random values
dist = 10  # distance between current coord and coord in same dir
alpha = 0  # angle to edge of cutting circle (0-90 degrees)
task_difficulty = "easy"

plot = False
save = True

start = {
    "position_n": {"min": -10, "max": 10},
    "position_e": {"min": -10, "max": 10},
    "position_d": {"min": -200, "max": -50},
    "velocity": {"min": 5, "max": 20},
    "wind": {"min": -6, "max": 6},
}

for n in range(num_tasks):
    task = []
    coords = []
    angles = []
    vels = []
    points = []
    pos = np.random.randint(5, 15, 1)[0] + 1

    # get random wind for whole trajectorie
    vel_range = sample("velocity", precision)
    wind = sample("wind", precision, 3)

    for p in range(pos):

        # calculate coordinates and velocities
        # first point is randomly sampled
        if p == 0:
            coord = []
            for v in ["position_n", "position_e", "position_d"]:
                coord.append(sample(v, precision))
            coord = np.asarray(coord)
            coords.append(coord)

            # just a placeholder
            vel = [0, 0, 0]
            vel = np.asarray(vel)
            vels.append(vel)
        else:
            # direction to second point is randomly sampled
            if p == 1:
                if fix_start_dir:
                    dir = start_dir
                else:
                    dir = np.random.randint(-10 * precision, 10 * precision, 3) / float(
                        precision
                    )
            # further directions are calculated between two last points
            else:
                dir = np.subtract(coords[p - 1], coords[p - 2])

            # normalize direction vector
            unit_dir = dir / np.linalg.norm(dir)

            # calc velocity vector
            vel = calc_vel(unit_dir, vel_range, fix_velocity)
            vels[p - 1] = vel
            vels.append(vel)

            # calc coordinates of next point
            previous_coord = coords[p - 1]
            coord = calc_coord(dist, alpha, unit_dir, previous_coord, precision)
            coords.append(coord)

        # calculate roll, pitch, yaw
        # calculate yaw, pitch from direction vector between two points
        if p != 0:
            v = coords[p] - coords[p - 1]
            angle = calc_angle(v)
            angles[p - 1] = angle
        else:
            angle = [0, 0, 0]
            angle = np.asarray(angle)

        angles.append(angle)

        # set wind
        wind = calc_wind(wind, use_wind, fix_wind)

        point = np.concatenate([coord, angle, vel, wind], axis=0)
        points.append(point)

        # put point into a dictionary
        point_dict = asDict(point, var)
        task.append(point_dict)

    if plot:
        plot_task(coords, dist, alpha)

    task = task[1:]
    task = np.asarray(task)

    if save:
        save_task(taskDir, task_difficulty, str(n), task)
