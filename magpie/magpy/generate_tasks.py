# maybe at some coord add trajectory controller
# for now just randomly sample some tasks
# save each task in a npy file inside task folder

import os
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import pylab


def plot_task(coords, ball_radius):

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

num_tasks = 200
var = ["roll", "pitch", "Va"]
scaling = [1, 1, 1]

fix_start_dir = True
start_dir = [1, 0, 0]
precision = 10
dist = 10  # distance between current coord and coord in same dir
alpha = 0  # angle to edge of cutting circle (0-90 degrees)
np.random.seed(10)
plot = False
save = True

start = {
    "roll": {"min": -10, "max": 10},
    "pitch": {"min": -10, "max": 10},
    "Va": {"min": 5, "max": 25},
    "position_n": {"min": -10, "max": 10},
    "position_e": {"min": -10, "max": 10},
    "position_d": {"min": 50, "max": 200},
}

ball_radius = dist / np.cos(np.deg2rad(alpha))
circle_radius = np.sqrt(ball_radius ** 2 - dist ** 2)
names = ["position_n", "position_e", "position_d", "roll", "pitch", "Va"]

for n in range(num_tasks):
    task = []
    coords = []
    angles = []
    points = []
    pos = np.random.randint(5, 15, 1)[0]
    for p in range(pos):

        coord = []
        if p == 0:
            for v in ["position_n", "position_e", "position_d"]:
                coord.append(
                    np.random.randint(
                        start[v]["min"] * precision, start[v]["max"] * precision, 1
                    )[0]
                    / float(precision)
                )
            coord = np.asarray(coord)
            coords.append(coord)
        else:
            if p == 1:
                if fix_start_dir:
                    dir = start_dir
                else:
                    dir = np.random.randint(-10 * precision, 10 * precision, 3) / float(
                        precision
                    )
            else:
                dir = np.subtract(coords[p - 1], coords[p - 2])

            unit_dir = dir / np.linalg.norm(dir)
            circle_middle = dist * unit_dir + coords[p - 1]

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
            coords.append(coord)

        val = np.random.randint(-1 * precision, 1 * precision, len(var)) / float(
            precision
        )

        angle = []
        for vnum, v in enumerate(var):

            # calculate pitch from direction vector between two points
            if v in ["pitch"]:
                if p != 0:
                    q = coords[p] - coords[p - 1]
                    vp = np.copy(q)
                    vp[2] = 0
                    vpp = np.copy(vp)
                    vpp[0] = 0
                    num = np.dot(vp, vpp)
                    den = np.linalg.norm(vp) * np.linalg.norm(vpp)
                    if den == 0:
                        pitch = 0
                    else:
                        pitch = np.arccos(num / den)
                    angles[p - 1][vnum] = pitch
                else:
                    pitch = 0
                angle.append(pitch)

            else:
                if p != 0:
                    prev = angles[p - 1][vnum]
                else:
                    prev = np.random.randint(
                        start[v]["min"] * precision, start[v]["max"] * precision, 1
                    )[0] / float(precision)

                angle.append(prev + val[vnum] * scaling[vnum])

        angle = np.asarray(angle)
        angles.append(angle)

        point = np.concatenate([coord, angle], axis=0)
        points.append(point)

        d = {}
        for i in range(len(names)):
            d[names[i]] = point[i]

        task.append(d)

    if plot:
        plot_task(coords, ball_radius)

    task = np.asarray(task)
    if save:
        name = "task_" + str(n)
        np.save(os.path.join(taskDir, name), task)
