# maybe at some point add trajectory controller
# for now just randomly sample some tasks
# save each task in a npy file inside task folder

import os
import numpy as np

startingDir = os.path.dirname(os.path.realpath(__file__))
print(startingDir)
os.chdir(startingDir)
taskDir = os.path.join(startingDir, "tasks")

num_tasks = 200
var = ["roll", "pitch", "Va", "position_n", "position_e", "position_d"]
start = {
    "roll": {"min": -10, "max": 10},
    "pitch": {"min": -10, "max": 10},
    "Va": {"min": 5, "max": 25},
    "position_n": {"min": -10, "max": 10},
    "position_e": {"min": -10, "max": 10},
    "position_d": {"min": 50, "max": 200},
}
scaling = [1, 1, 1, 1, 1, 1]

# TODO: make sure points have a distance of 10m

np.random.seed(10)
tasks = []

for n in range(num_tasks):
    task = []
    pos = np.random.randint(5, 15, 1)[0]
    for p in range(pos):
        point = {}
        val = np.random.rand(len(var))
        for v in enumerate(var):
            if p != 0:
                prev = task[p - 1][v[1]]
            else:
                prev = np.random.randint(start[v[1]]["min"], start[v[1]]["max"], 1)[0]
            point[v[1]] = prev + val[v[0]] * scaling[v[0]]
        task.append(point)
    task = np.asarray(task)
    name = "task_" + str(n)
    np.save(os.path.join(taskDir, name), task)
