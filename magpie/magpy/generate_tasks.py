# maybe at some point add trajectory controller
# for now just randomly sample some tasks
# save each task in a npy file inside task folder

import os
import numpy as np

startingDir = os.path.dirname(os.path.realpath(__file__))
print(startingDir)
os.chdir(startingDir)
taskDir = os.path.join(startingDir, "tasks")

num_tasks = 10


point = [{"roll": 0, "pitch": 0, "Va": 0}]
task = [point, point, point]

task = np.asarray(task)

np.save(os.path.join(taskDir, "task39"), task)
np.save(os.path.join(taskDir, "task33"), task)
np.save(os.path.join(taskDir, "task32"), task)
np.save(os.path.join(taskDir, "task36"), task)
np.save(os.path.join(taskDir, "task21"), task)
np.save(os.path.join(taskDir, "task4"), task)
