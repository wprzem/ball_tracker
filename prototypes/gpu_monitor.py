import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import subprocess
import datetime as dt
import numpy as np
from collections import deque

bash_command = 'nvidia-smi --query-gpu=utilization.gpu --format=csv'


def new_series(i):
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    percent_used = int(output.decode('UTF-8').split('\n')[1].split(' ')[0])
    gpu_usage.popleft()
    gpu_usage.append(percent_used)

    cur_time.popleft()
    cur_time.append(dt.datetime.now())

    gpu_plot.cla()
    gpu_plot.set_ylabel("GPU usage [%]")
    plt.grid(b=True)
    gpu_plot.plot(cur_time, gpu_usage)
    gpu_plot.set_ylim(0, 100)


num_records = 100
gpu_usage = deque(np.zeros(num_records))
interval = 500
delta = dt.timedelta(milliseconds=interval)
cur_time = deque([dt.datetime.now() - i * delta for i in reversed(range(num_records))])
fig = plt.figure(figsize=(6, 6))
gpu_plot = plt.subplot(111)

ani = FuncAnimation(fig, new_series, interval=interval)
plt.show()

