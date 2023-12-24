import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker


def normalization(x):
    return (x[:, ] - np.mean(x[:, ])) / np.std(x[:, ])


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.style.use('seaborn-muted')
plt.figure(figsize=(7, 4))
plt.grid(ls="--", lw=0.15, color="#4E616C")
# ax = plt.gca()
# ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
# ax.xaxis.set_tick_params(length=2, color="#4E616C", labelcolor="#4E616C", labelsize=7)
# ax.yaxis.set_tick_params(length=2, color="#4E616C", labelcolor="#4E616C", labelsize=7)
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
x = pd.read_excel('plot_x.xlsx')
y = pd.read_excel('plot_y.xlsx', sheet_name='y_1')
color = ['#F27970', '#BB9727', '#54B345', '#32B897', '#05B9E2']
for i in range(x.shape[1]):
    plt.scatter(x.iloc[:, i], y.iloc[:, i], color=color[i])
plt.xlabel('Time/s')
plt.ylabel('Dominating Number')
plt.xticks([round(0.1 * i, 1) for i in range(0, 20, 2)], [str(round(0.1 * i, 1)) for i in range(0, 20, 2)])
plt.legend(['CC²FS', 'FastMWDS', 'FastDS', 'DSP-RL', 'Greedy'])
plt.savefig('sctter.png', dpi=200)
plt.show()

# plt.figure(figsize=(7, 4))
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.style.use('seaborn-muted')
# # plt.grid(ls="--", lw=0.15, color="#4E616C")
# # ax = plt.gca()
# # ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
# # ax.xaxis.set_tick_params(length=2, color="#4E616C", labelcolor="#4E616C", labelsize=7)
# # ax.yaxis.set_tick_params(length=2, color="#4E616C", labelcolor="#4E616C", labelsize=7)
#
# y1 = [0.697142857, 1.192857143, 1.604285714, 0.161428571, 0.071428571]
# y2 = [1, 0.999871685, 0.999857286, 0.983002384, 0.837821364]
# x = np.arange(len(y1))
# tick = ['CC²FS', 'FastMWDS', 'FastDS', 'DSP-RL', 'Greedy']
# plt.xlabel('Model')
# plt.ylabel('Time')
# total_width, n = 0.6, 2
# width = total_width / n
# x1 = x - width/2
# x2 = x1 + width
#
# plt.xticks(x, tick)
# plt.bar(x1, y1, width=0.25, label='Time', color='#8ecfc9')
# plt.legend(loc='upper left')
#
# ax = plt.twinx()
# ax.set_ylabel('Approximation rate of Optimal value')
# ax.set_ylim(0, 1.1)
#
# plt.bar(x2, y2, width=0.25, label='Approximation rate', color='#ffbe71')
# ax.legend(loc='best')
# plt.savefig('bar.png', dpi=200)
# plt.show()
