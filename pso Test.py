import numpy as np
from SO_PSO import pso
from betterLHS import real_LHS
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.animation import FuncAnimation, PillowWriter

def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix


def rosenbrock_fun(DV):
    x, y = DV
    return (1-x)**2 + 100*(y-x**2)**2


def rosenbrock_const(OV, DV):
    x, y = DV
    feasible = True
    if ((x-1)**3 - y + 1) <= 0:
        feasible = True
    else:
        feasible = False
    if (x + y - 2) <= 0:
        feasible = True
    else:
        feasible = False
    return feasible


LB = np.array([-1.5, -0.5])
UB = np.array([1.5, 2.5])
InitialPop = real_LHS([[LB[0], UB[0]], [LB[1], UB[1]]], 100)

# x = np.linspace(LB[0], UB[0], 200)
# y = np.linspace(LB[1], UB[1], 200)
# X, Y = np.meshgrid(x, y)
# z = rosenbrock_fun([X, Y])
# # Z = normalize_2d(z)
#
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#
# # plt.contourf(X, Y, z, 100, cmap='YlGnBu')
# fig.colorbar(surf, shrink=0.7, aspect=10)
# ax.set_xlim(-1.5, 1.5)
# ax.set_ylim(-0.5, 2.5)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')
# ax.axes.zaxis.set_ticklabels([])
# ax.axes.zaxis.set_label('f(x, y)')
# # pop = np.array(InitialPop2)
# # plt.scatter(pop[:, 0], pop[:, 1])
# plt.show()

#
# N, fg_best, xg_best, f_pop, f_pop_best, x_pop, x_pop_best, pos= pso(rosenbrock_fun, rosenbrock_const, InitialPop, LB,
#                                                                     UB, 5000, 0.7, 1.5, 2)
# print('Rosenbrock Test function Correct Answer f(1, 1) = 0')
# print('Optimized Answer f(,', xg_best, ') =', fg_best)


def MishraBird_fun(DV):
    x, y = DV
    return np.sin(y)*np.exp((1 - np.cos(x))**2) + np.cos(x)*np.exp((1-np.sin(y))**2) + (x-y)**2


def MishraBird_const(OV, DV):
    x, y = DV
    if (x+5)**2 + (y+5)**2 < 25:
        feasible = True
    else:
        feasible = False
    return feasible


def constraint_fake(OV, DV):
    return True


LB2 = np.array([-10, -10])
UB2 = np.array([0, 0])
InitialPop2 = real_LHS([[LB2[0], UB2[0]], [LB2[1], UB2[1]]], 100)

N2, fg_best2, xg_best2, f_pop2, f_pop_best2, x_pop2, x_pop_best2, pos = pso(MishraBird_fun, constraint_fake,
                                                                           InitialPop2, LB2, UB2, 1000, 0.7, 0.2, 0.2)
print('Mishras Bird Test function Correct Answer f(-3.1302468, -1.5821422) = -106.7645367')
print('Optimized Answer f(,', xg_best2, ') =', fg_best2)

x = np.linspace(LB2[0], UB2[0], 500)
y = np.linspace(LB2[1], UB2[1], 500)
X, Y = np.meshgrid(x, y)
z = MishraBird_fun([X, Y])
# Z = normalize_2d(z)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# plt.contourf(X, Y, z, 100, cmap='YlGnBu')
cbar = fig.colorbar(surf, shrink=0.7, aspect=10)
cbar.ax.set_title('f(x,y)')
ax.set_xlim(-10, 0)
ax.set_ylim(-10, 0)
ax.set_xlabel('x', fontsize = 15)
ax.set_ylabel('y', fontsize = 15)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
ax.axes.zaxis.set_ticklabels([])
ax.axes.zaxis.set_label('f(x, y)')
#pop = np.array(InitialPop2)


fig, ax = plt.subplots()
cont = plt.contourf(X, Y, z, 100, cmap=cm.coolwarm)
cbar = fig.colorbar(cont, shrink=0.7, aspect=10)
cbar.ax.set_title('f(x,y)')
ax.set_xlim(-10, 0)
ax.set_ylim(-10, 0)
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
scat = plt.scatter(pos[0, :, 0], pos[0, :, 1])
stringiter = 'Iteration ' + str(0)
txt = plt.text(-9.8, -0.4, stringiter)

def animate(iteration):
    x_i = pos[iteration, :, 0]
    y_i = pos[iteration, :, 1]
    scat.set_offsets(np.c_[x_i, y_i])
    stringiter = 'Iteration ' + str(iteration)
    txt.set_text(stringiter)

anim = FuncAnimation(fig, animate, interval=500, frames=N2 - 21)
plt.draw()
writer1 = PillowWriter(fps=8)
anim.save('swarmsearch.gif', writer=writer1)
plt.show()


print()
