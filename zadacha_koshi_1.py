from diffur_methods import yn3
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq


#m, q, B, U, r0, z0 = 1, 1, 1, -5, 10, 10
m, q, B, U, r0, z0 = 76, 27*10**8, 0.25, -5, .02, .02


def fy1(sp):
    return sp


def fx1(sp):
    return sp


def fz1(sp):
    return sp


def fx2(arg, ysp, yco, xsp, xco, zsp, zco):
    a = (q * B / m) * ysp - (2 * q * U / (m * (r0**2 + 2 * z0**2))) * xco
    return a


def fy2(arg, ysp, yco, xsp, xco, zsp, zco):
    a = -(q * B / m) * xsp - (2 * q * U / (m * (r0**2 + 2 * z0**2))) * yco
    return a


def fz2(arg, ysp, yco, xsp, xco, zsp, zco):
    a = (4 * q * U / (m * (r0**2 + 2 * z0**2))) * zco
    return a


dt = .000000005
N = 100000
t = 0
tl = dt * N
y, x, z = 0.001, 0.001, 0.001
y1, x1, z1 = 300, 400, 50
graphx = [x]
graphy = [y]
graphz = [z]

while t < tl:
    g = yn3(fy2, fy1, fx2, fx1, fz2, fz1, t, y, y1, x, x1, z, z1, dt)
    y1 = g[0]
    y = g[1]
    x1 = g[2]
    x = g[3]
    z1 = g[4]
    z = g[5]
    graphx.append(x)
    graphy.append(y)
    graphz.append(z)
    t = t + dt


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(graphx, graphy, graphz)
plt.show()

signal = graphx
#signal = graphz
yf = rfft(signal)
xf = rfftfreq(N, dt)
plt.plot(xf[:1000], np.abs(yf)[:1000])
#plt.plot(xf[:10000], np.abs(yf)[:10000])
plt.show()

signal = graphz
yf = rfft(signal)
xf = rfftfreq(N, dt)
plt.plot(xf[:1000], np.abs(yf)[:1000])
#plt.plot(xf[:10000], np.abs(yf)[:10000])
plt.show()
