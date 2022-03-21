import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq


m, q, B, U, r0, z0 = 1.66*76*10**(-27), 1.6*27*10**(-19), 0.25, -5, .02, .02
k = 9*10**9


class Particle:
    def __init__(self, x, y, z, x1, y1, z1, mass, charge):
        self.x, self.y, self.z, self.x1, self.y1, self.z1 = x, y, z, x1, y1, z1
        self.m, self.q = mass, charge

    def __str__(self):
        return [self.x, self.y, self.z, self.x1, self.y1, self.z1]

    def get_coord(self):
        return [self.x, self.y, self.z]

    def get_speed(self):
        return [self.x1, self.y1, self.z1]

    def force(self, xs, ys, zs):
        dx = self.x - xs
        dy = self.y - ys
        dz = self.z - zs
        r = (dx**2 + dy**2 + dz**2)**0.5
        fx = (k * q**2 / m) * (dx / r**3)
        fy = (k * q**2 / m) * (dy / r**3)
        fz = (k * q**2 / m) * (dz / r**3)
        return [fx, fy, fz]


def fx2(xco, ysp, for1, for2):
    a = (q * B / m) * ysp - (2 * q * U / (m * (r0**2 + 2 * z0**2))) * xco + for1 + for2
    return a


def fy2(yco, xsp, for1, for2):
    a = -(q * B / m) * xsp - (2 * q * U / (m * (r0**2 + 2 * z0**2))) * yco + for1 + for2
    return a


def fz2(zco, for1, for2):
    a = (4 * q * U / (m * (r0**2 + 2 * z0**2))) * zco + for1 + for2
    return a


def funk(p1, p2, p3, shag, fx, fy, fz):
    # k - neiz, l - proiz
    yk11 = shag * p1.y1
    yl11 = shag * fy(p1.y, p1.x1, p1.force(p2.x, p2.y, p2.z)[1], p1.force(p3.x, p3.y, p3.z)[1])
    xk11 = shag * p1.x1
    xl11 = shag * fx(p1.x, p1.y1, p1.force(p2.x, p2.y, p2.z)[0], p1.force(p3.x, p3.y, p3.z)[0])
    zk11 = shag * p1.z1
    zl11 = shag * fz(p1.z, p1.force(p2.x, p2.y, p2.z)[2], p1.force(p3.x, p3.y, p3.z)[2])

    yk21 = shag * p2.y1
    yl21 = shag * fy(p2.y, p2.x1, p2.force(p1.x, p1.y, p1.z)[1], p2.force(p3.x, p3.y, p3.z)[1])
    xk21 = shag * p2.x1
    xl21 = shag * fx(p2.x, p2.y1, p2.force(p1.x, p1.y, p1.z)[0], p2.force(p3.x, p3.y, p3.z)[0])
    zk21 = shag * p2.z1
    zl21 = shag * fz(p2.z, p2.force(p1.x, p1.y, p1.z)[2], p2.force(p3.x, p3.y, p3.z)[2])

    yk31 = shag * p3.y1
    yl31 = shag * fy(p3.y, p3.x1, p3.force(p2.x, p2.y, p2.z)[1], p3.force(p1.x, p1.y, p1.z)[1])
    xk31 = shag * p3.x1
    xl31 = shag * fx(p3.x, p3.y1, p3.force(p2.x, p2.y, p2.z)[0], p3.force(p1.x, p1.y, p1.z)[0])
    zk31 = shag * p3.z1
    zl31 = shag * fz(p3.z, p3.force(p2.x, p2.y, p2.z)[2], p3.force(p1.x, p1.y, p1.z)[2])

    yk12 = shag * (p1.y1 + yl11 / 2)
    yl12 = shag * fy(p1.y + yk11 / 2, p1.x1 + xl11 / 2, p1.force(p2.x + xk21 / 2, p2.y + yk21 / 2, p2.z + zk21 / 2)[1], p1.force(p3.x + xk31 / 2, p3.y + yk31 / 2, p3.z + zk31 / 2)[1])
    xk12 = shag * (p1.x1 + xl11 / 2)
    xl12 = shag * fx(p1.x + xk11 / 2, p1.y1 + yl11 / 2, p1.force(p2.x + xk21 / 2, p2.y + yk21 / 2, p2.z + zk21 / 2)[0], p1.force(p3.x + xk31 / 2, p3.y + yk31 / 2, p3.z + zk31 / 2)[0])
    zk12 = shag * (p1.z1 + zl11 / 2)
    zl12 = shag * fz(p1.z + zk11 / 2, p1.force(p2.x + xk21 / 2, p2.y + yk21 / 2, p2.z + zk21 / 2)[2], p1.force(p3.x + xk31 / 2, p3.y + yk31 / 2, p3.z + zk31 / 2)[2])

    yk22 = shag * (p2.y1 + yl21 / 2)
    yl22 = shag * fy(p2.y + yk21 / 2, p2.x1 + xl21 / 2, p2.force(p1.x + xk11 / 2, p1.y + yk11 / 2, p1.z + zk11 / 2)[1], p2.force(p3.x + xk31 / 2, p3.y + yk31 / 2, p3.z + zk31 / 2)[1])
    xk22 = shag * (p2.x1 + xl21 / 2)
    xl22 = shag * fx(p2.x + xk21 / 2, p2.y1 + yl21 / 2, p2.force(p1.x + xk11 / 2, p1.y + yk11 / 2, p1.z + zk11 / 2)[0], p2.force(p3.x + xk31 / 2, p3.y + yk31 / 2, p3.z + zk31 / 2)[0])
    zk22 = shag * (p2.z1 + zl21 / 2)
    zl22 = shag * fz(p2.z + zk21 / 2, p2.force(p1.x + xk11 / 2, p1.y + yk11 / 2, p1.z + zk11 / 2)[2], p2.force(p3.x + xk31 / 2, p3.y + yk31 / 2, p3.z + zk31 / 2)[2])

    yk32 = shag * (p3.y1 + yl31 / 2)
    yl32 = shag * fy(p3.y + yk31 / 2, p3.x1 + xl31 / 2, p3.force(p2.x + xk21 / 2, p2.y + yk21 / 2, p2.z + zk21 / 2)[1], p3.force(p1.x + xk11 / 2, p1.y + yk11 / 2, p1.z + zk11 / 2)[1])
    xk32 = shag * (p3.x1 + xl31 / 2)
    xl32 = shag * fx(p3.x + yk31 / 2, p3.y1 + yl31 / 2, p3.force(p2.x + xk21 / 2, p2.y + yk21 / 2, p2.z + zk21 / 2)[0], p3.force(p1.x + xk11 / 2, p1.y + yk11 / 2, p1.z + zk11 / 2)[0])
    zk32 = shag * (p3.z1 + zl31 / 2)
    zl32 = shag * fz(p3.z + zk31 / 2, p3.force(p2.x + xk21 / 2, p2.y + yk21 / 2, p2.z + zk21 / 2)[2], p3.force(p1.x + xk11 / 2, p1.y + yk11 / 2, p1.z + zk11 / 2)[2])

    yk13 = shag * (p1.y1 + yl12 / 2)
    yl13 = shag * fy(p1.y + yk12 / 2, p1.x1 + xl12 / 2, p1.force(p2.x + xk22 / 2, p2.y + yk22 / 2, p2.z + zk22 / 2)[1], p1.force(p3.x + xk32 / 2, p3.y + yk32 / 2, p3.z + zk32 / 2)[1])
    xk13 = shag * (p1.x1 + xl12 / 2)
    xl13 = shag * fx(p1.x + xk12 / 2, p1.y1 + yl12 / 2, p1.force(p2.x + xk22 / 2, p2.y + yk22 / 2, p2.z + zk22 / 2)[0], p1.force(p3.x + xk32 / 2, p3.y + yk32 / 2, p3.z + zk32 / 2)[0])
    zk13 = shag * (p1.z1 + zl12 / 2)
    zl13 = shag * fz(p1.z + zk12 / 2, p1.force(p2.x + xk22 / 2, p2.y + yk22 / 2, p2.z + zk22 / 2)[2], p1.force(p3.x + xk32 / 2, p3.y + yk32 / 2, p3.z + zk32 / 2)[2])

    yk23 = shag * (p2.y1 + yl22 / 2)
    yl23 = shag * fy(p2.y + yk22 / 2, p2.x1 + xl22 / 2, p2.force(p1.x + xk12 / 2, p1.y + yk12 / 2, p1.z + zk12 / 2)[1], p2.force(p3.x + xk32 / 2, p3.y + yk32 / 2, p3.z + zk32 / 2)[1])
    xk23 = shag * (p2.x1 + xl22 / 2)
    xl23 = shag * fx(p2.x + xk22 / 2, p2.y1 + yl22 / 2, p2.force(p1.x + xk12 / 2, p1.y + yk12 / 2, p1.z + zk12 / 2)[0], p2.force(p3.x + xk32 / 2, p3.y + yk32 / 2, p3.z + zk32 / 2)[0])
    zk23 = shag * (p2.z1 + zl22 / 2)
    zl23 = shag * fz(p2.z + zk22 / 2, p2.force(p1.x + xk12 / 2, p1.y + yk12 / 2, p1.z + zk12 / 2)[2], p2.force(p3.x + xk32 / 2, p3.y + yk32 / 2, p3.z + zk32 / 2)[2])

    yk33 = shag * (p3.y1 + yl32 / 2)
    yl33 = shag * fy(p3.y + yk32 / 2, p3.x1 + xl32 / 2, p3.force(p2.x + xk22 / 2, p2.y + yk22 / 2, p2.z + zk22 / 2)[1], p3.force(p1.x + xk12 / 2, p1.y + yk12 / 2, p1.z + zk12 / 2)[1])
    xk33 = shag * (p3.x1 + xl32 / 2)
    xl33 = shag * fx(p3.x + yk32 / 2, p3.y1 + yl32 / 2, p3.force(p2.x + xk22 / 2, p2.y + yk22 / 2, p2.z + zk22 / 2)[0], p3.force(p1.x + xk12 / 2, p1.y + yk12 / 2, p1.z + zk12 / 2)[0])
    zk33 = shag * (p3.z1 + zl32 / 2)
    zl33 = shag * fz(p3.z + zk32 / 2, p3.force(p2.x + xk22 / 2, p2.y + yk22 / 2, p2.z + zk22 / 2)[2], p3.force(p1.x + xk12 / 2, p1.y + yk12 / 2, p1.z + zk12 / 2)[2])

    yk14 = shag * (p1.y1 + yl13)
    yl14 = shag * fy(p1.y + yk13, p1.x1 + xl13, p1.force(p2.x + xk23, p2.y + yk23, p2.z + zk23)[1], p1.force(p3.x + xk33, p3.y + yk33, p3.z + zk33)[1])
    xk14 = shag * (p1.x1 + xl13)
    xl14 = shag * fx(p1.x + xk13, p1.y1 + yl13, p1.force(p2.x + xk23, p2.y + yk23, p2.z + zk23)[0], p1.force(p3.x + xk33, p3.y + yk33, p3.z + zk33)[0])
    zk14 = shag * (p1.z1 + zl13)
    zl14 = shag * fz(p1.z + zk13, p1.force(p2.x + xk23, p2.y + yk23, p2.z + zk23)[2], p1.force(p3.x + xk33, p3.y + yk33, p3.z + zk33)[2])

    yk24 = shag * (p2.y1 + yl23)
    yl24 = shag * fy(p2.y + yk23, p2.x1 + xl23, p2.force(p1.x + xk13, p1.y + yk13, p1.z + zk13)[1], p2.force(p3.x + xk33, p3.y + yk33, p3.z + zk33)[1])
    xk24 = shag * (p2.x1 + xl23)
    xl24 = shag * fx(p2.x + xk23, p2.y1 + yl23, p2.force(p1.x + xk13, p1.y + yk13, p1.z + zk13)[0], p2.force(p3.x + xk33, p3.y + yk33, p3.z + zk33)[0])
    zk24 = shag * (p2.z1 + zl23)
    zl24 = shag * fz(p2.z + zk23, p2.force(p1.x + xk13, p1.y + yk13, p1.z + zk13)[2], p2.force(p3.x + xk33, p3.y + yk33, p3.z + zk33)[2])

    yk34 = shag * (p3.y1 + yl33)
    yl34 = shag * fy(p3.y + yk33, p3.x1 + xl33, p3.force(p2.x + xk23, p2.y + yk23, p2.z + zk23)[1], p3.force(p1.x + xk13, p1.y + yk13, p1.z + zk13)[1])
    xk34 = shag * (p3.x1 + xl33)
    xl34 = shag * fx(p3.x + xk33, p3.y1 + yl33, p3.force(p2.x + xk23, p2.y + yk23, p2.z + zk23)[0], p3.force(p1.x + xk13, p1.y + yk13, p1.z + zk13)[0])
    zk34 = shag * (p3.z1 + zl33)
    zl34 = shag * fz(p3.z + zk33, p3.force(p2.x + xk23, p2.y + yk23, p2.z + zk23)[2], p3.force(p1.x + xk13, p1.y + yk13, p1.z + zk13)[2])

    p1.x = p1.x + (1 / 6) * (xk11 + 2 * xk12 + 2 * xk13 + xk14)
    p1.y = p1.y + (1 / 6) * (yk11 + 2 * yk12 + 2 * yk13 + yk14)
    p1.z = p1.z + (1 / 6) * (zk11 + 2 * zk12 + 2 * zk13 + zk14)
    p1.x1 = p1.x1 + (1 / 6) * (xl11 + 2 * xl12 + 2 * xl13 + xl14)
    p1.y1 = p1.y1 + (1 / 6) * (yl11 + 2 * yl12 + 2 * yl13 + yl14)
    p1.z1 = p1.z1 + (1 / 6) * (zl11 + 2 * zl12 + 2 * zl13 + zl14)

    p2.x = p2.x + (1 / 6) * (xk21 + 2 * xk22 + 2 * xk23 + xk24)
    p2.y = p2.y + (1 / 6) * (yk21 + 2 * yk22 + 2 * yk23 + yk24)
    p2.z = p2.z + (1 / 6) * (zk21 + 2 * zk22 + 2 * zk23 + zk24)
    p2.x1 = p2.x1 + (1 / 6) * (xl21 + 2 * xl22 + 2 * xl23 + xl24)
    p2.y1 = p2.y1 + (1 / 6) * (yl21 + 2 * yl22 + 2 * yl23 + yl24)
    p2.z1 = p2.z1 + (1 / 6) * (zl21 + 2 * zl22 + 2 * zl23 + zl24)

    p3.x = p3.x + (1 / 6) * (xk31 + 2 * xk32 + 2 * xk33 + xk34)
    p3.y = p3.y + (1 / 6) * (yk31 + 2 * yk32 + 2 * yk33 + yk34)
    p3.z = p3.z + (1 / 6) * (zk31 + 2 * zk32 + 2 * zk33 + zk34)
    p3.x1 = p3.x1 + (1 / 6) * (xl31 + 2 * xl32 + 2 * xl33 + xl34)
    p3.y1 = p3.y1 + (1 / 6) * (yl31 + 2 * yl32 + 2 * yl33 + yl34)
    p3.z1 = p3.z1 + (1 / 6) * (zl31 + 2 * zl32 + 2 * zl33 + zl34)

    return [[p1.x, p1.y, p1.z, p1.x1, p1.y1, p1.z1], [p2.x, p2.y, p2.z, p2.x1, p2.y1, p2.z1], [p3.x, p3.y, p3.z, p3.x1, p3.y1, p3.z1]]


dt = .000000005
N = 100000
t = 0
tl = dt * N

xp, yp, zp = 0.002, 0.002, 0.002
x11, y11, z11 = 300, 400, 50

x2, y2, z2 = -0.001, -0.001, -0.001
x21, y21, z21 = 300, 400, 50

x3, y3, z3 = 0.001, 0.001, 0.001
x31, y31, z31 = 300, 400, 50

g1 = Particle(xp, yp, zp, x11, y11, z11, m, q)
g2 = Particle(x2, y2, z2, x21, y21, z21, m, q)
g3 = Particle(x3, y3, z3, x31, y31, z31, m, q)

graphx1 = [xp]
graphy1 = [yp]
graphz1 = [zp]

graphx2 = [x2]
graphy2 = [y2]
graphz2 = [z2]

graphx3 = [x3]
graphy3 = [y3]
graphz3 = [z3]

while t < tl:
    g = funk(g1, g2, g3, dt, fx2, fy2, fz2)
    graphx1.append(g1.x)
    graphy1.append(g1.y)
    graphz1.append(g1.z)
    graphx2.append(g2.x)
    graphy2.append(g2.y)
    graphz2.append(g2.z)
    graphx3.append(g3.x)
    graphy3.append(g3.y)
    graphz3.append(g3.z)
    t = t + dt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(graphx1, graphy1, graphz1)
ax.plot(graphx2, graphy2, graphz2)
ax.plot(graphx3, graphy3, graphz3)
plt.show()

signal = graphx1
#signal = graphz1
yf = rfft(signal)
xf = rfftfreq(N, dt)
plt.plot(xf[:1000], np.abs(yf)[:1000])
#plt.plot(xf, np.abs(yf))
plt.show()

signal = graphx2
#signal = graphz2
yf = rfft(signal)
xf = rfftfreq(N, dt)
plt.plot(xf[:1000], np.abs(yf)[:1000])
#plt.plot(xf, np.abs(yf))
plt.show()

signal = graphx3
#signal = graphz3
yf = rfft(signal)
xf = rfftfreq(N, dt)
plt.plot(xf[:1000], np.abs(yf)[:1000])
#plt.plot(xf, np.abs(yf))
plt.show()
