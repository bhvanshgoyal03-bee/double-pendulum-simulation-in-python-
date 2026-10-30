# simulating a double pendulum using sympy and scipy in python having springs
import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter

t,m,g,k = smp.symbols('t m g k')
theta1, theta2 = smp.symbols('theta1 theta2', cls=smp.Function)
r1,r2 = smp.symbols('r1 r2', cls=smp.Function)

theta1 = theta1(t)
theta2 = theta2(t)  
r1 = r1(t)
r2 = r2(t)

theta1_dot = smp.diff(theta1, t)
theta2_dot = smp.diff(theta2, t)
r1_dot = smp.diff(r1, t)
r2_dot = smp.diff(r2, t)

theta1_dot_dot = smp.diff(theta1_dot, t)
theta2_dot_dot = smp.diff(theta2_dot, t)
r1_dot_dot = smp.diff(r1_dot, t)
r2_dot_dot = smp.diff(r2_dot, t)

x1,y1,x2,y2 = smp.symbols('x1,x2,y1,y2', cls=smp.Function)

x1 = x1(theta1,r1)
y1 = y1(theta1,r1)

x2 = x2(theta1,theta2,r1,r2)
y2 = y2(theta1,theta2,r1,r2)

x1 = (1+r1)*smp.cos(theta1)
y1 = -(1+r1)*smp.sin(theta1)
x2 = (1+r1)*smp.cos(theta1) + (1+r2)*smp.cos(theta2)
y2 = -(1+r1)*smp.sin(theta1) - (1+r2)*smp.sin(theta2)

T = 1/2 * m * (smp.diff(x1, t)**2 + smp.diff(y1, t)**2 + smp.diff(x2, t)**2 + smp.diff(y2, t)**2)
V = m*g*y1 + m*g*y2 + 1/2 * k *(r1**2 + r2**2)
L = T - V

LE1 = smp.diff(smp.diff(L, theta1_dot), t) - smp.diff(L, theta1)  # euler lagrange equation for q = theta1
LE1 = LE1.simplify()

LE2 = smp.diff(smp.diff(L, theta2_dot), t) - smp.diff(L, theta2) # euler lagrange equation for q = theta2
LE2 = LE2.simplify()

LE3 = smp.diff(smp.diff(L, r1_dot), t) - smp.diff(L, r1)  # euler lagrange equation for q = r1
LE3 = LE3.simplify()

LE4 = smp.diff(smp.diff(L, r2_dot), t) - smp.diff(L, r2)  # euler lagrange equation for q = r2
LE4 = LE4.simplify()

sols = smp.solve([LE1, LE2, LE3, LE4], [theta1_dot_dot, theta2_dot_dot, r1_dot_dot, r2_dot_dot],simplify=False, rational=False)

dw1dt_f = smp.lambdify((theta1, theta2, r1, r2, theta1_dot, theta2_dot, r1_dot, r2_dot, g, k, m), sols[theta1_dot_dot], 'numpy')
dtheta1dt_f = smp.lambdify(theta1_dot, theta1_dot, 'numpy')

dw2dt_f = smp.lambdify((theta1, theta2, r1, r2, theta1_dot, theta2_dot, r1_dot, r2_dot, g, k, m), sols[theta2_dot_dot], 'numpy')
dtheta2dt_f = smp.lambdify(theta2_dot, theta2_dot, 'numpy')

dv1dt_f = smp.lambdify((theta1, theta2, r1, r2, theta1_dot, theta2_dot, r1_dot, r2_dot, g, k, m), sols[r1_dot_dot], 'numpy')
dr1dt_f = smp.lambdify(r1_dot, r1_dot, 'numpy') 

dv2dt_f = smp.lambdify((theta1, theta2, r1, r2, theta1_dot, theta2_dot, r1_dot, r2_dot, g, k, m), sols[r2_dot_dot], 'numpy')
dr2dt_f = smp.lambdify(r2_dot, r2_dot,'numpy')

def dSdt(S,t):
    theta1,w1,theta2,w2,r1,v1,r2,v2 = S
    return[
        dtheta1dt_f(w1),
        dw1dt_f(theta1, theta2, r1, r2, w1, w2, v1, v2, m, g, k),
        dtheta2dt_f(w2),
        dw2dt_f(theta1, theta2, r1, r2, w1, w2, v1, v2, m, g, k),
        dr1dt_f(v1),
        dv1dt_f(theta1, theta2, r1, r2, w1, w2, v1, v2, m, g, k),
        dr2dt_f(v2),
        dv2dt_f(theta1, theta2, r1, r2, w1, w2, v1, v2, m, g, k)
    ]

t = np.linspace(0, 20, 1000)
g = 9.81
k = 10  
m= 1

ans = odeint(dSdt,y0 = [np.pi/2, 0, (3/2) * np.pi/2, 0, 0, 5, 0, 5], t=t)

fig,axes = plt.subplots(4,2)

axes[0,0].plot(t,ans.T[0])
axes[0,0].set_ylabel('theta1')   

axes[0,1].plot(t,ans.T[1])
axes[0,1].set_ylabel('w1')

axes[1,0].plot(t,ans.T[2])
axes[1,0].set_ylabel('theta2')   

axes[1,1].plot(t,ans.T[3])
axes[1,1].set_ylabel('w2')

axes[2,0].plot(t,ans.T[4])
axes[2,0].set_ylabel('r1')   

axes[2,1].plot(t,ans.T[5])
axes[2,1].set_ylabel('v1')

axes[3,0].plot(t,ans.T[6])
axes[3,0].set_ylabel('r2')   

axes[3,1].plot(t,ans.T[7])
axes[3,1].set_ylabel('v2')

fig.supxlabel('t (s)')
plt.show()


th1 = ans[:, 0]
th2 = ans[:, 2]
R1  = ans[:, 4]
R2  = ans[:, 6]


X1 = (1 + R1) * np.cos(th1)
Y1 = -(1 + R1) * np.sin(th1)
X2 = X1 + (1 + R2) * np.cos(th2)
Y2 = Y1 - (1 + R2) * np.sin(th2)

step = 2                      # use every 2nd point to keep the gif small
frames = range(0, len(t), step)

fig_a, ax = plt.subplots(figsize=(6, 6))
lim = 1.1 * max(np.abs(X1).max(), np.abs(Y1).max(), np.abs(X2).max(), np.abs(Y2).max())
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.grid(alpha=0.3)

line, = ax.plot([], [], 'k-', lw=2)           
mass1, = ax.plot([], [], 'bo', ms=8)
mass2, = ax.plot([], [], 'ro', ms=8)
trail, = ax.plot([], [], 'r-', lw=0.8, alpha=0.5)   # path of m2
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def update(i):
    line.set_data([0, X1[i], X2[i]], [0, Y1[i], Y2[i]])
    mass1.set_data([X1[i]], [Y1[i]])
    mass2.set_data([X2[i]], [Y2[i]])
    trail.set_data(X2[:i+1], Y2[:i+1])
    time_text.set_text(f't = {t[i]:.2f} s')
    return line, mass1, mass2, trail, time_text

ani = animation.FuncAnimation(fig_a, update, frames=frames, interval=40, blit=True)
ani.save('double_spring_pendulum.gif', writer=PillowWriter(fps=25))
plt.show()