import numpy as np
import matplotlib.pyplot as plt

# initial parameters
g = 9.81
m = 0.15
b = 0.0012


def derivatives(state, b_value):
    x, y, vx, vy = state

    v = np.sqrt(vx**2 + vy**2)

    dxdt = vx
    dydt = vy
    dvxdt = -(b_value / m) * v * vx
    dvydt = -g - (b_value / m) * v * vy

    return np.array([dxdt, dydt, dvxdt, dvydt])


def rk4_step(state, dt, b_value):
    k1 = derivatives(state, b_value)
    k2 = derivatives(state + 0.5 * dt * k1, b_value)
    k3 = derivatives(state + 0.5 * dt * k2, b_value)
    k4 = derivatives(state + dt * k3, b_value)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_projectile(v0, theta_deg, b_value=b, dt=0.001, max_time=20):
    theta = np.deg2rad(theta_deg)

    x0 = 0.0
    y0 = 0.0
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    state = np.array([x0, y0, vx0, vy0], dtype=float)

    t_list = []
    x_list = []
    y_list = []
    vx_list = []
    vy_list = []

    K_list = []
    U_list = []
    E_mech_list = []
    E_diss_list = []
    E_total_list = []

    E_diss = 0.0
    t = 0.0

    while t <= max_time:
        x, y, vx, vy = state
        v = np.sqrt(vx**2 + vy**2)

        K = 0.5 * m * v**2
        U = m * g * y
        E_mech = K + U
        E_total = E_mech + E_diss

        t_list.append(t)
        x_list.append(x)
        y_list.append(y)
        vx_list.append(vx)
        vy_list.append(vy)

        K_list.append(K)
        U_list.append(U)
        E_mech_list.append(E_mech)
        E_diss_list.append(E_diss)
        E_total_list.append(E_total)

        state = rk4_step(state, dt, b_value)

        # dissipated energy accumulation
        # power dissipated = b v^3
        E_diss += b_value * v**3 * dt

        t += dt

        if state[1] < 0 and t > 0:
            break

    return {
        "t": np.array(t_list),
        "x": np.array(x_list),
        "y": np.array(y_list),
        "vx": np.array(vx_list),
        "vy": np.array(vy_list),
        "K": np.array(K_list),
        "U": np.array(U_list),
        "E_mech": np.array(E_mech_list),
        "E_diss": np.array(E_diss_list),
        "E_total": np.array(E_total_list),
    }

# Experiment 3
v0 = 40
theta = 50

result_drag = simulate_projectile(v0=v0, theta_deg=theta, b_value=b)

plt.figure(figsize=(8, 5))

plt.plot(result_drag["t"], result_drag["K"], label="Kinetic Energy K")
plt.plot(result_drag["t"], result_drag["U"], label="Potential Energy U")
plt.plot(result_drag["t"], result_drag["E_mech"], label="Mechanical Energy K + U")
plt.plot(result_drag["t"], result_drag["E_diss"], label="Dissipated Energy")
plt.plot(result_drag["t"], result_drag["E_total"], label="K + U + Dissipated")

plt.xlabel("Time [s]")
plt.ylabel("Energy [J]")
plt.title("Energy Change with Quadratic Drag")
plt.legend()
plt.grid(True)

plt.savefig("Figure_3.png", dpi=300, bbox_inches="tight")
plt.show()