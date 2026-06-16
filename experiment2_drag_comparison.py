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

    t = 0.0

    while t <= max_time:
        x, y, vx, vy = state

        t_list.append(t)
        x_list.append(x)
        y_list.append(y)
        vx_list.append(vx)
        vy_list.append(vy)

        state = rk4_step(state, dt, b_value)
        t += dt

        if state[1] < 0 and t > 0:
            x, y, vx, vy = state

            t_list.append(t)
            x_list.append(x)
            y_list.append(y)
            vx_list.append(vx)
            vy_list.append(vy)

            break

    return {
        "t": np.array(t_list),
        "x": np.array(x_list),
        "y": np.array(y_list),
        "vx": np.array(vx_list),
        "vy": np.array(vy_list),
    }


def extract_summary_interpolated(result):
    """
    Range, maximum height, and flight time
    using linear interpolation at y = 0.
    """
    x = result["x"]
    y = result["y"]
    t = result["t"]

    max_height = np.max(y)

    projectile_range = x[-1]
    flight_time = t[-1]

    for i in range(1, len(y)):
        if y[i] < 0:
            x1, y1, t1 = x[i-1], y[i-1], t[i-1]
            x2, y2, t2 = x[i], y[i], t[i]

            alpha = -y1 / (y2 - y1)

            projectile_range = x1 + alpha * (x2 - x1)
            flight_time = t1 + alpha * (t2 - t1)
            break

    return projectile_range, max_height, flight_time


# Experiment 2
v0 = 40
theta = 50

result_no_drag = simulate_projectile(v0=v0, theta_deg=theta, b_value=0.0)
result_drag = simulate_projectile(v0=v0, theta_deg=theta, b_value=b)

plt.figure(figsize=(8, 5))

plt.plot(result_no_drag["x"], result_no_drag["y"], label="No air resistance")
plt.plot(result_drag["x"], result_drag["y"], label="Quadratic drag")

plt.xlabel("Horizontal Distance x [m]")
plt.ylabel("Height y [m]")
plt.title(f"No Drag vs Quadratic Drag, v0={v0} m/s, theta={theta}°")
plt.legend()
plt.grid(True)

plt.savefig("Figure_2.png", dpi=300, bbox_inches="tight")
plt.show()


# Table 1
range_no_drag, height_no_drag, time_no_drag = extract_summary_interpolated(result_no_drag)
range_drag, height_drag, time_drag = extract_summary_interpolated(result_drag)

print("\nNumerical Comparison: No Drag vs Quadratic Drag")
print("=" * 75)
print(f"{'Case':<25}{'Range [m]':>15}{'Max Height [m]':>18}{'Flight Time [s]':>18}")
print("-" * 75)
print(f"{'No air resistance':<25}{range_no_drag:>15.2f}{height_no_drag:>18.2f}{time_no_drag:>18.2f}")
print(f"{'Quadratic drag':<25}{range_drag:>15.2f}{height_drag:>18.2f}{time_drag:>18.2f}")
print("=" * 75)