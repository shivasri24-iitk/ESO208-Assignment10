
import numpy as np
import matplotlib.pyplot as plt
from Q2b import crank_nicolson

dx = 2.0
dt = 0.1
total_time = 30.0

x, time, T = crank_nicolson(dx, dt, total_time)

required_times = [6, 12, 18, 24, 30]

plt.figure(figsize=(9, 6))

for t in required_times:
    step = int(round(t / dt))
    plt.plot(x, T[step, :], marker='o', linewidth=2,
             label=f't = {t} s')

plt.xlabel('Distance, x (cm)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Distribution in the Rod')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(x)
plt.tight_layout()

plt.savefig('Q2c_temperature_profile.png', dpi=300)
plt.show()

print("Temperature values at required times:")

for t in required_times:
    step = int(round(t / dt))
    print(f"\nt = {t} s")
    for i in range(len(x)):
        print(f"x = {x[i]:.1f} cm : T = {T[step, i]:.4f} °C")

print("\nQ2(c) completed successfully.")
