# Implementing a Multi-Compartment Hodgkin-Huxley Model in Python
~ Leonardo Ferrisi
## **Step 1: Required Python Packages**
Before coding, ensure you have the necessary Python libraries installed:

- **NumPy (`numpy`)** – For numerical operations and handling arrays.
- **Matplotlib (`matplotlib.pyplot`)** – For visualizing simulation results.
- **SciPy (`scipy.integrate.solve_ivp`)** – For solving the system of differential equations.

**Import the required libraries at the start of your script:**
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
```

---

## **Step 2: Understanding the Hodgkin-Huxley Model**
Before coding, it's important to understand the **biophysical principles** behind the model:
1. **Gating variables** represent the probability of ion channels being open.
2. **Membrane potential dynamics** depend on sodium, potassium, and leak currents.
3. **Axial currents** allow interactions between neighboring compartments.
4. **Injected current** in the first compartment initiates the action potential.

---

## **Step 3: Implementing the Model - A Step-by-Step Guide**

### **1. Define the Hodgkin-Huxley Gating Rate Functions**
- Write functions `alpha_m(V)`, `beta_m(V)`, etc., for voltage-dependent transition rates.
- Ensure the functions return values in **1/ms** and accept voltage in **mV**.

**Example:**
```python
def alpha_m(V):
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
```

*INFO*: This can also be achieved using lambda functions

```python
alpha_m = lambda V : 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
alpha_m(V) # returns same value
```

### **2. Implement the Steady-State Gating Functions (`m_inf`, `h_inf`, `n_inf`)**
- Compute steady-state values using `alpha` and `beta` functions.
- Handle possible NaN cases (e.g., divide-by-zero errors).

**Example:**
```python
def m_inf(V):
    a = alpha_m(V)
    b = beta_m(V)
    return a / (a + b)
```

### **3. Define the `hh_cable_ode` Function to Represent the System of ODEs**
- Initialize an **empty array of zeros** using `np.zeros_like(y)`.
- Loop through each **compartment `i`** to compute:
  - **Ionic currents** (sodium, potassium, leak).
  - **Axial current** (between neighboring compartments).
  - **Injected current** (applied only in the first compartment within `t < 20 ms`).
  - **Gating variable derivatives** using `alpha` and `beta` functions.
- Return the **array of derivatives**.

+ Recall that in this instance, we are representing the compartments as chunks of 4 scalar quantities in a 1D array. The result of a 2 compartment axon may look like this:
```python
y0 = [V1, m1, h1, n1, V2, m2, h2, n2]
```
+ The same goes for `dydt`, which is a 1D array of the outputs arranged in a similar fashion. 
    + I tried doing a dictionary or 2D array but `solve_ivp` appeared incompatible with this. TODO: Read deeper into solve_ivp documentation.



**Example:**
```python
dydt = np.zeros_like(y)  # Initialize zero array for derivatives
for i in range(N):
    idx = 4 * i
    V = y[idx]   # Membrane voltage in mV
    m = y[idx + 1]
    h = y[idx + 2]
    n = y[idx + 3]

    # Compute derivatives
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n

    dydt[idx + 1] = dmdt
    dydt[idx + 2] = dhdt
    dydt[idx + 3] = dndt
```

### **4. Implement `simulate_and_plot` to Integrate and Visualize the Results**
- Define **axon parameters** (length, radius, resistivity).
- Compute **capacitance and conductance per compartment**.
- Define `y0`, the **initial state vector** (resting voltage + gating variables).
- Use `solve_ivp` to **integrate the equations numerically**.
- Extract **voltage traces** and **plot results** for each compartment.

**Key Steps:**
- Compute the **surface area** of each compartment.
- Convert **conductance per unit area** to **total conductance**.
- Compute **axial resistance** based on compartment geometry.
- Solve the system using `solve_ivp`.
- Extract and plot the **membrane potential over time**.

**Example:**
```python
def simulate_and_plot(N_values, step_size=0.1):
    plt.figure(figsize=(15, 10))
    for plot_idx, N in enumerate(N_values, 1):
        # Define axon properties
        L_comp_um = 10000 / N
        surface_area_um2 = 2 * np.pi * 100 * L_comp_um

        # Compute capacitance and conductances
        C = (10 * 1e-15) * surface_area_um2
        g_Na_total = (1200 * 1e-12) * surface_area_um2
        g_K_total  = (360 * 1e-12) * surface_area_um2
        g_L_total  = (3 * 1e-12) * surface_area_um2

        # Axial conductance
        R_axial = (1e6 * L_comp_um) / (np.pi * (100)**2)
        G_axial = 1 / R_axial

        # Initial conditions
        V_rest = -65
        y0 = np.array([V_rest, m_inf(V_rest), h_inf(V_rest), n_inf(V_rest)] * N)

        # Solve ODEs
        sol = solve_ivp(
            lambda t, y: hh_cable_ode(t, y, N, G_axial, g_Na_total, g_K_total, g_L_total, C, True),
            t_span=(0, 50),
            y0=y0,
            method='BDF',
            rtol=1e-6,
            atol=1e-6,
            max_step=step_size
        )

        # Plot results
        plt.subplot(2, 2, plot_idx)
        for i in range(N):
            plt.plot(sol.t, sol.y[4 * i], label=f'Comp {i+1}')
        plt.title(f'{N} Compartments')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.legend()
    plt.tight_layout()
    plt.show()
```

---

## **Step 4: Running and Debugging the Simulation**
### **Test the Code with Different Numbers of Compartments**
Start with a small number of compartments (`N=3`) and gradually increase:
```python
simulate_and_plot([3, 10, 30, 100], step_size=0.1)
```
### **Predict Before Running:**
- How will the signal propagate? 
    - should be offset in time, you shouldnt have both ends firing at the same time

### **Debugging Tips:**
1. **Check array sizes** (e.g., `y` should have `4*N` elements).
2. **Use print statements** inside `hh_cable_ode` to track variable updates.
3. **Start with `N=1`** to test a single-compartment model.
4. **Reduce the simulation time** to **5 ms** (`t_span=(0,5)`) for faster 
5. **Personally**, I would get comfortable with Debugger in VSCode. (https://code.visualstudio.com/docs/python/debugging), it absolutley carried me through getting this to work

---
