# **VEKTOR-S: Safe Swarm Decision Stack**

VEKTOR-S is a **small but principled research demonstrator** for **distributed decision-making and safe coordination in multi-robot systems**.

The focus is **not** on scale or photorealism, but on **clarity**:
how high-level decisions emerge from *local sensing and neighbour interactions*, and how those decisions are grounded in *physical motion under safety constraints*.

The implementation is intentionally lightweight, designed to make the **control structure and information flow easy to inspect, modify, and reason about**.

---

## **What this demonstrates**

VEKTOR-S implements a **hierarchical control architecture** commonly used in swarm robotics and autonomous systems research.

### **Distributed decision layer**

* Each agent has **limited sensing** and **local communication**
* Target detections and neighbour interactions are fused into a **local belief**
* Collective behaviour emerges through **local consensus**
  *(no central controller)*

### **Role-based task coordination**

* Agents dynamically assume roles:

  * *Scout*
  * *Tracker*
  * *Interceptor*
* Role assignment is governed by **local information exchange**, inspired by **wolf-pack hunting behaviour**
* Minimum role requirements (e.g. interceptors vs trackers) are enforced in a **distributed manner**

### **Low-level safety layer**

* Nominal control commands are filtered through a fast, **CBF-like safety mechanism**
* Inter-agent collisions, obstacle proximity, and boundary violations are prevented **at runtime**
* Safety is treated as a **hard constraint**, not an optimisation objective

### **Embodied execution**

* Decisions are expressed as **acceleration commands** acting on simple point-mass dynamics
* The target follows a **continuous, random-turn motion model** and never “freezes” on intercept
* All behaviour is shaped by **sensing limits, dynamics, and safety constraints**

---

## **Architecture (conceptual)**

```
Sensing
   ↓
Belief / Opinion Dynamics
   ↓
Role & Mode Selection
   ↓
Nominal Behaviour
   ↓
Safety Filter
   ↓
Dynamics
```

At a high level:

* Local sensing provides **noisy, intermittent** target observations
* Belief and opinion dynamics combine:

  * detection evidence
  * neighbour influence
  * local risk
* Discrete modes and roles (`SEARCH / PURSUIT / INTERCEPT`) emerge from **distributed decision rules**
* Nominal motion commands are generated based on **role and belief**
* A safety filter enforces **collision and obstacle avoidance** before execution

This separation allows **decision-making logic** to be studied independently of **safety enforcement**, while still operating in a physically grounded loop.

---

## **What it is (and isn’t)**

This is **not**:

* a full simulator (not yet!)
* a product
* a benchmark suite

It **is**:

* a **minimal research platform**
* a testbed for **embodied, distributed decision-making**
* a foundation for extensions into **learning, formal verification, or higher-fidelity dynamics**

The emphasis is on **structure**, not polish.

---

## **Running the demo**

### **1) Create a virtual environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### **2) Run the simulation**

```bash
python run.py
```

This will generate:

* `results/demo.gif` — visualisation of swarm behaviour
* `results/metrics.png` — safety and task-level performance metrics

---

## **Metrics and analysis**

The included plots track:

* Minimum inter-agent distance and **safety margin**
* Average and minimum distance to the target
* Fraction of agents engaged in **pursuit / intercept**
* Fraction of agents with active target detections

A **runtime safety invariant**
`min_distance ≥ d_min`
is explicitly monitored, providing a simple hook for future **formal verification** or **CBF-based analysis**.

---

## **Motivation**

This project was developed as a **compact, inspectable example** of how:

* distributed decision-making,
* role-based coordination, and
* safety-critical control

can be integrated into a **single embodied control loop**.

It also serves as a **swarm-level extension** of the author’s ongoing work on pursuit and intercept autonomy (**VEKTOR**).

---

## **Possible extensions**

* Formal control barrier functions or verified safety filters
* Learning-based decision layers operating above the safety filter
* More complex agent dynamics or sensing models
* Hardware-in-the-loop or ROS-based deployment

---

## **Contact**

If you’re interested in discussing this work or extending it in a research context, feel free to get in touch.

---

