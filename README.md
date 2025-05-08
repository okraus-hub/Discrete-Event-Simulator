# Network Packet Simulation

This project simulates the behavior of packets in a simple network topology using discrete event simulation. It includes two experiments:
- A **single-link** experiment between two hosts.
- A **switch-based** experiment with multiple sources and intermediate routing.

## Features

- Discrete event-driven simulation framework.
- Support for host and switch behavior with customizable delays.
- Packet-level tracking including queuing, transmission, propagation, and end-to-end delays.
- CSV output of simulation results.
- Visual delay distribution plotting using matplotlib.

## Requirements

- Python 3.7+
- `matplotlib`

Install dependencies with:
```bash
pip install matplotlib
