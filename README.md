# IF747-Grupo4

Implementation of a Intrustion Detection model for CAN networks 

## Dependencies

```bash
source venv
sudo apt-get update
sudo apt-get install can-utils tmux
``` 

# DBC Files
The DBC files are located in the `dbc` directory. These files define the structure of the CAN messages used in the simulation.

They can be seen in a human-readable format using the `analyzer.py` script:

```bash
python3 dbc/analyzer.py dbc/chassis.dbc
```

# Running the Scripts

## Start Benign Traffic Simulation
This starts the ECU simulators and the data collector.

```bash
./scripts/run_benign.sh

# View session:
tmux attach -t can_simulation

# Stop session:
tmux kill-session -t can_simulation
```

## Start Attack Simulation (in a new terminal)
This injects malicious traffic onto the CAN bus. Run this concurrently with the benign simulation.

```bash
./attacks/run_attacks.sh

# View session:
tmux attach -t can_attacks

# Stop session:
tmux kill-session -t can_attacks
```

## Train the Model
```bash
python3 scripts/train_model.py
```