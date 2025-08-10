#!/bin/bash

SESSION_NAME="can_simulation"
CAN_INTERFACE="vcan0"

cd "$(dirname "$0")"

echo "Setting up virtual CAN interface: ${CAN_INTERFACE}..."
sudo modprobe vcan
sudo ip link add dev ${CAN_INTERFACE} type vcan 2>/dev/null || true
sudo ip link set up ${CAN_INTERFACE}
echo "Interface ${CAN_INTERFACE} is up."

tmux has-session -t ${SESSION_NAME} 2>/dev/null
if [ $? -eq 0 ]; then
    echo "An old '${SESSION_NAME}' session was found. Killing it."
    tmux kill-session -t ${SESSION_NAME}
fi

echo "Starting new tmux session: ${SESSION_NAME}"
tmux new-session -d -s ${SESSION_NAME}

tmux select-pane -t 0
tmux rename-window 'ECU Simulators'
tmux send-keys "echo '--- Chassis ECU ---' && python3 chassis.py -c ${CAN_INTERFACE} -o /tmp/chassis.csv" C-m

tmux split-window -v
tmux select-pane -t 1
tmux send-keys "echo '--- PowerTrain ECU ---' && python3 powertrain.py ../dbc/*.dbc -c ${CAN_INTERFACE} -o /tmp/powertrain.csv" C-m

tmux split-window -h
tmux select-pane -t 2
tmux send-keys "echo '--- Body ECU ---' && python3 body.py ../dbc/*.dbc -c ${CAN_INTERFACE} -o /tmp/body.csv" C-m

tmux select-pane -t 0
tmux split-window -h
tmux select-pane -t 1

tmux send-keys "echo '--- IDS ECU ---' && sleep 1 && python3 model.py -c ${CAN_INTERFACE}" C-m

echo ""
echo "To view the output, attach to the tmux session with the command:"
echo "   tmux attach -t ${SESSION_NAME}"
echo ""
echo "To stop the entire simulation, run:"
echo "   tmux kill-session -t ${SESSION_NAME}"
echo ""
