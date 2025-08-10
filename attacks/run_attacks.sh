#!/bin/bash

SESSION_NAME="can_attacks"
CAN_INTERFACE="vcan0"

cd "$(dirname "$0")"

tmux has-session -t ${SESSION_NAME} 2>/dev/null
if [ $? -eq 0 ]; then
    echo "An old '${SESSION_NAME}' session was found. Killing it."
    tmux kill-session -t ${SESSION_NAME}
fi

echo "Starting new tmux session for attacks: ${SESSION_NAME}"
tmux new-session -d -s ${SESSION_NAME}

tmux rename-window 'Attacks'
ATTACK_DELAY=3

tmux send-keys "echo '--- DoS Attack (starts in ${ATTACK_DELAY}s) ---' && sleep ${ATTACK_DELAY} && python3 dos.py -c ${CAN_INTERFACE} -d 300" C-m

tmux split-window -h
tmux send-keys "echo '--- Fuzzing Attack (starts in ${ATTACK_DELAY}s) ---' && sleep ${ATTACK_DELAY} && python3 fuzzing.py -c ${CAN_INTERFACE}" C-m

tmux select-pane -t 0
tmux split-window -v
tmux send-keys "echo '--- Spoofing Attack (starts in ${ATTACK_DELAY}s) ---' && sleep ${ATTACK_DELAY} && python3 spoofing.py -c ${CAN_INTERFACE}" C-m

tmux select-pane -t 1
tmux split-window -v
tmux send-keys "echo '--- Replay Attack (starts in ${ATTACK_DELAY}s) ---' && sleep ${ATTACK_DELAY} && python3 replay.py -c ${CAN_INTERFACE} -d 15" C-m

tmux select-layout tiled

echo ""
echo "Attack scripts are running in tmux session '${SESSION_NAME}'."
echo ""
echo "To view the attacks, attach to the session with the command:"
echo "   tmux attach -t ${SESSION_NAME}"
echo ""
echo "To stop all attacks, run:"
echo "   tmux kill-session -t ${SESSION_NAME}"
echo ""
