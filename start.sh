
#!/bin/bash


# # Start a virtual display with desired resolution
# Xvfb :1 -screen 0 1280x800x16 &

# # Set DISPLAY environment variable to use the virtual display
# export DISPLAY=:1

# Function to run a command in a new tmux session
function run_in_tmux {
    local session_name="$1"
    local venv="$2"
    local script="$3"
    

    # Check if session already exists and kill it if it does
    if tmux has-session -t "$session_name" 2>/dev/null; then
        tmux kill-session -t "$session_name"
    fi
    
    # Start a new tmux session with a unique name
    echo "Creating session '$session_name'..."
    tmux new-session -d -s "$session_name" "source $venv/bin/activate; python $script; read -p 'Press Enter to close this terminal...'"
    echo "Session '$session_name' created."

}

ls

# Run first Python script in terminal 1
# Appending timestamp to session name to ensure uniqueness
run_in_tmux "Terminal1Script1_1" ".venv" "main.py"

# Run second Python script in terminal 2
# Appending timestamp to session name to ensure uniqueness
# run_in_tmux "Terminal2Script2_1" ".venv" "joystick.py"

