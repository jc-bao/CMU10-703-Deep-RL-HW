# REINFORCE
python run.py &

# Baseline
python run.py --baseline &

# N-step A2C
python run.py --a2c --n 1 &
python run.py --a2c --n 10 &
python run.py --a2c --n 100 &