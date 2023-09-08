# run python bandits.py with different modes
for mode in greedy optimistic ucb boltzmann
do
    echo "Running bandits.py with --mode $mode"
    python bandits.py --mode $mode
done