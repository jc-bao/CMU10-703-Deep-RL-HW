for sample_mode in random expert
do
    for mode in relabel #vanilla relabel
    do
        python GCBC.py --sample-mode $sample_mode --mode $mode --exp gcbc
    done
done