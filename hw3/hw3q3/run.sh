for sample_mode in random # random expert
do
    for mode in vanilla relabel
    do
        # run parallelly
        python GCBC.py --sample-mode $sample_mode --mode $mode --exp gcbc &
    done
done