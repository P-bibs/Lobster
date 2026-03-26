# This is the script for running the ablation studies (pacman and pathfinder).

# frog is frog allocator (uses arena allocation)
# absorption=no means only one stratum is run on gpu, absoption-yes means adjacent strata are also run

dir="ablation_results"
mkdir $dir
export WANDB_MODE=disabled

# timeout
to="300s"

for i in `seq 5 25`; do
    echo "RUNNING $i"

    export REORDER="13<-3" NO_CHECK=1 TIME=1
    flags="experiments/pacman_maze/run_1d.py --grid-x $i --grid-y $i"
    STRATUM=16,16              timeout $to python $flags > "$dir/pacman${i}_no_frog_no_absorption.txt"    2> "$dir/pacman${i}_no_frog_no_absorption.err"  
    USE_FROG=1 STRATUM=16,16   timeout $to python $flags > "$dir/pacman${i}_yes_frog_no_absorption.txt"   2> "$dir/pacman${i}_yes_frog_no_absorption.err" 
    STRATUM=16,17              timeout $to python $flags > "$dir/pacman${i}_no_frog_yes_absorption.txt"   2> "$dir/pacman${i}_no_frog_yes_absorption.err" 
    USE_FROG=1  STRATUM=16,17  timeout $to python $flags > "$dir/pacman${i}_yes_frog_yes_absorption.txt"  2> "$dir/pacman${i}_yes_frog_yes_absorption.err"
    unset REORDER NO_CHECK TIME
    TIME=1 timeout $to python experiments/pacman_maze/run_1d.py --grid-x $i --grid-y $i > ablation_results/pacman${i}_scallop.txt 2> ablation_results/pacman${i}_scallop.err

    export NO_CHECK=1 TIME=1
    flags="experiments/pathfinder/128/run_with_cnn.py --grid-x $i --grid-y $i --provenance difftopkproofs "
    STRATUM=2,2             timeout $to python $flags > "$dir/pathfinder${i}_no_frog_no_absorption.txt" 2> "$dir/pathfinder${i}_no_frog_no_absorption.err"
    USE_FROG=1 STRATUM=2,2  timeout $to python $flags > "$dir/pathfinder${i}_yes_frog_no_absorption.txt" 2> "$dir/pathfinder${i}_yes_frog_no_absorption.err"
    STRATUM=2,3             timeout $to python $flags > "$dir/pathfinder${i}_no_frog_yes_absorption.txt" 2> "$dir/pathfinder${i}_no_frog_yes_absorption.err"
    USE_FROG=1 STRATUM=2,3  timeout $to python $flags > "$dir/pathfinder${i}_yes_frog_yes_absorption.txt" 2> "$dir/pathfinder${i}_yes_frog_yes_absorption.err"
    unset NO_CHECK TIME
    TIME=1 timeout $to python $flags > "$dir/pathfinder${i}_scallop.txt" 2> "$dir/pathfinder${i}_scallop.err"
    
done
