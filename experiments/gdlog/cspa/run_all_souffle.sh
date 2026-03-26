file=cspa_times_souffle.txt
scratch=cspa_times_souffle_scratch.txt

run () {
    echo "Running Souffle $1"
    date
    echo $1 >> $file
    souffle -F ./experiments/data/gdlog/cspa/$1 -D /tmp ./experiments/gdlog/cspa/cspa.dl -o $1 -j16 > $scratch 2>&1
    start=`date +%s`
    ./$1 -F ./experiments/data/gdlog/cspa/$1 -D /tmp -j16 > $scratch 2>&1
    end=`date +%s`
    runtime=$((end-start))
    echo $runtime >> $file
}

echo "" > $file
run httpd
run linux
run postgresql
