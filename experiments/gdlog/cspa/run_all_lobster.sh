file=cspa_times_lobster.txt
scratch=cspa_times_lobster_scratch.txt

run () {
    echo "Running Lobster $1"
    date
    echo $1 >> $file

    for i in `seq 1 5`;
    do
        echo "Run $i" | tee -a $file
        CUB=1 SUBSPACE=10 STRATUM=2,2 TIME=1 NO_CHECK=1 ./target/release/scli ./experiments/gdlog/cspa/$1.scl > $scratch 2>&1
        cat $scratch | grep "Stratum set: 2..3" | sed "s/Stratum set: 2..3\t Time in run(): //" | tr -d "us" >> $file
        cat $scratch | grep "Total sample time:" | sed "s/Total sample time: //" >> $file
    done
}

echo "" > $file
run httpd
run linux
run postgres
