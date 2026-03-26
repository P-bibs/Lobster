file=cspa_times_scallop.txt
scratch=cspa_times_scallop_scratch.txt

run () {
    echo "Running Scallop $1"
    date
    echo $1 >> $file
    TIME=1 ./target/release/scli ./experiments/gdlog/cspa/$1.scl > $scratch 2>&1
    cat $scratch | grep "Total sample time:" | sed "s/Total sample time: //" | tr -d "s" >> $file
}

echo "" > $file
run linux
#run httpd
#run postgresql
