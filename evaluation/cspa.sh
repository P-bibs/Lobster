# Evaluate Lobster on the "context sensitive pointer analysis" task on each graph
mkdir cspa_results

export USE_FROG=1 CUDA_VISIBLE_DEVICES=0 TIME=1 STRATUM=2,2 EARLY_EXIT=1

run_cspa () {
    echo "RUNNING $1"
    file="cspa_results/$1.txt"
    echo "" > $file
    for i in `seq 0 10`; do
        ./target/release/scli experiments/gdlog/cspa/$1.scl > out.txt 2> out.err
        #./target/release/scli experiments/gdlog/cspa/$1.scl > out.txt 2> out.err & tail --pid=$! -n +1 -F out.txt | (grep -q -m 1 "Stratum set: 2..3" && kill $\!)
        cat out.txt >> $file
    done
    cat $file | grep "Stratum set: 2..3" | sed "s/Stratum set: 2..3\t Time in run(): //" | tr -d "us"
}

run_cspa httpd
run_cspa linux
run_cspa postgres
