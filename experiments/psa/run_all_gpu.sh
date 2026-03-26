file=psa_times_gpu.txt
scratch=psa_times_gpu_scratch.txt

run_one () {
    echo "Running $1"
    date
    echo $1 >> $file
    NO_CHECK=1 TIME=1 STRATUM=13,15 CUDA_VISIBLE_DEVICES=1 ./target/release/scli ./experiments/data/psa/$1/$1.scl --provenance minmaxprob > $scratch 2>&1
    echo -n "GPU: " >> $file
    cat $scratch | grep "Total sample time:" | sed "s/Total sample time: //" | tr -d "s" >> $file
}


echo "" > $file

run_one sunflow-core-facts
run_one sunflow-facts
run_one biojava-core-facts
run_one graphchi-facts
run_one avrora-facts
run_one pmd-core-facts
run_one jme3-core-facts
run_one kafka-clients-facts
