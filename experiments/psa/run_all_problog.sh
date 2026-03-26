file=psa_times_problog.txt
scratch=psa_times_problog_scratch.txt

run_one () {
    echo "Running Problog $1"
    date
    echo $1 >> $file
    start=`date +%s`
    problog ./experiments/data/psa/$1.pl > $scratch 2>&1
    end=`date +%s`
    runtime=$((end-start))
    echo $runtime >> $file
}


echo "" > $file

#run_one sunflow-core-facts
run_one sunflow-facts
#run_one biojava-core-facts # timeout (5 hours)
#run_one graphchi-facts # (timeout 1 hour)
#run_one avrora-facts
#run_one pmd-core-facts
#run_one jme3-core-facts
#run_one kafka-clients-facts
