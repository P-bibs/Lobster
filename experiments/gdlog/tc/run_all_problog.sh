file=tc_times_problog.txt
scratch=tc_times_problog_scratch.txt

run () {
    echo "Running Problog $1"
    date

    wc -l ./experiments/data/gdlog/tc/$1.pl

    start=`date +%s%N`
    timeout 130m problog ./experiments/data/gdlog/tc/$1.pl > "$scratch_$1" 2>&1
    end=`date +%s%N`
    runtime=$((end-start))
    echo "\"$1\": " >> $file
    echo "{" >> $file
    echo "\"total_time\": $runtime" >> $file
    echo "}" >> $file
}
echo "{" > $file
run p2p-Gnutella25 &
run p2p-Gnutella24 &
run p2p-Gnutella30 &
run fe-sphere      &
run loc-Brightkite &
run SF.cedge       &
run fe_body        &
run cit-HepTh      &
run cit-HepPh      &
run Gnutella31     &
run com-dblp       &
run usroad         &
run vsp_finan      &
wait

#< tc_times_problog.txt ~/Builds/jq 'keys_unsorted[] as $key | $key + "," + (.[$key] | map(.total_time) | join(","))' | tr -d "\""
