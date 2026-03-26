file=tc_times_souffle.txt
scratch=tc_times_souffle_scratch.txt

run () {
    echo "Running Souffle $1"
    date
    echo "\"$1\": [" >> $file

    souffle -F ./experiments/data/gdlog/$1 -D /tmp ./experiments/gdlog/tc/tc.dl -o $1 -j32 > $scratch 2>&1
    for i in `seq 1 20`;
    do
        echo "Run $i"

        start=`date +%s%N`
        ./$1 -F ./experiments/data/gdlog/$1 -D /tmp -j32 > $scratch 2>&1
        end=`date +%s%N`
        runtime=$(((end-start)/1000))

        if [ $i -ne 1 ]; then
            echo "," >> $file
        fi
        echo "{" >> $file
        echo "\"total_time\": $runtime" >> $file
        echo "}" >> $file
    done
    echo "]," >> $file
}
echo "{" > $file
#run p2p-Gnutella25
#run p2p-Gnutella24
#run p2p-Gnutella30
run fe-sphere
#run loc-Brightkite
#run SF.cedge
#run fe_body
#run cit-HepTh
#run cit-HepPh
#run Gnutella31
#run com-dblp
#run usroad
#run vsp_finan

sed -i '$d' $file
echo "]" >> $file
echo "}" >> $file

< tc_times_souffle.txt ~/Builds/jq 'keys_unsorted[] as $key | $key + "," + (.[$key] | map(.total_time) | join(","))' | tr -d "\""
