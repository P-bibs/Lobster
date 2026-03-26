file=tc_times_scallop.txt
scratch=tc_times_scallop_scratch.txt

run () {
    echo "Running Scallop $1"
    date
    echo "\"$1\": [" >> $file

    for i in `seq 1 2`;
    do
        echo "Run $i"
        TIME=1 ./target/release/scli ./experiments/gdlog/tc/$1.scl > $scratch 2>&1
        total_time=`cat $scratch | grep "Total sample time:" | sed "s/Total sample time: //"`
        echo "total_time: $total_time"
        if [ $i -ne 1 ]; then
            echo "," >> $file
        fi
        echo "{" >> $file
        echo "\"total_time\": $total_time," >> $file
        echo "}" >> $file
    done
    echo "]," >> $file
}

echo "" > $file

echo "{" > $file

#run p2p-Gnutella25
#run p2p-Gnutella24
#run p2p-Gnutella30
run fe-sphere
run loc-Brightkite
run SF.cedge
run fe_body
#run cit-HepTh
#run cit-HepPh
#run Gnutella31
#run com-dblp
#run usroad
#run vsp_finan

sed -i '$d' $file
echo "]" >> $file

echo "}" >> $file

< tc_times_scallop.txt ~/Builds/jq 'keys_unsorted[] as $key | $key + "," + (.[$key] | map(.total_time) | join(","))' | tr -d "\""
