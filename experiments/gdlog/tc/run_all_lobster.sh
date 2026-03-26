file=tc_times_lobster.txt
scratch=tc_times_lobster_scratch.txt

run () {
    echo "Running Lobster $1"
    date
    echo "\"$1\": [" >> $file

    for i in `seq 1 20`;
    do
        echo "Run $i"
        CUB=1 STRATUM=1,1 TIME=1 NO_CHECK=1 ./target/release/scli ./experiments/gdlog/tc/$1.scl > $scratch 2>&1
        lobster_time=`cat $scratch | grep "Stratum set: 1..2" | sed "s/Stratum set: 1..2\t Time in run(): //" | tr -d "us"`
        total_time=`cat $scratch | grep "Total sample time:" | sed "s/Total sample time: //"`
        exec_time=`cat $scratch | grep "Timer \[main\]: " | sed "s/Timer \[main\]: //" | tr -d "us"`
        lobster_time=${lobster_time:-0}
        total_time=${total_time:-0}
        exec_time=${exec_time:-0}

        if [ $i -ne 1 ]; then
            echo "," >> $file
        fi
        echo "{" >> $file
        echo "\"lobster_time\": $lobster_time," >> $file
        echo "\"total_time\": $total_time," >> $file
        echo "\"exec_time\": $exec_time" >> $file
        echo "}" >> $file
    done
    echo "]," >> $file
}

run_arena () {
    export SUBSPACE=20 ARENA_SIZE=30 USE_ARENA=1
    run $1
}
run_normal () {
    export SUBSPACE=0
    unset ARENA_SIZE
    unset USE_ARENA
    run $1
}
run_arena_l () {
    export SUBSPACE=10 ARENA_SIZE=50 USE_ARENA=1
    run $1
}

echo "{" > $file

#run_arena p2p-Gnutella24
#run_arena p2p-Gnutella30
#run_arena cit-HepTh
#run_arena loc-Brightkite
#run_arena p2p-Gnutella25
#run_arena cit-HepPh
#run_arena vsp_finan
run_arena fe-sphere
#run_arena usroad
#run_arena SF.cedge
#run_arena fe_body
#run_normal Gnutella31
#run_normal com-dblp

sed -i '$d' $file && echo "]}" >> $file
exit

run_arena  fe-sphere # 49152 
run_arena  CA-HepTH # 51971 
run_arena  ego-Facebook # 88234 
run_normal Gnutella31 # 147892 
run_arena  fe_body # 163734 
run_arena  loc-Brightkite # 214078 
run_arena  SF.cedge # 223001 
run_normal com-dblp # 1049866 
run_arena  usroad # 165435 
run_arena  vsp_finan # 552020 
run_arena "p2p-Gnutella31"

run_arena "p2p-Gnutella04"
run_arena "p2p-Gnutella05"
run_arena "p2p-Gnutella06"
run_arena "p2p-Gnutella08"
run_arena "p2p-Gnutella09"
run_arena "p2p-Gnutella24"
run_arena "p2p-Gnutella25"
run_arena "p2p-Gnutella30"

# bad "soc-Epinions1"
# bad "soc-LiveJournal1"
# bad "soc-Slashdot0811"

# bad "com-youtube.ungraph"
run_arena "com-amazon.ungraph"
run_arena_l "email-Eu-core"
# bad "wiki-topcats"

# bad "email-EuAll"
# bad "email-Enron"
# bad "wiki-Talk"

run_arena cit-HepPh
run_arena cit-HepTh

# bad "ca-AstroPh"
# bad "ca-CondMat"
run_arena "ca-GrQc"
run_arena_l "ca-HepPh"
run_arena "ca-HepTh"

# bad web-BerkStan
# bad web-Google
# bad web-NotreDame
# bad web-Stanford

# bad amazon0302
# bad amazon0312
# bad amazon0505
# bad amazon0601

# bad roadNet-CA
# bad roadNet-PA
# bad roadNet-TX

# bad  soc-sign-epinions
# bad wiki-Elec
# bad wiki-RfA
# bad soc-sign-Slashdot081106
# bad soc-sign-Slashdot090216
# bad soc-sign-Slashdot090221

# bad  loc-gowalla_edges

run_arena wiki-Vote
# bad wiki-Talk

run_arena soc-RedditHyperlinks
# bad sx-stackoverflow
# bad sx-mathoverflow
# bad sx-superuser
# bad sx-askubuntu
# bad wiki-talk-temporal
# bad email-Eu-core-temporal
# bad CollegeMsg
run_arena soc-sign-bitcoin-otc
run_arena soc-sign-bitcoin-alpha

# remove trailing comma
sed -i '$d' $file
echo "]" >> $file

echo "}" >> $file

< tc_times_lobster.txt ~/Builds/jq 'keys_unsorted[] as $key | $key + "," + (.[$key] | map(.lobster_time) | join(","))' | tr -d "\""
