file=tc_times_gdlog.txt
scratch=tc_times_gdlog_scratch.txt

run () {
    echo "Running GDLog $1"
    date
    echo $1 >> $file
    start=`date +%s`
    ./build/TC ./data/$1/edges.facts 0 > $scratch 2>&1
    end=`date +%s`
    runtime=$((end-start))
    echo $runtime >> $file
}

echo "" > $file
run fe-sphere # 49152 
run CA-HepTH # 51971 
run ego-Facebook # 88234 
run Gnutella31 # 147892 
run fe_body # 163734 
run usroad # 165435 
run loc-Brightkite # 214078 
run SF.cedge # 223001 
run fc_ocean # 409593 
run vsp_finan # 552020 
run com-dblp # 1049866 
