# Evaluate Lobster on the "same generation" task on each graph

dir="sg_results"
mkdir $dir

run_sg () {
    echo "RUNNING $1"
    echo "" > "$dir/$1.txt"
    echo "" > "$dir/$1.err"
    for i in `seq 0 0`; do
        USE_FROG=1 CUDA_VISIBLE_DEVICES=0 TIME=1 STRATUM=1,1 timeout "240s" ./target/release/scli experiments/gdlog/sg/$1.scl >> "$dir/$1.txt" 2>> "$dir/$1.err"
    done
}

run_sg fe-sphere # 49152 
run_sg CA-HepTH # 51971 
run_sg ego-Facebook # 88234 
run_sg fe_body # 163734 
run_sg loc-Brightkite # 214078 
run_sg SF.cedge # 223001 
run_sg fc_ocean # 409593 

#run_sg Gnutella31 # 147892 OOM
#run_sg com-dblp # 1049866 
#run_sg usroad # 165435  OOM
#run_sg vsp_finan # 552020 OOM
