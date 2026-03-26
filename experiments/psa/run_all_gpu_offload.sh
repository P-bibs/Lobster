file=gpu_stratum_set_sweep.txt
scratch=psa_set_sweep_out.txt

sets="14,14 14,15 13,14 13,15 12,14 12,15"

echo "" > $file
for stratum_set in $sets; do
    echo "Running $stratum_set"
    date
    echo $stratum_set >> $file
    NO_CHECK=1 TIME=1 STRATUM=$stratum_set CUDA_VISIBLE_DEVICES=0 ./target/release/scli ./experiments/data/psa/sunflow-core-facts/sunflow-core-facts.scl --provenance minmaxprob > $scratch 2>&1
    cat $scratch | grep "Total sample time:" | sed "s/Total sample time: //" | tr -d "s" >> $file
done

