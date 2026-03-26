TIME=30

SIZES="5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

echo > times.txt

for i in $SIZES; do
    echo "RUNNING $i x $i"
    TIME=1 CUDA_VISIBLE_DEVICES=9 STRATUM=12,13 WANDB_MODE=disabled python experiments/pacman_maze/run_1d.py --grid-x $i --grid-y $i | python time.py $TIME > out.txt
    echo "$i x $i" >> times.txt
    tac out.txt | grep -m 1 -A 1 GPU >> times.txt
done

< times.txt python parse_times.py
