if [ $# -ne 2 ]; then
    echo "Usage: $0 <input file> <output file>"
    exit 1
fi

infile=$1
outfile=$2

cp rules.pl $outfile
cat $infile | sed 's/rel //' | sed 's/$/./' >> $outfile
echo "query(unhandled_exception(_,_))." >> $outfile
