if [ $# -ne 2 ]; then
    echo "Usage: $0 <input file> <output dir>"
    exit 1
fi

infile=$1
outdir=$2

split_rel () {
    rel=$1
    cat $infile | grep "rel $rel" | grep -o "(.*" | tr -d "( )" > $outdir/$rel.csv
}

split_rel_prob () {
    rel=$1
    cat $infile | grep -E "rel .*$rel" >> $outdir/$outdir.scl
}

mkdir -p $outdir

split_rel calls
split_rel cfg_edge
split_rel class
split_rel exception_handler
split_rel in_catch
split_rel in_try
split_rel location
split_rel method
split_rel method_throws
split_rel returns
split_rel subclass
split_rel throw

echo "" > $outdir/$outdir.scl

echo "@file(\"./experiments/data/psa/$outdir/calls.csv\")\ntype calls(x: u32,x: u32,x: u32)" >> $outdir/$outdir.scl
echo "@file(\"./experiments/data/psa/$outdir/cfg_edge.csv\")\ntype cfg_edge(x: u32,x: u32,x: u32)" >> $outdir/$outdir.scl
echo "@file(\"./experiments/data/psa/$outdir/class.csv\")\ntype class(x: String)" >> $outdir/$outdir.scl
echo "@file(\"./experiments/data/psa/$outdir/exception_handler.csv\")\ntype exception_handler(x: u32,x: u32,x: String)" >> $outdir/$outdir.scl
echo "@file(\"./experiments/data/psa/$outdir/in_catch.csv\")\ntype in_catch(x: u32,x: u32)" >> $outdir/$outdir.scl
echo "@file(\"./experiments/data/psa/$outdir/in_try.csv\")\ntype in_try(x: u32,x: u32)" >> $outdir/$outdir.scl
echo "@file(\"./experiments/data/psa/$outdir/location.csv\")\ntype location(x: u32,x: String,x: String,x: u32)" >> $outdir/$outdir.scl
echo "@file(\"./experiments/data/psa/$outdir/method.csv\")\ntype method(x: u32,x: String)" >> $outdir/$outdir.scl
echo "@file(\"./experiments/data/psa/$outdir/method_throws.csv\")\ntype method_throws(x: u32,x: String)" >> $outdir/$outdir.scl
echo "@file(\"./experiments/data/psa/$outdir/returns.csv\")\ntype returns(x: u32,x: u32)" >> $outdir/$outdir.scl
echo "@file(\"./experiments/data/psa/$outdir/subclass.csv\")\ntype subclass(x: String,x: String)" >> $outdir/$outdir.scl
echo "@file(\"./experiments/data/psa/$outdir/throw.csv\")\ntype throw(x: u32,x: String)" >> $outdir/$outdir.scl
#echo "@file(\"./experiments/data/psa/$outdir/prob_cfg_edge.csv\", has_probability=true)\ntype prob_cfg_edge(x: u32,x: u32,x: u32)" >> $outdir/$outdir.scl

echo 'import "../rules.scl"' >> $outdir/$outdir.scl

split_rel_prob prob_cfg_edge
