if [ ! -d "experiments" ]; then
    echo "Directory experiments/ does not exist. Run from root of scallop-v2"
    exit 1
fi

if [ ! -d "experiments/data/gdlog" ]; then
    mkdir -p experiments/data
    cd experiments/data
    wget --output-document gdlog.zip "https://drive.usercontent.google.com/download?id=1ImJiPwZF2ZNx1O96DnCY7CMiPTLEce-l&export=download&authuser=1&confirm=t&uuid=1cffcc51-beb5-4b78-83a1-926caa18be72&at=AEz70l6VjDZCyXvmVSMXvqSHMnAL:1741378216510"
    unzip gdlog.zip
    mv data gdlog
    rm gdlog.zip
    cd ../..
fi

cd experiments/data/gdlog

cd cspa/linux
cp assign.facts assign.csv
cp dereference.facts dereference.csv
cd -

cd cspa/httpd
cp assign.facts assign.csv
cp dereference.facts dereference.csv
cd -

cd cspa/postgresql
cp assign.facts assign.csv
cp dereference.facts dereference.csv
cd -

tc_files="vsp_finan usroad SF.cedge loc-Brightkite Gnutella31 fe_body fe-sphere fc_ocean ego-Facebook com-dblp CA-HepTH"
for p in $tc_files; do
    # convert to csv for scallop
    cp $p/edge.facts $p/edge.csv
    cd ../../gdlog/tc
    echo "@file(\"./experiments/data/gdlog/$p/edge.csv\", deliminator=\"\\\t\")\ntype edge(x: u32, y: u32)\n\nimport \"tc.scl\"" > $p.scl
    cd -

    mkdir -p tc
    # convert to .pl for problog
    cat $p/edge.facts | sed 's/\t/,/' | sed 's/^/edge(/' | sed 's/$/)./' > tc/$p.pl
    echo "path(X, Y) :- edge(X, Y).\npath(X, Y) :- path(X, Z), edge(Z, Y).\nquery(path(_,_))." >> tc/$p.pl
done

# get_mtx () {
#     NAME=$1
#     wget https://suitesparse-collection-website.herokuapp.com/MM/SNAP/$NAME.tar.gz
#     tar -xzf $NAME.tar.gz
#     rm $NAME.tar.gz
#     mv $NAME/$NAME.mtx .
#     rm -rf $NAME
#     # remove % comments, remove first line, replace spaces with commas
#     < $NAME.mtx grep -v "%" | tail -n +2 | sed 's/ /,/g' > $NAME.csv
# }
# 
# get_mtx "p2p-Gnutella04"
# get_mtx "p2p-Gnutella05"
# get_mtx "p2p-Gnutella06"
# get_mtx "p2p-Gnutella08"
# get_mtx "p2p-Gnutella09"
# get_mtx "p2p-Gnutella24"
# get_mtx "p2p-Gnutella25"
# get_mtx "p2p-Gnutella30"
# get_mtx "p2p-Gnutella31"
# 
# wget https://snap.stanford.edu/data/email-Eu-core.txt.gz
# gunzip email-Eu-core.txt.gz
# < email-Eu-core.txt sed 's/ /,/g' > email-Eu-core.csv

