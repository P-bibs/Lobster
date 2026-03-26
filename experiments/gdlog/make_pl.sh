convert () {
    # convert to .pl for problog
    cat experiments/data/gdlog/$1/edge.facts | tr -d '\015' | sed 's/\t/,/' | sed 's/^/edge(/' | sed 's/$/)./' > experiments/data/gdlog/tc/$1.pl
    echo "path(X, Y) :- edge(X, Y).\npath(X, Y) :- path(X, Z), edge(Z, Y).\nquery(path(_,_))." >> experiments/data/gdlog/tc/$1.pl
}

convert p2p-Gnutella25
convert p2p-Gnutella24
convert p2p-Gnutella30
convert fe-sphere
convert loc-Brightkite
convert SF.cedge
convert fe_body
convert cit-HepTh
convert cit-HepPh
convert Gnutella31
convert com-dblp
convert usroad
convert vsp_finan
