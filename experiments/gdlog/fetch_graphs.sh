if [ ! -d "experiments" ]; then
    echo "Directory experiments/ does not exist. Run from root of scallop-v2"
    exit 1
fi

if [ ! -d "experiments/data/gdlog" ]; then
    echo "Directory experiments/data/gdlog/ does not exist. Run fetch_data.sh first"
    exit 1
fi

cd experiments/data/gdlog

get () {
    NAME=`basename $1`
    echo "Fetching $NAME.txt"
    #if [ ! -f "$NAME.txt" ]; then
        curl -s -O https://snap.stanford.edu/data/$1.txt.gz
        if grep -q "not found" "$NAME.txt.gz"; then
            echo "Failed to download $NAME.txt.gz"
            rm $NAME.txt.gz
            return 1
        fi
        gunzip -f $NAME.txt.gz
    #else
    #    echo "$NAME.txt already exists"
    #fi
    # remove # comments, remove first line, replace spaces with commas
    < $NAME.txt grep -v "#" | sed 's/\t/,/g' > $NAME.csv
    < $NAME.csv sed 's/,/\t/g' > $NAME.txt

    mkdir -p $NAME
    cp $NAME.txt $NAME/edge.facts

    echo -e "@file(\"./experiments/data/gdlog/$NAME.csv\", deliminator=\",\")\ntype edge(x: u32, y: u32)\nimport \"tc.scl\"" > ../../gdlog/tc/$NAME.scl

}

get p2p-Gnutella25
get p2p-Gnutella24
get p2p-Gnutella30
#get fe-sphere
#get loc-Brightkite
#get SF.cedge
#get fe_body
get cit-HepTh
get cit-HepPh
get Gnutella31
get com-dblp
get usroad
get vsp_finan

#get "p2p-Gnutella04"
#get "p2p-Gnutella05"
#get "p2p-Gnutella06"
#get "p2p-Gnutella08"
#get "p2p-Gnutella09"
#get "p2p-Gnutella24"
#get "p2p-Gnutella25"
#get "p2p-Gnutella30"
#get "p2p-Gnutella31"

#get "ego-Facebook"
#get "ego-Gplus"
#get "ego-Twitter"
#get "soc-Epinions1"
#get "soc-LiveJournal1"
#get "soc-Pokec"
#get "soc-Slashdot0811"
#get "soc-Slashdot0922"

# bad "wiki-Vote"
# bad "wiki-RfA"
# bad "gemsec-Deezer"
# bad "gemsec-Facebook"
# bad "soc-RedditHyperlinks"
# bad "soc-sign-bitcoin-otc"
# bad "soc-sign-bitcoin-alpha"
# bad "comm-f2f-Resistance"
# bad "musae-twitch"
# bad "musae-facebook"
# bad "act-mooc"
# bad "musae-github"
# bad "feather-deezer-social"
# bad "feather-lastfm-social"
# bad "twitch-gamers"
# bad "congress-Twitter"

#get "bigdata/communities/com-youtube.ungraph"
#get "bigdata/communities/com-amazon.ungraph"
#get "email-Eu-core"
#get "wiki-topcats"

#get "email-EuAll"
#get "email-Enron"
#get "wiki-Talk"

#get cit-HepPh
#get cit-HepTh

# get "ca-AstroPh"
# get "ca-CondMat"
# get "ca-GrQc"
# get "ca-HepPh"
# get "ca-HepTh"

# get web-BerkStan
# get web-Google
# get web-NotreDame
# get web-Stanford

# get amazon0302
# get amazon0312
# get amazon0505
# get amazon0601
# 
# get roadNet-CA
# get roadNet-PA
# get roadNet-TX
# 
# get soc-sign-epinions
# get soc-sign-Slashdot081106
# get soc-sign-Slashdot090216
# get soc-sign-Slashdot090221
# 
# get loc-gowalla_edges
# 
# get wiki-Vote
# get wiki-Talk
# 
# get soc-RedditHyperlinks
# get sx-stackoverflow
# get sx-mathoverflow
# get sx-superuser
# get sx-askubuntu
# get wiki-talk-temporal
# get email-Eu-core-temporal
# get CollegeMsg
# get soc-sign-bitcoin-otc
# get soc-sign-bitcoin-alpha

