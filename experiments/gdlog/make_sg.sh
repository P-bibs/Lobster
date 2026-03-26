
make_sg () {
    body="@file(\"./experiments/data/gdlog/$1/edge.csv\", deliminator=\"\\\\t\")\ntype edge(x: u32, y: u32)\nimport \"sg.scl\""
    echo "$body" > experiments/gdlog/sg/$1.scl
    wc -l ./experiments/data/gdlog/$1/edge.csv
}

make_sg fe-sphere # 49152 
make_sg CA-HepTH # 51971 
make_sg ego-Facebook # 88234 
make_sg Gnutella31 # 147892 
make_sg fe_body # 163734 
make_sg loc-Brightkite # 214078 
make_sg SF.cedge # 223001 
make_sg com-dblp # 1049866 
make_sg usroad # 165435 
make_sg fc_ocean # 409593 
make_sg vsp_finan # 552020 
