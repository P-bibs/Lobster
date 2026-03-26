#!/bin/bash

# Get the version and create the directory for the distribution
VERSION=`python scripts/get_current_version.py`
mkdir -p target/distribution/$VERSION/x86_64-unknown-linux-gnu

# Build scli x86_64 unknown linux-gnu
cargo build --release --bin scli
cp target/release/scli target/distribution/$VERSION/x86_64-unknown-linux-gnu/scli

# Build sclrepl x86_64 unknown linux-gnu
cargo build --release --bin sclrepl
cp target/release/sclrepl target/distribution/$VERSION/x86_64-unknown-linux-gnu/sclrepl

# Build scallopy x86_64 unknown linux-gnu, python 3.8
pushd etc/scallopy; conda activate scallop-dev-cp38; maturin build --release; conda deactivate; popd
find target/wheels -name "scallopy-$VERSION-*_x86_64.whl" -exec cp '{}' target/distribution/$VERSION/x86_64-unknown-linux-gnu \;

# Build scallopy x86_64 unknown linux-gnu, python 3.9
pushd etc/scallopy; conda activate scallop-dev-cp39; maturin build --release; conda deactivate; popd
find target/wheels -name "scallopy-$VERSION-*_x86_64.whl" -exec cp '{}' target/distribution/$VERSION/x86_64-unknown-linux-gnu \;

# Build scallopy x86_64 unknown linux-gnu, python 3.10
pushd etc/scallopy; conda activate scallop-dev-cp310; maturin build --release; conda deactivate; popd
find target/wheels -name "scallopy-$VERSION-*_x86_64.whl" -exec cp '{}' target/distribution/$VERSION/x86_64-unknown-linux-gnu \;
