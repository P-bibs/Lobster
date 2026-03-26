#!/bin/sh

# Get the version and create the directory for the distribution
VERSION=`python scripts/get_current_version.py`
mkdir -p target/distribution/$VERSION/arm64-apple-darwin
mkdir -p target/distribution/$VERSION/x86_64-apple-darwin

# Build scli arm64 apple darwin
cargo build --release --bin scli
cp target/release/scli target/distribution/$VERSION/arm64-apple-darwin/scli

# Build scli x86_64 apple darwin
cargo build --release --bin scli --target x86_64-apple-darwin
cp target/x86_64-apple-darwin/release/scli target/distribution/$VERSION/x86_64-apple-darwin/scli

# Build sclrepl arm64 apple darwin
cargo build --release --bin sclrepl
cp target/release/sclrepl target/distribution/$VERSION/arm64-apple-darwin/sclrepl

# Build sclrepl x86_64 apple darwin
cargo build --release --bin sclrepl --target x86_64-apple-darwin
cp target/x86_64-apple-darwin/release/sclrepl target/distribution/$VERSION/x86_64-apple-darwin/sclrepl

# Build scallopy arm64 apple darwin
pushd etc/scallopy; maturin build --release; popd
find target/wheels -name "scallopy-$VERSION-*_arm64.whl" -exec cp '{}' target/distribution/$VERSION/arm64-apple-darwin \;

# Build sclrepl x86_64 apple darwin
pushd etc/scallopy; maturin build --release --target x86_64-apple-darwin; popd
find target/wheels -name "scallopy-$VERSION-*_x86_64.whl" -exec cp '{}' target/distribution/$VERSION/x86_64-apple-darwin \;
