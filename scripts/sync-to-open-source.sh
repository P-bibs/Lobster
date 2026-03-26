#!/bin/sh

# Remove things
rm -r ../scallop-os/.github
rm -r ../scallop-os/core
rm -r ../scallop-os/doc
rm -r ../scallop-os/etc
rm -r ../scallop-os/lib
rm -r ../scallop-os/res
rm -r ../scallop-os/examples
rm -r ../scallop-os/experiments
rm -r ../scallop-os/scripts

# Full folders to include
cp -r .github/ ../scallop-os/.github
cp -r core/ ../scallop-os/core
cp -r doc/ ../scallop-os/doc
cp -r etc/ ../scallop-os/etc
cp -r lib/ ../scallop-os/lib
cp -r res/ ../scallop-os/res
cp -r examples/ ../scallop-os/examples

# Experiments
mkdir -p ../scallop-os/experiments
cp -r experiments/mnist/ ../scallop-os/experiments/mnist
cp -r experiments/hwf/ ../scallop-os/experiments/hwf
cp -r experiments/pacman_maze/ ../scallop-os/experiments/pacman_maze
cp -r experiments/clutrr-v2/ ../scallop-os/experiments/clutrr-v2
cp -r experiments/big-bench/ ../scallop-os/experiments/big-bench
cp -r experiments/gsm8k/ ../scallop-os/experiments/gsm8k

# Scripts
mkdir -p ../scallop-os/scripts
cp -r scripts/link_torch_lib.py ../scallop-os/scripts/link_torch_lib.py
cp -r scripts/get_current_version.py ../scallop-os/scripts/get_current_version.py

# Files in the root folder
cp .gitignore ../scallop-os/.gitignore
cp Cargo.toml ../scallop-os/Cargo.toml
cp LICENSE ../scallop-os/LICENSE
cp changelog.md ../scallop-os/changelog.md
cp makefile ../scallop-os/makefile
cp readme.md ../scallop-os/readme.md
cp rustfmt.toml ../scallop-os/rustfmt.toml
