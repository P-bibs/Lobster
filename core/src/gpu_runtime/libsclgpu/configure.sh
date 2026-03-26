CMAKE="cmake"
CLANG="/home/paulbib/Builds/clang/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04/bin/clang++"

mkdir -p cmake-debug
mkdir -p cmake-release

mkdir -p cmake-debug-rmm
mkdir -p cmake-release-rmm

#${CMAKE} -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_COMPILER=${CLANG} -DCMAKE_CUDA_HOST_COMPILER=${CLANG} -DCMAKE_CXX_COMPILER=${CLANG} -DCMAKE_CUDA_EXTRA_FLAGS="--cuda-noopt-device-debug" ..

cd cmake-debug
echo "Configuring Debug build"
${CMAKE} -DCMAKE_BUILD_TYPE=Debug ..
cd ..

cd cmake-release
echo "Configuring Release build"
${CMAKE} -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cd ..
