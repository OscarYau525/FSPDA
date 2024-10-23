Install gcc-10 (https://askubuntu.com/questions/1192955/how-to-install-g-10-on-ubuntu-18-04)
    <!-- ln -s /usr/bin/gcc-10 /usr/bin/gcc -->
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt install gcc-10
    sudo apt install g++-10

    #Remove the previous alternatives
    sudo update-alternatives --remove-all gcc
    sudo update-alternatives --remove-all g++

    #Define the compiler
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 30
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 30

    sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
    sudo update-alternatives --set cc /usr/bin/gcc

    sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
    sudo update-alternatives --set c++ /usr/bin/g++

    #Confirm and update (You can use the default setting)
    sudo update-alternatives --config gcc
    sudo update-alternatives --config g++

Build MPI with cuda support https://github.com/Stonesjtu/pytorch-learning/blob/master/build-with-mpi.md: 
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
    ./configure --prefix=$HOME/opt/openmpi --with-cuda --enable-mpi-thread-multiple 
    make all
    make install
    export LD_LIBRARY_PATH="$HOME/opt/openmpi/bin:$LD_LIBRARY_PATH"
    export PATH="$HOME/opt/openmpi/bin:$PATH"
    sudo cp $HOME/opt/openmpi/lib/libmpi* /usr/lib

Build torch from source of https://github.com/pytorch/pytorch:
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    git clone --recursive https://github.com/pytorch/pytorch
    git submodule update --init --recursive
    python setup.py build develop

Build vision from source of https://github.com/pytorch/vision:
    git clone https://github.com/pytorch/vision.git
    python setup.py develop
    
