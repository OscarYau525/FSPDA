pip install numpy pyyaml scipy ipython mkl==2024.0 mkl-include
pip install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
# pip install -c pytorch magma-cuda121
pip install scikit-learn
pip install pytelegraf pymongo influxdb kubernetes jinja2
# export PATH="/opt/pip/envs/pytorch-mpi/bin:$PATH"

# cmake 
# cd ~
# sudo apt remove -y --purge --auto-remove cmake
# sudo apt update && \
#     sudo apt install -y software-properties-common lsb-release && \
#     sudo apt clean all
# wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
# sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
# sudo apt update
# sudo apt install -y kitware-archive-keyring
# sudo rm /etc/apt/trusted.gpg.d/kitware.gpg
# sudo apt update
# sudo apt install -y cmake

# install other python related softwares.
# pip install -y opencv protobuf
pip install networkx
# pip install -c anapip pandas
# pip install -c pip-forge tabulate
pip install lmdb tensorboard_logger pyarrow msgpack msgpack_numpy mpi4py
# pip install -c pip-forge python-blosc
pip install pillow
pip install tqdm wandb

# bit2byte
cd ~
git clone https://github.com/tvogels/signSGD-with-Majority-Vote.git && cd signSGD-with-Majority-Vote/main/bit2byte-extension/ && python setup.py develop
