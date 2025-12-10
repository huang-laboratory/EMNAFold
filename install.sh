#!/bin/bash

function check_last() {
  local status=$1
  local msg=$2

  if [ ! $status -eq 0 ]; then
    if [ -n "$msg" ]; then
      echo "$msg"
    fi
    exit 1
  fi
}

# 0 check existence
conda --help >/dev/null
check_last $? "Please install conda"

wget --help >/dev/null
check_last $? "Please install wget"

tar --hel >/dev/null
check_last $? "Please install tar"

echo "1 Create conda env"
conda env create -f env.yml
check_last $? "Failed at step 1, unable to create conda env"
# If conda fails, you can install the packages youself.
# Basically, you can first create an environment, then install the packages listed in env.yml using conda or pip.

echo "2 Install packages"
conda activate emnafold
check_last $? "Failed at step 2"

echo "2.1 Install flash attention"
cd flash_attn
sh install_flash_attn.sh
cd ..
check_last $? "Failed at step 2.1, unable to install flash attention"

echo "2.2 Install RiNALMo"
cd em3na/rinalmo/RiNALMo-1.0
pip install -e .
cd ..
check_last $? "Failed at step 2.2, unable to install RiNALMo"

echo "2.3 Download RiNALMo trained weights"
wget https://zenodo.org/records/15043668/files/rinalmo_giga_pretrained.pt \
  -O weights/rinalmo_giga_pretrained.pt
check_last $? "Failed at step 2.3, unable to download RiNALMo weights"
cd ../..

echo "3 Download pretrained weights"
cd em3na
wget http://huanglab.phys.hust.edu.cn/EMNAfold/weights/weights.tgz -O weights.tgz
check_last $? "Failed at step 3, unable to download pretrained weights"
cd ..

echo "4 install main program"
chmod +x em3na/bin/*
pip install -e .
# Please DO include "-e" in the pip command
check_last $? "Failed at step 4, unable to install main package"

echo "Done install"
