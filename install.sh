#!/bin/bash

# path to conda
conda_dir=$1

if [[ $# -lt 1 ]]; then
    echo "usage: bash install.sh /path/to/your/conda"
    exit 1
fi

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
if [[ ! -e ${conda_dir}/bin/conda ]]; then
    echo "Conda is not in this dir -> ${conda_dir}"
    exit 1
fi
wget --help >/dev/null
check_last $? "Failed to detect 'wget', maybe 'wget' is not installed?"
tar --help >/dev/null
check_last $? "Failed to detect 'tar', maybe 'tar' is not installed?"

source ${conda_dir}/bin/activate

echo "1 Create conda env"
conda env create -f env.yml
check_last $? "Failed at step 1, unable to create conda env"
## If conda fails, you can install the packages youself. Basically, you can first create an environment, then install the packages listed in env.yml using conda or pip.

echo "2 Install packages"
conda activate emnafold
check_last $? "Failed at step 2"

echo "2.1 Install flash attention"
cd flash_attn_whl && bash install_flash_attn.sh && cd ..
check_last $? "Failed at step 2.1, unable to install flash attention"

echo "2.2 Install RiNALMo"
cd em3na/rinalmo/RiNALMo-1.0 && pip install -e . && cd ../../..
check_last $? "Failed at step 2.2, unable to install RiNALMo"

echo "3.1 Download pretrained weights"
cd em3na && wget http://huanglab.phys.hust.edu.cn/EMNAfold/weights/weights.tgz -O weights.tgz && tar -zxf weights.tgz && cd ..
check_last $? "Failed at step 3.1, unable to download pretrained weights"

echo "3.2 Download RiNALMo trained weights"
cd em3na/rinalmo && mkdir -p weights && wget https://zenodo.org/records/15043668/files/rinalmo_giga_pretrained.pt -O weights/rinalmo_giga_pretrained.pt && cd ../..
check_last $? "Failed at step 3.2, unable to download RiNALMo weights"

echo "4 install main program"
### Excutables
chmod +x em3na/bin/* && ${conda_dir}/envs/emnafold/bin/pip install -e .
### Please DO include "-e" in the pip command
check_last $? "Failed at step 4, unable to install main package"

echo "Done install"
echo "Ready to run"
echo "Use"
echo "> ${conda_dir}/envs/emnafold/bin/emnafold --help"
echo "Or"
echo "> source ${conda_dir}/bin/activate emnafold"
echo "> emnafold --help"

