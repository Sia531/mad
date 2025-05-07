
# prepare for mujoco
cd ~
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir -p ~/.mujoco
tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
rm mujoco210-linux-x86_64.tar.gz
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin" >> ~/.bashrc
echo "export PATH=$PATH:~/.mujoco/mujoco210/bin" >> ~/.bashrc
source ~/.bashrc

# install apt packages
sudo apt update
sudo apt-get install python3-dev
sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3
sudo apt install patchelf

# venv setup
cd mad
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
uv venv -p 3.10
uv sync --extra cu126
uv pip uninstall typing
chmod +x ./.venv/bin/activate
source ./.venv/bin/activate

# data preparation
mkdir -p diffuser/datasets/data/mpe
cd diffuser/datasets/data/mpe
# wget "https://files.osf.io/v1/resources/jxawh/providers/osfstorage/64d11746c7ab293e72d4e0ee/\?view_only\=dd3264a695af4c03bffde0350b8e8c4a\&zip\=" -O mpe_data.zip
# unzip mpe_data.zip
# 7z -x simple_tag.zip
# rm mpe_data.zip
# cd ~/mad

# # run experiments
# python run_experiment.py -e exp_specs/mpe/tag/mad_mpe_tag_attn_exp_td3bc.yaml
