python -m venv .venv
source .venv/bin/activate
pip install -U -r ../requirements.txt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib
python -m bitsandbytes
accelerate config
chmod +x script.sh
./script.sh