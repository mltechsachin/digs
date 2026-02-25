sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd src
python3 run_experiments.py --epochs 2 --train_samples 192 --val_samples 48

