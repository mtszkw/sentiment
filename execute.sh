python3 -m venv ./env

. ./env/bin/activate

python3 -m pip install -r model/requirements.txt --quiet

python3 model/training_flow.py run \
  --data_path=data/training.1600000.processed.noemoticon.csv \
  --quickrun_pct=0.001
