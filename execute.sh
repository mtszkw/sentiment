echo "Creating new virtual environment..."
python3 -m venv ./env

echo "Activating the environment..."
. ./env/bin/activate

echo "Installing from requirements.txt file..."
python3 -m pip install -r pipeline/requirements.txt --quiet

echo "Running the training flow..."
python3 pipeline/training_flow.py run \
  --s3_input_csv_path=s3://sentimental-mlops-project/training.1600000.processed.noemoticon.csv \
  --quickrun_pct=0.01