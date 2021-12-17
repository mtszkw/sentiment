setup_environment:
	echo "Creating new virtual environment..."
	python3 -m venv sentiment-env

	echo "Activating the environment..."
	. sentiment-env/bin/activate

	echo "Installing from requirements.txt file..."
	python3 -m pip install -r pipeline/requirements.txt

training: setup_environment
	echo "Running the training flow..."
	python3 pipeline/training_flow.py run \
	  --s3_input_csv_path=s3://mtszkw-github/sentiment/training.1600000.processed.noemoticon.csv \
  	  --quickrun_pct=$(QUICKRUN_PCT) \
	  --rnd_seed=$(RND_SEED) \
	  --test_size=$(TEST_SIZE)