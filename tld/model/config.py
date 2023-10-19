from pathlib import Path

data_dir = Path(__file__).parent.parent / 'data'
joined_data_dir = data_dir / 'joined'
train_output_dir = data_dir / 'outputs'
train_results_dir = data_dir / 'results'

for directory in [joined_data_dir, train_output_dir, train_results_dir]:
    directory.mkdir(parents=True, exist_ok=True)
