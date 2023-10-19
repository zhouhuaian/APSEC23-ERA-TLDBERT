import logging
import argparse
import os
import pandas as pd
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='To concatenate csv files')
    parser.add_argument('--input', type=str, default='../data/processed/cleaned_issues', help='input dir which stores cleaned issues csv')
    parser.add_argument('--output', type=str, default='../data/processed', help='output dir which stores concatenated csv file')
    args = parser.parse_args()

    logger.info("Start to concatenate csv files...")

    # Get input file paths
    input_dir = args.input
    output_path = Path(args.output)
    file_paths = []
    for root, dirs, file_names in os.walk(input_dir):
        for file_name in file_names:
            file_paths.append(os.path.join(root, file_name))

    start_time = time.perf_counter()

    df_ls = []
    for file_path in file_paths:
        logger.info(file_path.split('/')[-1])

        file_df = pd.read_csv(file_path, sep=';', encoding='utf-8', low_memory=False, usecols=['title', 'description', 'comments'])
        file_df.fillna(' ', inplace=True)
        logger.info(f'nan value exists: {file_df.isnull().any().any()}')  # check if nan value exists
        
        texts = file_df['title'] + '. ' + file_df['description'] + '. ' + file_df['comments']
        file_df = file_df.assign(text=texts)
        
        df_ls.append(file_df['text'])

    concated_df = pd.concat(df_ls, ignore_index=True)

    concated_df.to_csv(output_path / 'Jira_issues.csv', columns=['text'], encoding='utf-8', index=False)

    end_time = time.perf_counter()

    logger.info(f"Time cost to concatenate {concated_df.shape[0]} issues: {end_time - start_time} s")