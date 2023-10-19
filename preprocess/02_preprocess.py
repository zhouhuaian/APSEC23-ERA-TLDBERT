import pandas as pd
import time
import logging
import argparse
import os
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_jira_formats(raw_text):
    md_re = re.compile(r"h\d\. ?|#{2,6} |\*{1,3} ")
    jira_re = re.compile(r"\{code.*?\}|\{noformat.*?\}|\{color.*?\}")
    
    raw_text = re.sub(md_re, " ", raw_text)
    cleaned_text = re.sub(jira_re, " ", raw_text)
    
    return cleaned_text


def preprocess(file_path: str, out_dir: Path):
    
    issue_ct = 0
    file_pd = pd.read_csv(file_path, sep=';', encoding='utf-8', low_memory=False)
    file_pd.fillna(' ', inplace=True)
    filename = file_path.split('/')[-1]
    logging.info(filename)

    for idx, row in file_pd.iterrows():
        try:
            row['title'] = ' '.join(clean_jira_formats(row['title']).split())
            row['description'] = ' '.join(clean_jira_formats(row['description']).split())
            row['comments'] = ' '.join(clean_jira_formats(row['comments']).split())
            issue_ct += 1
        
        except Exception as e:
            pass

    file_pd.fillna(' ', inplace=True)
    logging.info(f'Totally cleaned {issue_ct} issues for {filename}')

    file_pd.to_csv(out_dir / filename, sep=';', encoding='utf-8', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To preprocess the dataset')
    parser.add_argument('--input', type=str, default='../data/raw/issues', help='input dir which stores splited raw issues csv')
    parser.add_argument('--output', type=str, default='../data/processed', help='output dir which stores preprocessed datasets')
    args = parser.parse_args()
    
    logging.info("Start to preprocess datasets...")
    
    # Get input file paths
    input_dir = args.input
    output_dir = Path(args.output + '/cleaned_issues')
    output_dir.mkdir(parents=True, exist_ok=True)

    file_paths = []
    for root, dirs, file_names in os.walk(input_dir):
        for file_name in file_names:
            file_paths.append(os.path.join(root, file_name))
    
    start_time = time.perf_counter()

    for file_path in file_paths:
        preprocess(file_path, output_dir)

    end_time = time.perf_counter()
    
    logging.info(f"Time cost: {end_time - start_time} s")