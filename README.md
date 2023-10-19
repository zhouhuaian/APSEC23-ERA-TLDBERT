# TLDBERT

This repository stores the replication package of [APSEC 2023 ERA track](https://conf.researchr.org/track/apsec-2023/apsec-2023-early-research-achievements#submission-instructions) submission "TLDBERT: Leveraging Further Pre-trained Model
for Issue Typed Links Detection".

## Reproduction Steps

### Create Virtual Environment

```
conda env create -f environment.yml
conda activate tld
```

### Download Jira Issues Dataset

Download the [Jira issues dataset](https://zenodo.org/record/7182101) released by Montgomery et al. Follow the instructions detailed in the `README.md` of directory `3. DataDump/` to import the dataset into your MongoDB server.

### Preprocessing

Execute the Python scripts in directory `preprocess/` in order to clean the Jira issues dataset and construct the in-domain corpora.

### Further Pre-training

Run the command `bash pretrain/run_mlm_wwm.sh` to further pre-train [bert-base-uncased](https://huggingface.co/bert-base-uncased) on the in-domain corpora. The model checkpoints will be saved in `tmp/tldbert/`. Please modify the content in the script as needed.

### Typed Links Detection

1. First, execute the Jupyter script `tld/link_preprocess.ipynb` to obtain the links dataset;
2. Then, run the command `bash tld/model/run_finetune.sh` to fine-tune the models and execute the TLD task. The experimental results are saved in directory `tld/data/results/`;
3. Finally, execute the script `tld/correlation_analysis.ipynb` to collate the experimental results and conduct the correlation analysis.
