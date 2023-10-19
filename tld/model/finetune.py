import argparse
import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from datasets import ClassLabel, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, PreTrainedTokenizerBase,
                          Trainer, TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint, IntervalStrategy

from config import train_output_dir, train_results_dir
from dataset import get_link_dataset
from linktypes import Target


def main(
    model_name: str,
    tracker: str,
    target: Target,
    include_non_links: bool,
    train_batch_size: int,
    eval_batch_size: int,
    n_epochs: int,
):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)    

    input_ds = process_link_dataset(ds=get_link_dataset(tracker=tracker, target=target, include_non_links=include_non_links), tokenizer=tokenizer)
    label_feature: ClassLabel = input_ds['train'].features['label']
    print(f'Dataset contains {label_feature.num_classes} classes for {tracker}')
    print(label_feature.names)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=label_feature.num_classes)
    
    run_name = '_'.join([
        tracker,
        target,
        *(['plus'] if include_non_links else []),
        datetime.now().strftime('%Y%m%d_%H:%M'),
    ])
    run_training_output_dir = train_output_dir / model_name.split('/')[-1] / run_name

    training_args = TrainingArguments(
        output_dir=str(run_training_output_dir),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=n_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        save_strategy=IntervalStrategy.EPOCH,
        evaluation_strategy=IntervalStrategy.EPOCH,
        load_best_model_at_end=True,
        metric_for_best_model='fscore',
        group_by_length=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=input_ds['train'],
        eval_dataset=input_ds['val'],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=get_last_checkpoint(run_training_output_dir))

    # Save Results
    test_output_dir = train_results_dir / model_name.split('/')[-1] / run_name
    test_output_dir.mkdir(parents=True, exist_ok=True)

    with (test_output_dir / 'run_config.json').open('w', encoding='utf8') as f:
        json.dump(fp=f, obj={
            'run_name': run_name,
            'tracker': tracker,
            'target': target,
            'include_non_links': include_non_links,
            'model_name': model_name,
            'train_batch_size': train_batch_size,
            'eval_batch_size': eval_batch_size,
            'n_epochs': n_epochs,
            'score': trainer.state.best_metric,
            'label_names': label_feature.names,
        })
    
    test_output = predict_with_hidden_states(trainer, input_ds['test'])
    
    np.savez(
        test_output_dir / 'test_output.npz',
        labels=test_output.labels.cpu().numpy(),
        logits=test_output.logits.cpu().numpy(),
        last_hidden_states=test_output.last_hidden_states.cpu().numpy(),
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, fscore, support = precision_recall_fscore_support(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'fscore': fscore, 'support': support}


def prepare_sample(sample, max_issue_length: int):
    return {
        'issue_1': f'{sample["title_1"]}. {sample["description_1"]}'[:max_issue_length].strip(),
        'issue_2': f'{sample["title_2"]}. {sample["description_2"]}'[:max_issue_length].strip(),
    }


def process_link_dataset(ds: DatasetDict, tokenizer: PreTrainedTokenizerBase, max_seq_length: int = 192):
    max_text_len = get_upper_bound_text_chars(tokenizer=tokenizer, max_seq_length=max_seq_length)
    print(f'Trimming texts down to {max_text_len} characters')
    ds = ds.map(
        lambda sample: prepare_sample(sample, max_text_len),
        remove_columns=list(set(ds['train'].column_names) - {'label'}),
    )

    def tokenize(examples):
        return tokenizer(examples['issue_1'], examples['issue_2'], max_length=max_seq_length, truncation=True)

    return ds.map(tokenize, batched=True, batch_size=256, remove_columns=['issue_1', 'issue_2'])


def get_upper_bound_text_chars(tokenizer: PreTrainedTokenizerBase, max_seq_length: int) -> int:
    longest_token_char_length = max(len(tok.lstrip('#')) for tok in tokenizer.get_vocab().keys())
    return max_seq_length * longest_token_char_length + (max_seq_length - 1)


@dataclass
class ModelOutputs:
    labels: torch.tensor
    logits: torch.tensor
    last_hidden_states: torch.tensor


def predict_with_hidden_states(trainer, dataset) -> ModelOutputs:
    dataloader = trainer.get_test_dataloader(dataset)

    batches_labels = []
    batches_logits = []
    batches_last_hidden_states = []

    trainer.model.eval()
    for inputs in dataloader:
        with torch.no_grad():
            outputs = trainer.model(
                **{k: v.to(trainer.model.device) for k, v in inputs.items()},
                output_hidden_states=True,
            )
        
        batches_labels.append(inputs['labels'])
        batches_logits.append(outputs.logits.cpu())
        batches_last_hidden_states.append(outputs.hidden_states[-1][:, 0].cpu())

    return ModelOutputs(
        labels=torch.concat(batches_labels),
        logits=torch.concat(batches_logits),
        last_hidden_states=torch.concat(batches_last_hidden_states),
    )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True)
    parser.add_argument('--tracker', required=True)
    parser.add_argument('--target', default='linktype')
    parser.add_argument('--non-links', default=True)
    parser.add_argument('--train-batch-size', type=int, required=True)
    parser.add_argument('--eval-batch-size', type=int, required=True)
    parser.add_argument('--n-epochs', type=int, required=True)

    args = parser.parse_args()

    main(
        model_name=args.model,
        tracker=args.tracker,
        target=args.target,
        include_non_links=args.non_links,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        n_epochs=args.n_epochs,
    )