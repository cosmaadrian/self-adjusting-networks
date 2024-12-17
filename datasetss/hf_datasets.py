import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lib.accelerator import AcumenAccelerator
from lib.dataset_extra import AcumenDataset

from transformers import AutoTokenizer

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import datasets as hfds


class HuggingfaceDataset(AcumenDataset):
    def __init__(self, args, kind = 'train'):
        super().__init__(args)
        self.args = args
        self.kind = kind

        self.chunk_size = args.dataset_args.chunk_size

        subset = None if not len(args.dataset_args.subset.replace("'", "").replace('"', "").strip()) else args.dataset_args.subset
        print("[HuggingfaceDataset] Loading dataset", args.dataset_args.dataset_name, "subset", subset, "split", 'train' if kind == 'train' else 'test')

        if 'bigcode/the-stack' in args.dataset_args.dataset_name:
            # load up a particular programming language from the stack. It needs data_dir, not subset
            self.dataset = hfds.load_dataset(args.dataset_args.dataset_name, data_dir = subset, split = 'train', token = os.environ.get('HF_TOKEN', None))
        else:
            self.dataset = hfds.load_dataset(args.dataset_args.dataset_name, subset, split = 'train', token = os.environ.get('HF_TOKEN', None))

        self.dataset = self.dataset.train_test_split(test_size = 0.01, seed = 42)['train' if kind == 'train' else 'test']

        if kind == 'test':
            num_batches_of_interest = min(int(256 * self.args.eval_batch_size), len(self.dataset))
            self.dataset = self.dataset.select(range(num_batches_of_interest))

        self.dataset = self.dataset.shuffle()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_args.hf_model_name, token = os.environ.get('HF_TOKEN', None))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        key = 'content' if 'bigcode/the-stack' in self.args.dataset_args.dataset_name else 'text'

        text = self.dataset[idx][key]
        encoded = text.encode('utf-8')

        if self.kind == 'train':
            # randomly truncate text if it's larger than the chunk size
            if len(encoded) > self.chunk_size + 3:
                start = np.random.randint(0, len(encoded) - self.chunk_size)
                encoded = encoded[start:start + self.chunk_size]
        else:
            # truncate text to the chunk size
            encoded = encoded[:self.chunk_size]

        text = encoded.decode('utf-8', errors = 'ignore')

        return {'text': text}

    @classmethod
    def collate_fn(cls, args, tokenizer):
        def _collate_fn(batch):
            texts = [x['text'] for x in batch]
            tokenized = tokenizer(texts, padding = 'max_length', truncation = True, max_length = args.dataset_args.chunk_size, return_tensors = 'np')
            return {
                'text': texts,
                'input_ids': torch.tensor(tokenized['input_ids'], dtype = torch.long),
                'attention_mask': torch.tensor(tokenized['attention_mask'], dtype = torch.long),
            }

        return _collate_fn

    @classmethod
    def tokenizer_dataloader(cls, args, kind = 'train'):
        dataset = cls(args = args, kind = kind)
        sampler = DistributedSampler(dataset, shuffle = kind == 'train', drop_last = False) if AcumenAccelerator().is_distributed else None
        dataset.sampler = sampler

        return DataLoader(
            dataset,
            num_workers = args.environment.extra_args.num_workers,
            batch_size = args.batch_size,
            pin_memory = True,
            sampler = sampler,
            shuffle = (sampler is None and kind == 'train'),
            prefetch_factor = 2 if args.environment.extra_args.num_workers > 0 else None,
        )

    @classmethod
    def train_dataloader(cls, args):
        dataset = cls(args = args, kind = 'train')
        sampler = DistributedSampler(dataset, shuffle = True, drop_last = False) if AcumenAccelerator().is_distributed else None
        dataset.sampler = sampler

        return DataLoader(
            dataset,
            num_workers = args.environment.extra_args.num_workers,
            batch_size = args.batch_size,
            pin_memory = True,
            sampler = sampler,
            shuffle = (sampler is None),
            collate_fn = cls.collate_fn(args, dataset.tokenizer),
            prefetch_factor = 2,
        )

    @classmethod
    def val_dataloader(cls, args):
        dataset = cls(args = args, kind = 'test')
        sampler = DistributedSampler(dataset, shuffle = False, drop_last = False) if AcumenAccelerator().is_distributed else None
        dataset.sampler = sampler

        return DataLoader(
            dataset,
            num_workers = args.environment.extra_args.num_workers,
            batch_size = args.eval_batch_size,
            pin_memory = True,
            shuffle = False,
            sampler = sampler,
            collate_fn = cls.collate_fn(args, dataset.tokenizer),
            prefetch_factor = 2,
        )