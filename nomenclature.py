import models
MODELS = {
    'llm': models.LLM
}

import datasetss
DATASETS = {
    'hf_dataset': datasetss.HuggingfaceDataset,
}

import trainers
TRAINERS = {
    'lm_trainer': trainers.LMTrainer
}

import evaluators
EVALUATORS = {
}
