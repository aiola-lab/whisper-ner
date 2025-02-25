from transformers import LogitsProcessor
import torch

START_ENTITY_TOKEN_ID = 27


class EntityBiasingLogitsProcessor(LogitsProcessor):
    def __init__(self, bias=0.0):
        self.bias = bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Modify logits of the start of entity token (`<`)
        scores[:, START_ENTITY_TOKEN_ID] += self.bias
        return scores
