import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import torch
import torchaudio
from transformers import WhisperProcessor

from whisper_ner.utils import token_padding

SAMPLE_RATE = 16_000


class WhisperNERDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        processor: WhisperProcessor,
        max_seq_length: int = 448,
        target_sample_rate=16_000,
        audio_root_dir=None,
        entity_dropout_prob=0.1,
        max_samples=None,
        n_neg_samples=2,
        ner_mask_pct=0.0,
    ):
        # read json file + create dataset
        self.data_path = data_path
        with open(data_path, "r") as f:
            dataset = json.load(f)

        if max_samples is not None and len(dataset) > max_samples:
            logging.info(f"Using only {max_samples} samples from the dataset.")
            dataset = random.sample(dataset, max_samples)

        self.dataset = dataset
        all_ner_tags = [
            example[2]
            for d in dataset
            for example in [self._unify_ner_tag(t) for t in d["ner"]]
        ]
        self.all_ner_tags_unique = list(set(all_ner_tags))

        self.processor = processor
        self.max_seq_length = max_seq_length

        self.entity_dropout_prob = entity_dropout_prob

        self.target_sample_rate = target_sample_rate
        self.audio_root_dir = (
            Path(audio_root_dir) if audio_root_dir is not None else None
        )
        self.ner_mask_pct = ner_mask_pct

        self.n_neg_samples = n_neg_samples

    def __len__(self):
        return len(self.dataset)

    def _extract_masked_ner_labels(self, labels):
        # extract the masked NER labels
        masked_ner_labels = [-100] * len(
            labels
        )  # -100 is the ignore index for the cross-entropy loss
        open_tokens = self.processor.tokenizer(
            "< <", add_special_tokens=False
        ).input_ids
        close_tokens = self.processor.tokenizer(
            ">> >>", add_special_tokens=False
        ).input_ids
        is_open = False
        for i, t in enumerate(labels):
            if t in open_tokens:
                is_open = True
            elif t in close_tokens:
                masked_ner_labels[i] = t
                is_open = False
            if is_open:
                masked_ner_labels[i] = t
        return masked_ner_labels

    def get_audio_features(self, audio_path, resampling_to):
        # todo: consider using to soundfile to prevent issues with mp3 files with 'ffmpeg>=5'.
        #  If you encounter issues with mp3 files use `conda install 'ffmpeg<5'`
        speech_array, sampling_rate = torchaudio.load(audio_path)
        resampler = torchaudio.transforms.Resample(sampling_rate, resampling_to)
        array = resampler(speech_array)[0].numpy()
        return self.processor.feature_extractor(
            array, sampling_rate=resampling_to
        ).input_features[0]

    def _sample_negative_ner_tags(self, idx):
        query_record = self.dataset[idx]
        query_ner_tags = query_record["ner"]
        query_ner_tags = [self._unify_ner_tag(tag) for tag in query_ner_tags]
        query_ner_types = list(set([example[2] for example in query_ner_tags]))
        n_pos = len(query_ner_types)
        neg_ner_types = random.choices(
            self.all_ner_tags_unique, k=int(n_pos * self.n_neg_samples)
        )
        neg_ner_types = list(set(neg_ner_types))

        return neg_ner_types

    @staticmethod
    def _unify_ner_text(text, symbols_to_replace=("/", " ", ":", "_")):
        # remove multiple spaces
        text = " ".join(text.split())
        # Replace symbols with "-"
        for symbol in symbols_to_replace:
            text = text.replace(symbol, "-")
        # to lower
        text = text.lower()
        return text

    def _unify_ner_tag(self, ner_tag):
        ner_tag[2] = self._unify_ner_text(ner_tag[2])
        return ner_tag

    def _construct_labels(self, text, ner_tags, ner_labels_to_keep, mask_flag):
        labels = ""
        curr_idx = 0
        last_end_idx = 0
        last_tag = ""

        for start_idx, end_idx, tag, tag_text, tag_explanation in ner_tags:
            tag = tag.lower()
            if tag not in ner_labels_to_keep:
                continue

            if start_idx < last_end_idx and tag == last_tag:
                # There is an overlap with the same tag type
                if end_idx <= last_end_idx:
                    continue  # Skip this tag entirely as it's fully covered
                else:
                    # Only append and tag the part not covered
                    labels += text[last_end_idx:end_idx]
                    last_end_idx = end_idx
            else:
                # No overlap or different ner_type
                labels += text[curr_idx:start_idx]
                if mask_flag:
                    labels += f"<{tag}>>"
                else:
                    labels += f"<{tag}>{tag_text}<{tag}>>"
                curr_idx = end_idx  # Move the current index forward
                last_end_idx = end_idx
                last_tag = tag

        # Append any remaining text after the last processed entity
        labels += text[curr_idx:]

        return labels

    def __getitem__(self, item):
        record = self.dataset[item]
        text = record["text"]
        # record contains 'text' and 'ner'
        # inside each `ner` key we have: start_idx, end_idx, tag, tag_text, tag_explanation
        ner_tags = record["ner"]
        # Replace "/" and " " with "-" in the third element of each sublist using replace
        ner_tags = [self._unify_ner_tag(tag) for tag in ner_tags]

        # Sort ner_tags by start_idx to handle them in order
        ner_tags = sorted(ner_tags, key=lambda x: (x[0], x[1]))
        unique_ner_labels = list(set([tag for _, _, tag, _, _ in ner_tags]))
        # Drop some entities with probability entity_dropout_prob
        ner_labels_to_keep = [
            label
            for label in unique_ner_labels
            if random.random() > self.entity_dropout_prob
        ]

        mask_flag = random.random() <= self.ner_mask_pct

        labels_text = self._construct_labels(
            text, ner_tags, ner_labels_to_keep, mask_flag
        )

        # positive NER types
        pos_ner_types = ner_labels_to_keep

        # get random element from the dataset for negative NERs
        neg_ner_types = self._sample_negative_ner_tags(item)

        # combine positive and negative NER types
        ner_types = list(set(pos_ner_types + neg_ner_types))

        # randomize the order of the NER types
        random.shuffle(ner_types)
        prompt = ", ".join(ner_types)
        if mask_flag:
            prompt = f"<|mask|>" + prompt

        prompt_ids = self.processor.tokenizer.get_prompt_ids(prompt).tolist()
        max_len = self.max_seq_length - len(prompt_ids)

        assert max_len > 0, "Prompt is too long"

        orig_decoder_input_ids = self.processor.tokenizer(
            labels_text, max_length=max_len
        ).input_ids

        decoder_input_ids = prompt_ids + orig_decoder_input_ids
        n_mask_tokens = len(prompt_ids)

        example = dict()
        example["original_text"] = self.processor.tokenizer(
            text, max_length=max_len
        ).input_ids
        example["decoder_input_ids"] = decoder_input_ids
        example["labels"] = [-100] * n_mask_tokens + orig_decoder_input_ids
        example["ner_labels"] = self._extract_masked_ner_labels(example["labels"])
        example["prompt"] = prompt
        example["ner_types"] = self.processor.tokenizer(prompt).input_ids

        # try and get audio
        audio_path = record.get("audio", None)
        if audio_path:
            example["input_features"] = self.get_audio_features(
                (
                    audio_path
                    if self.audio_root_dir is None
                    else self.audio_root_dir / audio_path
                ),
                resampling_to=self.target_sample_rate,
            )

        return example


@dataclass
class DataCollatorSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        batch = dict()
        # process audio features if they are available
        if features[0].get("input_features", None) is not None:
            input_features = [
                {"input_features": feature["input_features"]} for feature in features
            ]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )

        # all labels
        batch["labels"] = token_padding(features, self.processor, batch_key="labels")
        batch["labels"] = batch["labels"][:, 1:]

        # original text
        batch["original_text"] = token_padding(
            features, self.processor, batch_key="original_text"
        )

        if features[0].get("ner_labels", None) is not None:
            # ner labels
            batch["ner_labels"] = token_padding(
                features, self.processor, batch_key="ner_labels"
            )
            batch["ner_labels"] = batch["ner_labels"][:, 1:]
            batch["ner_types"] = token_padding(
                features, self.processor, batch_key="ner_types"
            )

        if features[0].get("decoder_input_ids", None) is not None:
            batch["decoder_input_ids"] = token_padding(
                features,
                self.processor,
                batch_key="decoder_input_ids",
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
            batch["decoder_input_ids"] = batch["decoder_input_ids"][:, :-1]

        return batch


def get_dataset(args_i, processor):
    dataset = dict()
    dataset["train"] = WhisperNERDataset(
        data_path=args_i.train_data_path,
        processor=processor,
        audio_root_dir=args_i.audio_root_dir,
        entity_dropout_prob=args_i.entity_dropout_prob,
        n_neg_samples=args_i.n_neg_samples,
        ner_mask_pct=args_i.ner_mask_pct,
    )

    dataset["test"] = WhisperNERDataset(
        data_path=args_i.test_data_path,
        processor=processor,
        audio_root_dir=args_i.audio_root_dir,
        entity_dropout_prob=0.0,
        max_samples=args_i.max_eval_samples,
        ner_mask_pct=args_i.ner_mask_pct,
    )

    dataset["validation"] = WhisperNERDataset(
        data_path=args_i.validation_data_path,
        processor=processor,
        audio_root_dir=args_i.audio_root_dir,
        entity_dropout_prob=0.0,
        max_samples=args_i.max_eval_samples,
        ner_mask_pct=args_i.ner_mask_pct,
    )

    return dataset
