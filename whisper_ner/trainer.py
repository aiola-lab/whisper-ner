import json
import logging
from functools import partial
from pathlib import Path

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperProcessor

from whisper_ner.dataset import DataCollatorSeq2SeqWithPadding, get_dataset
from whisper_ner.models import get_model
from whisper_ner.utils import (parse_args, remove_suppress_tokens, set_logger,
                               set_seed)
from whisper_ner.utils.metrics import compute_metrics


def get_training_args(arguments):
    training_args = Seq2SeqTrainingArguments(
        output_dir=arguments.output_path,  # change to a repo name of your choice
        per_device_train_batch_size=arguments.batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        learning_rate=arguments.lr,
        warmup_steps=arguments.warmup_steps,
        max_steps=arguments.max_steps,
        gradient_checkpointing=False,
        fp16=arguments.fp16,
        evaluation_strategy="steps",
        save_total_limit=2,
        per_device_eval_batch_size=arguments.batch_size,
        predict_with_generate=arguments.predict_with_generate,
        generation_max_length=225,
        save_steps=arguments.save_steps,
        eval_steps=arguments.eval_steps,
        logging_steps=1,
        report_to=["wandb"] if arguments.wandb_logging else ["none"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_validation_loss",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim=arguments.optim,
        label_names=["labels"],
    )
    return training_args


def main(args_i, training_args):
    set_seed(args_i.seed)

    model = get_model(args_i)
    # remove suppress_tokens
    remove_suppress_tokens(model)

    # datasets
    processor = WhisperProcessor.from_pretrained(
        args_i.whisper_model_name, language=args_i.language, task="transcribe"
    )
    dataset = get_dataset(args_i, processor)
    data_collator = DataCollatorSeq2SeqWithPadding(processor)

    compute_metrics_arg = (
        partial(compute_metrics, tokenizer=processor.tokenizer)
        if args_i.compute_wer
        else None
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset={"validation": dataset["validation"]},
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics_arg,
    )

    if args_i.train:
        if not args_i.use_lora:
            model.freeze_model_parts(args_i.parts_to_freeze)
        trainer.train()

    if args_i.use_lora and args_i.lora_merge_and_unload:
        model = model.merge_and_unload()

    model_comp_path_obj = Path(args_i.output_path) / "model_components"
    model_comp_path_obj.mkdir(parents=True, exist_ok=True)
    model_comp_path_str = model_comp_path_obj.as_posix()

    model.save_pretrained(model_comp_path_str)
    processor.tokenizer.save_pretrained(model_comp_path_str)
    processor.save_pretrained(model_comp_path_str)

    results = trainer.evaluate(eval_dataset=dataset["test"])

    message = f"test loss: {results['eval_loss']}"
    if args_i.compute_wer:
        message += f", test WER: {results['eval_wer']}"
    logging.info(message)


if __name__ == "__main__":
    set_logger()
    args = parse_args()
    print(json.dumps(args.__dict__, indent=2))
    train_args = get_training_args(args)
    assert train_args.greater_is_better == (
        "loss" not in train_args.metric_for_best_model
    ), "training_args.greater_is_better should be set to True if your measuring metric is 'loss'"
    main(args, train_args)
