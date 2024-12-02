import random

import numpy as np
import logging
import torch
import torchaudio
import wandb
import argparse


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def get_device(gpu_id="0"):
    if torch.cuda.is_available():
        logging.info(f"GPU available. Using GPU {gpu_id}")
        return torch.device(f"cuda:{gpu_id}")
    # todo: add MPS support
    else:
        logging.info("Using CPU")
        return torch.device("cpu")


# ["<", " <", ">", " >", ">>", " >>"]
UNSUPPRESS_TOKEN = (27, 2627, 29, 12331, 893, 902)


def remove_suppress_tokens(model, unsuppress_token=UNSUPPRESS_TOKEN):
    model.config.suppress_tokens = [
        t for t in model.config.suppress_tokens if t not in unsuppress_token
    ]
    model.generation_config.suppress_tokens = [
        t for t in model.generation_config.suppress_tokens if t not in unsuppress_token
    ]


def audio_preprocess(audio_file_path, processor):
    target_sample_rate = 16000
    signal, sampling_rate = torchaudio.load(audio_file_path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sample_rate)
    signal = resampler(signal)
    # convert to mono or remove first dim if needed
    if signal.ndim == 2:
        signal = torch.mean(signal, dim=0)
    # pre-process to get the input features
    input_features = processor(
        signal, sampling_rate=target_sample_rate, return_tensors="pt"
    ).input_features
    return input_features


def _unify_ner_text(text, symbols_to_replace=("/", " ", ":", "_")):
    # remove multiple spaces
    text = " ".join(text.split())
    # Replace symbols with "-"
    for symbol in symbols_to_replace:
        text = text.replace(symbol, "-")
    # to lower
    text = text.lower()
    return text


def prompt_preprocess(prompt, processor):
    tags = prompt.split(",")
    prompt = ", ".join([_unify_ner_text(tag) for tag in tags])
    logging.info(f"Inference with prompt: '{prompt}'")
    prompt_ids = processor.get_prompt_ids(prompt, return_tensors="pt")
    return prompt_ids


def token_padding(features, processor, batch_key, pad_token_id=-100):
    # get the tokenized label sequences
    token_features = [{"input_ids": feature[batch_key]} for feature in features]
    # pad the labels to max length
    token_batch = processor.tokenizer.pad(token_features, return_tensors="pt")

    # replace padding with -100 to ignore loss correctly
    padded_tokens = token_batch["input_ids"].masked_fill(
        token_batch.attention_mask.ne(1), pad_token_id
    )

    # if bos token is appended in previous tokenization step,
    # cut bos token here as it's append later anyways
    if (padded_tokens[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
        padded_tokens = padded_tokens[:, 1:]

    return padded_tokens


def str_or_list(s):
    if s is None:
        return s
    else:
        new_s = [i.strip() for i in s.split(",")]
        if len(new_s) == 1:  # case it is a single string
            return new_s[0]
        else:
            return new_s


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser("Whisper finetune")

    parser.register("type", "custom_bool", str2bool)
    # data paths
    parser.add_argument(
        "--audio-root-dir",
        type=str,
        help="audio root directory",
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="training data path",
    )
    parser.add_argument(
        "--validation-data-path",
        type=str,
        required=True,
        help="validation data path",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        required=True,
        help="test data path",
    )

    # eval
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="number of every steps to save model checkpoint",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="number of every steps to evaluate the model",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--predict-with-generate",
        type="custom_bool",
        default="False",
        help="use generate for prediction",
    )
    parser.add_argument(
        "--compute-wer",
        type="custom_bool",
        default="False",
        help="compute WER or not",
    )
    # training args
    parser.add_argument(
        "--train", type="custom_bool", default=True, help="whether to train or not"
    )
    parser.add_argument("--ner-mask-pct", type=float, default=0.0, help="ner mask pct")
    parser.add_argument(
        "--entity-dropout-prob",
        type=float,
        default=0.1,
        help="probability to drop an entity label",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="gradient accumulation steps",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="number of update steps to train for",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="learning rate",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed to use")
    parser.add_argument(
        "--warmup-steps", type=int, default=0, help="warmup steps for scheduler"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./outputs/whisper_ft",
        help="where (path) to output the results",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="language",
    )
    parser.add_argument(
        "--fp16",
        type="custom_bool",
        default="True",
        help="use fp16 training",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adafactor",
        help="optimization strategy",
    )
    parser.add_argument(
        "--whisper-model-name",
        type=str,
        default="openai/whisper-large-v2",
        help="open ai's whisper model name",
    )
    parser.add_argument(
        "--parts-to-freeze",
        type=str,
        default="encoder",
        choices=["encoder", "encoder_attn", None],
        help="which model parts to freeze",
    )
    parser.add_argument(
        "--n-neg-samples",
        type=int,
        default=2,
        help="number of negative samples to consider",
    )
    # LoRA configuration
    parser.add_argument(
        "--use-lora",
        type="custom_bool",
        default=False,
        help="use training with LoRA",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=4,
        help="the rank of the update matrices, expressed in int. "
        "Lower rank results in smaller update matrices with fewer trainable parameters.",
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=64, help="LoRA scaling factor."
    )
    parser.add_argument(
        "--lora-target-module",
        type=str_or_list,
        default=".*decoder.*(self_attn|encoder_attn).*(q_proj|v_proj|k_proj)",
        help="LoRA scaling factor.",
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05, help="LoRA dropout factor."
    )
    parser.add_argument(
        "--lora-bias",
        type=str,
        default="none",
        help="LoRA Specifies if the bias parameters should be trained. "
        "Can be 'none', 'all' or 'lora_only.",
    )
    parser.add_argument(
        "--lora-merge-and-unload",
        type="custom_bool",
        default=False,
        help="This method merges the LoRa layers into the base model. "
        "This is needed if someone wants to use the base model as a standalone model.",
    )
    # wandb
    parser.add_argument(
        "--wandb-logging",
        type="custom_bool",
        default=False,
        help="If true, use wandb to report training metrics.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="wandb entity",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="",
        help="Experiment name",
    )

    args_ = parser.parse_args()

    if args_.wandb_logging:
        name = f"whisper_{args_.whisper_model_name}_{args_.exp_name}"
        wandb.init(name=name, project="whisper-ner", entity=args_.wandb_entity)
        wandb.config.update(args_)

    return args_
