import logging
import torch
import torchaudio


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
