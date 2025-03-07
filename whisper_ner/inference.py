import argparse
import logging

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_ner.utils import (
    audio_preprocess,
    get_device,
    prompt_preprocess,
    set_logger,
)
from whisper_ner.utils.entity_logits_processor import EntityBiasingLogitsProcessor


@torch.no_grad()
def main(model_path, audio_file_path, prompt, max_new_tokens, language, device, entity_bias=0.0):
    # load model and processor from pre-trained
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)

    model = model.to(device)

    # load audio file: user is responsible for loading the audio files themselves
    input_features = audio_preprocess(audio_file_path, processor)
    input_features = input_features.to(device)

    prompt_ids = prompt_preprocess(prompt, processor)
    prompt_ids = prompt_ids.to(device)

    # generate token ids by running model forward sequentially
    if model_path == "aiola/whisper-ner-v1" and language != "en":
        logging.info(
            f"Using language code: {language}. Please note that the model was trained on English only data."
        )


    # add entity biasing logits processor
    logits_processor = EntityBiasingLogitsProcessor(bias=entity_bias)

    predicted_ids = model.generate(
        input_features,
        max_new_tokens=max_new_tokens,
        language=language,
        prompt_ids=prompt_ids,
        generation_config=model.generation_config,
        logits_processor=[logits_processor],
    )

    # post-process token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    logging.info(f"Tagged Transcription: {transcription}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Whisper model."
    )
    parser.add_argument(
        "--model-path", type=str, default="aiola/whisper-ner-v1", help="Path to model"
    )
    parser.add_argument(
        "--audio-file-path",
        type=str,
        required=True,
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="person, company",
        help="Prompt with entity tags to detect, comma seperated.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for the transcription.",
    )
    parser.add_argument(
        "--entity-bias",
        type=float,
        default=0.0,
        help="Bias for the start of entity token (`<`).",
    )

    set_logger()

    args = parser.parse_args()
    device = get_device()

    main(
        args.model_path,
        args.audio_file_path,
        args.prompt,
        args.max_new_tokens,
        args.language,
        device,
        args.entity_bias,
    )
