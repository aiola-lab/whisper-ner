import logging
import argparse
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisper_ner.utils import set_logger, get_device, audio_preprocess, prompt_preprocess


@torch.no_grad()
def main(model_path, audio_file_path, prompt, max_new_tokens, language, device):
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
    predicted_ids = model.generate(
        input_features,
        max_new_tokens=max_new_tokens,
        language=language,
        prompt_ids=prompt_ids,
        generation_config=model.generation_config,
    )

    # post-process token ids to text
    transcription = processor.batch_decode(
        predicted_ids[:, prompt_ids.shape[0]:], skip_special_tokens=True
    )[0]
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
    )
