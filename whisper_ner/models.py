from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.utils import logging, ModelOutput
from transformers import WhisperModel, WhisperConfig, WhisperForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperDecoder, WhisperDecoderLayer, shift_tokens_right
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

logger = logging.get_logger(__name__)


@dataclass
class WhisperNEROutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    ner_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class WhisperNERForConditionalGeneration(WhisperForConditionalGeneration):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = WhisperModel(config)
        self.freeze_name2func = {
            "encoder": self._freeze_encoder,
            "encoder_attn": self._freeze_encoder_and_cross_attn,
        }

    def _freeze_cross_attn(self):
        for layer in self.model.decoder.layers:
            for p in layer.encoder_attn.parameters():
                p.requires_grad = False

    def _freeze_decoder(self):
        decoder_layers = list(self.model.decoder.children())

        # Freeze all layers except the last one
        for layer in decoder_layers[:-1]:
            for p in layer.parameters():
                p.requires_grad = False

    def _freeze_encoder(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = False

    def _freeze_encoder_and_cross_attn(self):
        self._freeze_cross_attn()
        self._freeze_encoder()

    def freeze_model_parts(self, parts_to_freeze):
        if parts_to_freeze is None:
            return

        if parts_to_freeze not in self.freeze_name2func:
            raise ValueError(
                f"parts_to_freeze {parts_to_freeze} is not supported, "
                f"select from {list(self.freeze_name2func.keys())}"
            )

        self.freeze_name2func[parts_to_freeze]()

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            ner_labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            label_smoothing: float = 0.0,
            **kwargs
    ) -> Union[Tuple[torch.Tensor], WhisperNEROutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(label_smoothing=label_smoothing)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
        # if ner_labels is False then ner_loss is -1
        ner_loss = torch.tensor(-1)
        if ner_labels is not None:
            ner_loss_fct = CrossEntropyLoss(label_smoothing=label_smoothing)
            # move labels to correct device to enable PP
            ner_labels = ner_labels.to(lm_logits.device)
            ner_loss = ner_loss_fct(lm_logits.view(-1, self.config.vocab_size), ner_labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return WhisperNEROutput(
            loss=loss,
            ner_loss=ner_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


def get_model(args_i):
    if args_i.use_lora:
        model = WhisperNERForConditionalGeneration.from_pretrained(
            args_i.whisper_model_name,
            load_in_8bit=False,
            # device_map="auto",
        )
        config = LoraConfig(
            r=args_i.lora_rank,
            lora_alpha=args_i.lora_alpha,
            target_modules=args_i.lora_target_module,
            lora_dropout=args_i.lora_dropout,
            bias=args_i.lora_bias,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        model = WhisperNERForConditionalGeneration.from_pretrained(
            args_i.whisper_model_name
        )

    return model


if __name__ == '__main__':
    from whisper_ner.dataset import WhisperNERDataset, DataCollatorSeq2SeqWithPadding
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from whisper_ner.utils import remove_suppress_tokens

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="en", task="transcribe"
    )
    # data_path = "/home/ec2-user/workspace/nlp-artifacts/whisper_ner/pilener/pilener_train.json"
    data_path = "/Users/avivnavon/Desktop/aiola/nlp-artifacts/whisper_ner/nuner/training_files/test.json"
    data_collator = DataCollatorSeq2SeqWithPadding(processor=processor)
    dataset = WhisperNERDataset(data_path=data_path, processor=processor, entity_dropout_prob=0.5)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=data_collator
    )

    batch = next(iter(dataloader))
    model = WhisperNERForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    remove_suppress_tokens(model)
    out = model(**batch)
    print(out.loss)
