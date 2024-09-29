# WhisperNER

Implementation for the peper [_WhisperNER: Unified Open Named Entity and Speech Recognition_](https://arxiv.org/abs/2409.08107).

[//]: # (add image from assets, make it smaller)
<p align="center">
<img src="assets/WhisperNER.png" alt="drawing" width="500"/>
</p>


## Links

- Paper: [WhisperNER: Unified Open Named Entity and Speech Recognition](https://arxiv.org/abs/2409.08107)
- Demo: TBD
- Models: 
  - [aiola/whisper-ner-v1](https://huggingface.co/aiola/whisper-ner-v1)
- Datasets:
  - TBD

## Installation
Start with creating a virtual environment and activating it:

```bash
conda create -n whisper-ner python=3.10 -y
conda activate whisper-ner
pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

Then install the package:
```bash
git clone https://github.com/aiola-lab/whisper-ner.git
cd whisper-ner
pip install -e .
```

--------
## Usage
Inference can be done using the following code:

```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisper_ner.utils import audio_preprocess, prompt_preprocess

model_path = "aiola/whisper-ner-v1"
audio_file_path = "path/to/audio/file"
prompt = "person, company, location"  # comma separated entity tags
    
# load model and processor from pre-trained
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# load audio file: user is responsible for loading the audio files themselves
input_features = audio_preprocess(audio_file_path, processor)
input_features = input_features.to(device)

prompt_ids = prompt_preprocess(prompt, processor)
prompt_ids = prompt_ids.to(device)

# generate token ids by running model forward sequentially
with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        prompt_ids=prompt_ids,
        generation_config=model.generation_config,
        language="en",
    )

# post-process token ids to text, remove prompt
transcription = processor.batch_decode(
    predicted_ids, skip_special_tokens=True
)[0]
print(transcription)
```