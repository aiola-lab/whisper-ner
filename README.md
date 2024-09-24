# WhisperNER

Implementation for the peper [_WhisperNER: Unified Open Named Entity and Speech Recognition_](https://arxiv.org/abs/2409.08107).

[//]: # (add image from assets, make it smaller)
<p align="center">
<img src="assets/WhisperNER.png" alt="drawing" width="500"/>
</p>


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

```