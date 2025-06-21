# Speech Recognition with Dual-Head Transformer

A PyTorch implementation of a speech recognition system using a dual-head transformer architecture with both CTC (Connectionist Temporal Classification) and attention-based decoder heads, trained on the LJSpeech dataset.

## 🎯 Features

- **Dual-Head Architecture**: Combines CTC and attention-based decoding for robust speech recognition
- **Transformer-Based**: Uses transformer encoder-decoder architecture for better sequence modeling
- **End-to-End Training**: Direct audio-to-text mapping without intermediate phoneme representations
- **LJSpeech Integration**: Automatic dataset download, preprocessing, and training pipeline
- **Mel Spectrogram Features**: Audio preprocessing with configurable mel-frequency parameters
- **Real-time Inference**: Fast inference pipeline for transcribing new audio files

## 🏗️ Architecture

The model consists of:

1. **Audio Feature Extractor**: Converts raw audio to mel spectrograms
2. **Transformer Encoder**: Processes acoustic features with self-attention
3. **CTC Head**: Provides alignment-free sequence prediction
4. **Decoder Head**: Attention-based autoregressive text generation
5. **Combined Loss**: Weighted combination of CTC and cross-entropy losses

```
Audio → Mel Spectrogram → Transformer Encoder → [CTC Head, Decoder Head] → Text
```

## 📋 Requirements

```
torch>=1.9.0
torchaudio>=0.9.0
numpy>=1.19.0
matplotlib>=3.3.0
pandas>=1.3.0
tqdm>=4.62.0
urllib3>=1.26.0
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/speech-recognition-transformer.git
cd speech-recognition-transformer
pip install -r requirements.txt
```

### Basic Usage

```python
from speech_recognition import main_with_training

# Train the model (downloads LJSpeech automatically)
model, history = main_with_training()
```

### Inference on New Audio

```python
from speech_recognition import infer, LJSpeechProcessor, AudioFeatureExtractor, DualHeadTransformer

# Load trained model
config = Config()
model = DualHeadTransformer(config)
model.load_state_dict(torch.load('checkpoints/best_model.pt')['model_state_dict'])

# Initialize components
processor = LJSpeechProcessor(config)
feature_extractor = AudioFeatureExtractor(config)

# Transcribe audio
text = infer(model, feature_extractor, processor, 'path/to/audio.wav', config, device)
print(f"Transcription: {text}")
```

## ⚙️ Configuration

Key hyperparameters can be adjusted in the `Config` class:

```python
class Config:
    # Audio preprocessing
    SAMPLE_RATE = 22050
    N_FFT = 1024
    HOP_LENGTH = 256
    N_MELS = 80
    
    # Model architecture
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    
    # Training parameters
    batch_size = 16
    learning_rate = 0.0001
    max_epochs = 100
```

## 📊 Dataset

The model is trained on the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/):
- **Size**: ~24 hours of speech
- **Speaker**: Single female speaker
- **Quality**: High-quality recordings
- **Text**: Normalized transcriptions included

The dataset is automatically downloaded and processed when you run the training script.

## 🎓 Training

### Full Training Pipeline

```python
# Run complete training with validation
python speech_recognition.py
```

### Training Features

- **Automatic Data Loading**: Downloads and preprocesses LJSpeech
- **Train/Validation Split**: 90/10 split for model evaluation
- **Checkpointing**: Saves best model based on validation loss
- **Learning Rate Scheduling**: Reduces LR on plateau
- **Gradient Clipping**: Prevents gradient explosion
- **Progress Monitoring**: Real-time loss tracking and sample inference

### Training Output

```
Epoch 1/100
Train Loss: 2.1234 | CTC: 1.8901 | CE: 2.3567
Val Loss: 2.0123 | CTC: 1.7890 | CE: 2.2345
Sample inference:
Ground truth: hello world this is a test
Prediction: helo wrld this is a tst
```

## 📈 Model Performance

The dual-head approach provides several advantages:

- **CTC Head**: Handles variable-length sequences without explicit alignment
- **Decoder Head**: Leverages language modeling for better accuracy
- **Combined Training**: Balances alignment-free and attention-based learning

## 🔧 Advanced Usage

### Custom Dataset

```python
# Adapt for your own dataset
class CustomDataset(Dataset):
    def __init__(self, audio_paths, transcriptions, config, processor, feature_extractor):
        # Your implementation here
        pass
```

### Beam Search Decoding

```python
# Implement beam search for better results
def beam_search_decode(logits, beam_size=5):
    # Your beam search implementation
    pass
```

### Fine-tuning

```python
# Load pre-trained model and fine-tune
model = DualHeadTransformer(config)
checkpoint = torch.load('pretrained_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Continue training with lower learning rate
config.learning_rate = 0.00001
```

## 🐛 Common Issues & Solutions

### CUDA Out of Memory
```python
# Reduce batch size
config.batch_size = 8

# Use gradient accumulation
accumulation_steps = 2
```

### Device Mismatch Errors
```python
# Ensure all tensors are on the same device
features = features.to(device)
feature_lengths = torch.tensor([features.size(1)], device=device)
```

### Audio Loading Issues
```python
# Verify audio file format and sample rate
import torchaudio
waveform, sr = torchaudio.load(audio_path)
print(f"Sample rate: {sr}, Duration: {waveform.size(1)/sr:.2f}s")
```

## 📊 Results

Training on LJSpeech dataset (first 1000 samples):

| Metric | Value |
|--------|-------|
| Train Loss | 1.23 |
| Val Loss | 1.45 |
| CTC Loss | 0.89 |
| CE Loss | 1.67 |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 TODO

- [ ] Implement beam search decoding
- [ ] Add support for streaming inference
- [ ] Language model integration
- [ ] Multi-speaker support
- [ ] Data augmentation techniques
- [ ] ONNX export for deployment
- [ ] Docker containerization
- [ ] Web API interface

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/) by Keith Ito
- [PyTorch](https://pytorch.org/) team for the deep learning framework
- [Transformer architecture](https://arxiv.org/abs/1706.03762) by Vaswani et al.
- [CTC Loss](https://www.cs.toronto.edu/~graves/icml_2006.pdf) by Graves et al.

## 📚 References

1. Vaswani, A., et al. "Attention is all you need." NIPS 2017.
2. Graves, A., et al. "Connectionist temporal classification." ICML 2006.
3. Chorowski, J., et al. "Attention-based models for speech recognition." NIPS 2015.

## 📞 Contact

Your Name - [@yourusername](https://twitter.com/yourusername) - your.email@example.com

Project Link: [https://github.com/yourusername/speech-recognition-transformer](https://github.com/yourusername/speech-recognition-transformer)

---

⭐ Star this repository if you find it helpful!
