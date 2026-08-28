---
license: mit
base_model:
- Oxidane/tmr-ai-text-detector
tags:
- text-classification
- ai-detection
- robeta
- focal-loss
- raid-benchmark
language:
- en
pipeline_tag: text-classification
library_name: transformers.js
---



# tmr-ai-text-detector (ONNX)


This is an ONNX version of [Oxidane/tmr-ai-text-detector](https://huggingface.co/Oxidane/tmr-ai-text-detector). It was automatically converted and uploaded using [this Hugging Face Space](https://huggingface.co/spaces/onnx-community/convert-to-onnx).


## Usage with Transformers.js


See the pipeline documentation for `text-classification`: https://huggingface.co/docs/transformers.js/api/pipelines#module_pipelines.TextClassificationPipeline


---


# TMR: Target Mining RoBERTa - AI Text Detector

A robust AI-generated text detector based on RoBERTa-base, trained with Focal Loss and Self-Hard-Negative iterative mining on the RAID dataset.

## Model Description

TMR (Target Mining RoBERTa) is designed to detect AI-generated text with high accuracy while maintaining low false positive rates. The model uses:

- **Architecture**: RoBERTa-base (125M parameters)
- **Loss Function**: Focal Loss (gamma=2.0, alpha=[0.85, 0.15]) to focus on hard examples
- **Training Strategy**: Self-Hard-Negative (Self-HN) iterative mining
- **Training Data**: 50,000 stratified samples from RAID (45% human, 55% AI)

## Performance

### RAID Leaderboard (Official Results)

| Metric | All Settings | No Adversarial |
|--------|--------------|----------------|
| **AUROC** | **99.28%** | **99.85%** |
| **TPR @ 5% FPR** | **95.79%** | **99.65%** |
| **TPR @ 1% FPR** | **90.17%** | **98.56%** |

*Results from [RAID Benchmark](https://raid-bench.xyz/leaderboard) evaluation on 672,000 test samples (including adversarial attacks).*

### Held-out Evaluation (100k samples)

| Metric | Score |
|--------|-------|
| **AUROC** | 99.69% |
| **Accuracy** | 97.42% |
| **FPR** | 2.61% |
| **FNR** | 2.58% |

*Held-out evaluation on RAID train split (seed=999, excluded training/validation samples).*

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_path = "Oxidane/tmr-ai-text-detector"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Predict
text = "Your text here..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

# Probability that text is AI-generated
ai_probability = probs[0][1].item()
print(f"AI probability: {ai_probability:.4f}")

# Binary classification (threshold=0.5)
is_ai = ai_probability > 0.5
print(f"Prediction: {'AI-generated' if is_ai else 'Human-written'}")
```

## Training

Trained on the RAID dataset (ACL 2024) with Self-Hard-Negative mining: iteratively identifying human samples misclassified as AI, then retraining with these hard examples.

## Limitations

- **Language**: Primarily trained on English text
- **Domain**: Best performance on text similar to RAID training domains (news, books, abstracts, reviews, recipes, Wikipedia, poetry, Reddit)
- **Threshold**: Optimized for threshold=0.5
- **Out-of-distribution**: May have higher false positive rates on casual conversation, short text, or domains not seen during training

## License

MIT License

## Citation

If you use this model, please cite:

```bibtex
@misc{tmr-ai-text-detector,
  title={TMR: Target Mining RoBERTa for AI Text Detection},
  author={Oxidane},
  year={2025},
  url={https://huggingface.co/Oxidane/tmr-ai-text-detector}
}
```

## Contact

For questions, contact me@oxidane.net
