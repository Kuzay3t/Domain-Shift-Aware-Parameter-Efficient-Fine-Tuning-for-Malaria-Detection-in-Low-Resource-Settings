# Domain-Shift-Aware-Parameter-Efficient-Fine-Tuning-for-Malaria-Detection-in-Low-Resource-Settings

# Domain-Shift-Aware Parameter-Efficient Fine-Tuning for Malaria Detection in Low-Resource Clinical Settings

> **DSA-LoRA** selects LoRA adapter layers using Maximum Mean Discrepancy (MMD) to concentrate fine-tuning capacity on transformer layers most affected by domain shift, enabling efficient adaptation of medical foundation models to tropical disease imaging with as few as 10 labeled LMIC examples.

---

## 1. What is the question you want to answer?

When a vision transformer pretrained on high-income country (HIC) medical imaging data is fine-tuned for malaria detection using a small number of labeled examples from a low-and-middle-income country (LMIC) clinical setting, does the placement of LoRA adapters across transformer layers affect adaptation performance — and can Maximum Mean Discrepancy (MMD) be used to identify which layers should be adapted?

Specifically: **standard LoRA applies adapters uniformly across all transformer layers regardless of where domain shift actually occurs. Is this uniform placement optimal under extreme domain shift and data scarcity, or does a principled, data-driven layer selection strategy produce better results with fewer adapter parameters?**

---

## 2. Why is this question important?

Malaria kills over 600,000 people annually, with the vast majority of deaths occurring in sub-Saharan Africa and other low-resource regions. Automated malaria detection from blood smear microscopy has the potential to reduce diagnostic delays in settings with few trained pathologists. However, AI models developed on clean, well-resourced HIC datasets fail to generalise to LMIC clinical conditions — where imaging equipment is older, image quality is lower, and labeled training data is extremely scarce.

Parameter-efficient fine-tuning methods like LoRA offer a path to adapting large pretrained models with minimal data and compute. But no prior work has examined whether the standard practice of uniform adapter placement is appropriate when the domain shift is large and labeled target-domain data is limited to tens of examples. This is precisely the situation faced by researchers and clinicians attempting to deploy AI in African and other LMIC healthcare settings.

This work directly addresses MIRASOL Workshop Focus Area 1: *Machine Learning for Medical Imaging in settings with data scarcity, imbalanced representations, and limited computational resources.*

---

## 3. What work does this question build on? Key papers to read.

**On PEFT and LoRA in medical imaging:**
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Xiao et al. (2024). *Less Could Be Better: Parameter-Efficient Fine-Tuning Advances Medical Vision Foundation Models.* arXiv:2401.12215 — **start here**
- Dutt et al. (2023). *Parameter-Efficient Fine-Tuning for Medical Image Analysis: The Missed Opportunity.* arXiv:2305.08252

**On domain shift in LMIC medical imaging:**
- Gao et al. (2020). *Deep learning for automated classification of tuberculosis-related chest X-Ray: dataset distribution shift limits diagnostic performance generalizability.* Scientific Reports
- Nakasi et al. (2024). *CodaMal: Contrastive Domain Adaptation for Malaria Detection in Low-Cost Microscopes.* arXiv:2402.10478

**On few-shot and foundation models for medical imaging:**
- Woerner et al. (2024). *Navigating Data Scarcity using Foundation Models: A Benchmark of Few-Shot and Zero-Shot Learning Approaches in Medical Imaging.* arXiv:2408.08058

**On selective fine-tuning and catastrophic forgetting:**
- Chen et al. (2025). *Fine Tuning without Catastrophic Forgetting via Selective Low Rank Adaptation.* arXiv:2501.15377
- Veiga et al. (2024). *Block Expanded DINORET: Adapting Natural Domain Foundation Models for Retinal Imaging Without Catastrophic Forgetting.* arXiv:2409.17332

**On MMD for domain adaptation:**
- Gretton et al. (2012). *A Kernel Two-Sample Test.* Journal of Machine Learning Research

---

## 4. Is this a publishable question?

Yes. The contribution is novel along three axes:

**Methodological novelty:** Prior PEFT work in medical imaging applies adapters uniformly. This is the first work to use layer-wise MMD analysis to guide selective LoRA placement specifically for LMIC medical imaging under extreme domain shift.

**Application novelty:** No prior work has studied PEFT behaviour under the specific domain shift imposed by smartphone-over-microscope LMIC malaria imaging versus clean HIC cell image datasets. The double shift — equipment-induced and population-induced — makes this a harder and more realistic test than prior benchmarks.

**Empirical contribution:** The ablation study demonstrates that principled MMD-based selection outperforms random layer selection and reversed heuristics, and that the MMD criterion correctly identifies non-monotonic shift patterns that position-based heuristics miss.

This work is submitted to the **MIRASOL Workshop at MICCAI 2026** (Medical Image Computing in Resource-Constrained Settings), Strasbourg, France.

---

## 5. What is the most simple experimental setting in which this hypothesis can be tested?

The hypothesis is tested in a binary malaria classification setting:

1. **Source model:** ViT-B/16 pretrained on ImageNet, fine-tuned fully on the NIH clean malaria cell image dataset. This simulates a model trained under HIC conditions.

2. **Domain shift measurement:** Forward-pass both datasets through the source model. Extract CLS token features at each of the 12 transformer blocks. Compute MMD between NIH and Lacuna feature distributions per layer to identify high-shift layers.

3. **Few-shot adaptation:** Apply LoRA to the source model using N labeled samples from the Lacuna LMIC dataset, where N ∈ {10, 20, 50, 100, 200}. Compare three strategies: full fine-tuning, standard LoRA (all layers), and DSA-LoRA (MMD-selected layers only).

4. **Evaluation:** AUC-ROC, F1, and accuracy on a fixed held-out Lacuna validation set of 200 samples.

This setting is simple enough to run entirely on free Kaggle GPU resources (T4), reproducible from a single repository, and directly motivated by the practical constraints of LMIC AI deployment.

---

## 6. Key baselines and benchmarks

| Baseline | Description | Why included |
|---|---|---|
| Full Fine-Tuning | All 85M parameters updated on N Lacuna samples | Upper bound that overfits under low N |
| Standard LoRA (all layers) | LoRA applied uniformly to all 12 transformer blocks | Primary PEFT baseline from the literature |
| Freeze-Early (last N layers) | Common heuristic — LoRA on last 6 layers only | Tests whether position-based selection is sufficient |
| Freeze-Late (first N layers) | Reverse heuristic — LoRA on first 6 layers only | Negative control |
| Random Layer Selection | Random selection of 6 layers (averaged over 3 runs) | Tests whether any selective strategy works or MMD is needed |

---

## 7. Datasets, models, evaluation, and resources

### Datasets

| Dataset | Role | Source | Size |
|---|---|---|---|
| NIH Malaria Cell Images | Source domain (HIC) | [Kaggle](https://www.kaggle.com/datasets/shahriar26s/malaria-detection) | 13,152 train / 1,253 val / 626 test |
| Lacuna Malaria Detection Challenge | Target domain (LMIC) | [Kaggle](https://www.kaggle.com/datasets/rajsahu2004/lacuna-malaria-detection-dataset) | 2,747 images (Uganda + Ghana) |

The Lacuna dataset was collected by Makerere AI Lab (Uganda) and minoHealth (Ghana) using smartphones mounted over microscope eyepieces. Images are full field-of-view (4160x3120 pixels), in contrast to the standardised 224x224 NIH cell crops.

### Model

- **Architecture:** ViT-B/16 via `timm` library
- **Pretraining:** ImageNet-21k
- **Parameters:** 85.8M total
- **LoRA rank:** 4, alpha: 8
- **DSA-LoRA trainable parameters:** 3,638,786 (4.24% of total)

Pretrained weights: https://www.kaggle.com/datasets/gwaceee/miccai-dataset

Stage 2 MMD results: https://www.kaggle.com/datasets/gwaceee/miccaidata2

### Evaluation metrics

- **AUC-ROC** — primary metric, robust to class imbalance
- **F1 Score** — balances precision and recall
- **Accuracy** — for comparison with prior work

### Compute

All experiments run on **Kaggle free tier (GPU T4 x2)**. Total GPU time: approximately 6 hours across all four stages.

---

## 8. Additional details and initial experiments

### MMD analysis finding (Stage 2)

Domain shift is not uniform across transformer layers. Early layers (0-3) show near-zero MMD. Shift concentrates in later layers — layer 9 shows MMD of approximately 0.19, roughly 100x higher than early layers. Layers 4 and 7 also show moderate shift, a non-monotonic pattern that position-based heuristics cannot capture.

### Main results (Stage 3)

| N samples | Full FT AUC | Standard LoRA AUC | DSA-LoRA AUC |
|---|---|---|---|
| 10 | 0.9905 | 0.9451 | **0.9515** |
| 20 | 0.9985 | 0.9989 | 0.9951 |
| 50 | 0.9942 | **0.9996** | 0.9956 |
| 100 | 0.9954 | **0.9964** | 0.9928 |
| 200 | 0.9985 | 0.9989 | **0.9995** |

DSA-LoRA outperforms Standard LoRA at n=10 using 50% fewer adapter parameters (3.6M vs 7.3M).

### Ablation results (Stage 4)

| Strategy | n=10 AUC | n=20 AUC |
|---|---|---|
| DSA-LoRA — MMD-Aware (Ours) | 0.9516 | 0.9949 |
| Freeze-Early (last 6 layers) | 0.9581 | 0.9963 |
| Freeze-Late (first 6 layers) | 0.8496 | 0.9916 |
| Random (avg of 3 runs) | 0.8973 | 0.9960 |

### Reproducing experiments

```bash
git clone https://github.com/Kuzay3t/dsa-lora-malaria
cd dsa-lora-malaria
pip install -r requirements.txt
python download_datasets.py
# Then run notebooks in order: stage1 → stage2 → stage3 → stage4
```

---

## Repository Structure

```
dsa-lora-malaria/
    config/                     hyperparameter configuration
    docs/                       paper PDF (after publication)
    experiments/                stage-level experiment configs
    notebooks/                  Kaggle notebooks for all 4 stages
    results/                    CSVs and figures from experiments
    src/
        dataset.py              NIHMalariaDataset, LacunaDataset
        model.py                ViT builder, LoRALinear, DSA-LoRA
        mmd.py                  MMD computation, layer selection
        train.py                training loop, evaluation utilities
    download_datasets.py        download all required datasets
    requirements.txt
    setup.py
```

---

## Citation

```bibtex
@inproceedings{gwacee2026dsalora,
  title     = {Domain-Shift-Aware Parameter-Efficient Fine-Tuning for
               Malaria Detection in Low-Resource Clinical Settings},
  author    = {Gwacee, Kuzayet},
  booktitle = {MIRASOL Workshop, MICCAI 2026},
  year      = {2026},
  address   = {Strasbourg, France}
}
```
