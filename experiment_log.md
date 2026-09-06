# DeepGlobe UNet Segmentation — Experiment Log

## Dataset
- **Dataset**: DeepGlobe Land Cover Classification
- **Total pairs**: 803 (train)
- **Split**: 80/20 → 642 train / 161 val
- **Image size**: 2448x2448 → resized to 256x256
- **Classes**: 7 (urban, agriculture, rangeland, forest, water, barren, unknown)
- **Batch size**: 8

---

## Experiment 1: Basic UNet (From Scratch)
**Date**: Sep 6, 2026
**Model**: Custom UNet (DoubleConv + Encoder + Decoder)
**Encoder**: 3→64→128→256→512→1024 (random init)
**Loss**: CrossEntropyLoss
**Optimizer**: Adam, lr=1e-4
**Epochs**: 30

| Class      | IoU     | Dice    |
|------------|---------|---------|
| urban      | 0.3679  | 0.4572  |
| agriculture| 0.6625  | 0.7296  |
| rangeland  | 0.0270  | 0.0461  |
| forest     | 0.1505  | 0.1732  |
| water      | 0.0727  | 0.0878  |
| barren     | 0.0783  | 0.1025  |
| unknown    | 0.0000  | 0.0000  |
| **Mean**   | **0.1941** | **0.2281** |

**Observation**: Agriculture dominates, rare classes barely learned. Class imbalance issue.

---

## Experiment 2: Basic UNet + Weighted Loss + Augmentation
**Date**: Sep 6, 2026
**Model**: Same custom UNet
**Encoder**: 3→64→128→256→512→1024 (random init)
**Loss**: CrossEntropyLoss (weighted by inverse class frequency)
**Optimizer**: Adam, lr=1e-4
**Epochs**: 70
**Augmentation**: Random horizontal + vertical flips

| Class      | IoU     | Dice    |
|------------|---------|---------|
| urban      | 0.3895  | 0.4794  |
| agriculture| 0.5318  | 0.6237  |
| rangeland  | 0.1473  | 0.2179  |
| forest     | 0.1239  | 0.1487  |
| water      | 0.1518  | 0.1987  | 
| barren     | 0.1480  | 0.1917  |
| unknown    | 0.0063  | 0.0084  |
| **Mean**   | **0.2141** | **0.2669** |

**Observation**: Weighted loss helped rare classes slightly (rangeland 0.03→0.15, barren 0.08→0.15). Agriculture dropped (0.66→0.53) due to balance. Overall +0.02 improvement.

---

## Experiment 3: ResNet34 Encoder + 5 Epoch Test
**Date**: Sep 6, 2026
**Model**: Custom UNet with pretrained ResNet34 encoder
**Encoder**: ResNet34 (ImageNet pretrained)
**Loss**: CrossEntropyLoss (weighted)
**Optimizer**: Adam, lr=1e-4 (same for all)
**Epochs**: 5

| Class      | IoU     | Dice    |
|------------|---------|---------|
| urban      | 0.3235  | 0.4180  |
| agriculture| 0.2572  | 0.3394  |
| rangeland  | 0.0596  | 0.0989  |
| forest     | 0.1158  | 0.1335  |
| water      | 0.1210  | 0.1567  |
| barren     | 0.1256  | 0.1648  |
| unknown    | 0.0087  | 0.0120  |
| **Mean**   | **0.1445** | **0.1890** |

**Observation**: WORSE than baseline! Pretrained features destroyed by high LR (1e-4) on encoder. Need differential learning rates.

---

## Experiment 4: ResNet34 + Differential LR
**Date**: Sep 6, 2026
**Model**: Custom UNet with pretrained ResNet34 encoder
**Encoder**: ResNet34 (ImageNet pretrained)
**Loss**: CrossEntropyLoss (weighted)
**Optimizer**: Adam — encoder lr=1e-6, decoder lr=1e-4
**Epochs**: 5

| Class      | IoU     | Dice    |
|------------|---------|---------|
| urban      | 0.2348  | 0.3127  |
| agriculture| 0.1622  | 0.2555  |
| rangeland  | 0.0069  | 0.0134  |
| forest     | 0.1555  | 0.1813  |
| water      | 0.0753  | 0.1082  |
| barren     | 0.1161  | 0.1533  |
| unknown    | 0.0006  | 0.0011  |
| **Mean**   | **0.1073** | **0.1465** |

**Observation**: Even worse! LR 1e-6 on encoder too conservative — pretrained features couldn't adapt to satellite domain. Custom decoder also had architectural issues (skip connection bugs, wrong upsampling count).

---

## Experiment 5: smp UNet + DiceLoss + 6 Classes (PENDING)
**Date**: Sep 7, 2026
**Model**: smp.Unet with pretrained ResNet34 encoder (battle-tested decoder)
**Encoder**: ResNet34 (ImageNet pretrained)
**Loss**: smp.losses.DiceLoss(mode='multiclass') — handles class imbalance automatically
**Optimizer**: Adam, lr=1e-4
**Epochs**: 50
**Key changes**:
- Dropped "unknown" class (ignore class per original DeepGlobe challenge) → 6 classes
- Used smp library instead of custom decoder (eliminates architecture bugs)
- DiceLoss instead of weighted CrossEntropy
**Status**: NOT YET TRAINED

**Expected**: Mean IoU 0.45-0.55+

---

## Experiment 6: smp UNet + DiceLoss + 6 Classes + Augmentation + Scheduler
**Date**: Sep 7, 2026
**Model**: smp.Unet with pretrained ResNet34 encoder
**Loss**: smp.losses.DiceLoss(mode='multiclass')
**Optimizer**: Adam, lr=1e-4
**Scheduler**: CosineAnnealingLR (decays LR from 1e-4 → 0 over 200 epochs)
**Epochs**: 200
**Augmentation**: Horizontal flip + Vertical flip + Random rotation (90/180/270)
**Key changes from Exp 5**:
- Added rotation augmentation (3x more diverse training data)
- Added cosine LR scheduler (helps model fine-tune in later epochs)
- 200 epochs instead of 50 (more training time)
**Status**: COMPLETED

| Class      | IoU     | Dice    |
|------------|---------|---------|
| urban      | 0.3475  | 0.4198  |
| agriculture| 0.6822  | 0.7457  |
| rangeland  | 0.1844  | 0.2545  |
| forest     | 0.1727  | 0.1953  |
| water      | 0.1626  | 0.1983  |
| barren     | 0.1854  | 0.2282  |
| **Mean**   | **0.2891** | **0.3403** |
| **Pixel Accuracy** | **87.24%** | |

**Observation**: +38% improvement from Exp 5 (0.21 → 0.29). Augmentation + scheduler + 200 epochs with early stopping (stopped at 167) worked well. Agriculture at 0.68 is strong. Rare classes improved but still weak. Pixel accuracy is 87.24% but misleading — agriculture dominates the dataset, so even a naive "predict everything as agriculture" would get ~60% pixel accuracy.

**Expected**: Mean IoU 0.30-0.45+

**Monitoring**: Each epoch prints train/val loss, current LR, per-epoch time, and total elapsed time. Early stopping triggers if no improvement for 20 consecutive epochs.

---

## Key Learnings
1. Class imbalance needs weighted loss or Dice loss
2. Pretrained encoder needs careful LR tuning — too high destroys features, too low prevents adaptation
3. 803 images is small — augmentation helps
4. Unknown class has almost no data — it's an ignore class per original challenge, should be excluded from metrics
5. Agriculture is easiest class (most data, clear features)
6. Custom decoder introduced bugs (skip connections, upsampling count) — battle-tested libraries like smp eliminate these
7. 2-phase training (freeze encoder then unfreeze) is the proper approach for pretrained backbones
8. Early stopping prevents overfitting on small datasets — patience=20 is a good default
9. Per-epoch time tracking helps plan overnight training runs
10. Pixel accuracy is misleading for imbalanced datasets — always report IoU

---

## Deployment
- **Gradio app**: `app.py` — upload satellite image, get segmentation mask
- **Run locally**: `python app.py` → opens at http://127.0.0.1:7860
- **Share publicly**: `demo.launch(share=True)` → gives public URL
- **Model file**: `best_unet_256.pth` (124MB) — needed for deployment
- **Note**: Only works with actual satellite imagery, not Google Maps screenshots
9. Per-epoch time tracking helps plan overnight training runs
