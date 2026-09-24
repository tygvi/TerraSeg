# TerraSeg

Land cover segmentation on DeepGlobe satellite images. A U-Net with a ResNet34 encoder that labels every pixel in an image as one of six classes: urban, agriculture, rangeland, forest, water, or barren.

This repo has the training notebook I worked in, the experiment log, and a small Gradio app for trying the model out.

## Results

The best model so far is `best_unet_256.pth` — smp's U-Net with a pretrained ResNet34 encoder, trained with DiceLoss, flip and rotation augmentation, and a cosine LR schedule for 200 epochs (early stopped at 167).

| Class       | IoU    | Dice   |
|-------------|--------|--------|
| urban       | 0.3475 | 0.4198 |
| agriculture | 0.6822 | 0.7457 |
| rangeland   | 0.1844 | 0.2545 |
| forest      | 0.1727 | 0.1953 |
| water       | 0.1626 | 0.1983 |
| barren      | 0.1854 | 0.2282 |
| **Mean**    | **0.2891** | **0.3403** |

Pixel accuracy comes out at 87.24%, but that number is misleading — agriculture covers most of the data, so predicting everything as agriculture alone would get you ~60%. IoU is the one to look at. Agriculture is solid, the rarer classes are still weak.

## What worked and what didn't

The earlier runs are all in `experiment_log.md`, but short version:

- A custom U-Net written from scratch topped out at ~0.19 mean IoU. Weighted loss helped the rare classes (rangeland 0.03 → 0.15) but pulled agriculture down from 0.66 to 0.53.
- Bolting a pretrained ResNet34 onto the custom decoder made things *worse* — same LR destroyed the pretrained features, and lowering it didn't help either. The custom decoder also had skip connection and upsampling bugs I didn't spot until later.
- Switching to smp's U-Net (someone else's tested decoder) with DiceLoss, augmentation, and a cosine scheduler got it to 0.29. DiceLoss handles the imbalance without hand-tuning class weights.
- Dropped the `unknown` class. In the original DeepGlobe challenge it's an ignore class and there was almost no data for it.

## Running the app

Install the dependencies and start it:

```
pip install -r requirements.txt
python app.py
```

It comes up at http://127.0.0.1:7860. The three images in `test_samples/` show up as examples you can click, or you can upload your own. There's a colour legend under the title showing what each class looks like in the mask.

### Sharing it with other people

The localhost link only works on your own machine. To hand someone a URL they can open, launch with `share=True`. Easiest way without editing anything is to start it from a Python shell:

```
python -c "import app; app.demo.launch(share=True)"
```

It prints a `https://....gradio.live` URL you can send to anyone. Two things to know: the link dies the moment you stop the script or your laptop sleeps, and it's best-effort for up to a week. It's fine for showing someone, not for leaving up.

If you want it online permanently, Gradio can push it to Hugging Face Spaces for free (needs an HF account and `pip install gradio`):

```
gradio deploy
```

It asks for a Space name and uploads from the current folder. Remember to put `best_unet_256.pth` in there too, since it's not in this repo — Spaces handles a ~93 MB file fine.

## Model weights

`best_unet_256.pth` (~93 MB) is not committed here. Drop it in the project root next to `app.py` before running, otherwise the app won't start.

## Data

DeepGlobe Land Cover Classification dataset — 803 training images at 2448x2448, split 80/20 into train/val, resized to 256x256 for training. Trained on an M-series Mac using MPS.
