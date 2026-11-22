# Applying the Model to Real Room Images

This section applies the trained model to real room photos that are not part of the training data. The goal is to see how well the system identifies the **type** and **style** of each piece of furniture in a photo when used in practice.

## Pipeline

A room photo usually contains several pieces of furniture, so inference runs in two stages:

1. **Detector (YOLOv8n, COCO-pretrained).** Finds each piece of furniture and draws a bounding box. COCO labels are mapped to the project's categories (`couch→sofa`, `dining table→table`, `chair→chair`, `bed→bed`). Other labels, such as `potted plant`, are not sent to the classifier.
2. **Project classifier (DenseNet121, condition C6, furniture-v3-style5 dataset, seed 123).** Takes the crop from each box and predicts type and style (5 styles) at the same time, using two heads on a shared backbone.

Box colours:
- **Green:** the detector and the classifier agree on the furniture type.
- **Red:** the detector and the classifier disagree on the type.
- **Grey:** the object is outside the model's furniture categories.

## Results

### Image 1: Bright living room by the window

Original (Pexels, photographer Mostaza, ID 27118108):

![Original image 1](../data/room-independent-review-v1/images/final-27118108.jpg)

Model output:

![Result image 1](../artifacts/real-room-demo-v3s5/final-27118108/annotated.png)

| # | Detector | Type (classifier) | Style (top-1 / top-2) | Visual check |
|---|---|---|---|---|
| 1 | couch 0.63 | **sofa** 0.73 | Industrial & Rustic 0.45 / Modern & Minimalist 0.38 | Type is correct. The style looks more like Modern & Minimalist, which the model ranks second, only 0.07 behind first place. |
| 0, 2 | potted plant | not classified | – | Correctly filtered out. |

**Note:** The wooden stump side table on the right was not detected at all, because the COCO detector does not recognise a table of this shape. The floor lamp and the basket are outside the model's categories.

### Image 2: Dim seating corner with orange lighting

Original (Pexels, photographer Erin Abbott, ID 12821240):

![Original image 2](../data/room-independent-review-v1/images/development-12821240.jpg)

Model output:

![Result image 2](../artifacts/real-room-demo-v3s5/development-12821240/annotated.png)

| # | Detector | Type (classifier) | Style (top-1) | Visual check |
|---|---|---|---|---|
| 0 | chair 0.92 | **chair** 1.00 | Classic & Ornamental 0.75 | Type is correct. It is a retro wooden chair with a red leather seat, so the style is plausible. |
| 1 | dining table 0.50 | **table** 0.94 | Cultural & Eclectic 0.41 | Type is correct, but the style is uncertain (top 3: 0.41 / 0.26 / 0.22). |
| 4 | dining table 0.33 | **table** 0.86 | Cultural & Eclectic 0.41 | Duplicate of #1; the box covers only the tabletop. |
| 2 | couch 0.44 | **table** 0.82 ✗ | Classic & Ornamental 0.57 | Wrong. It is a sofa, but the table and chair in front cover most of the crop, so the classifier answers with the most prominent object in the crop. |
| 3 | chair 0.38 | **table** 0.98 ✗ | Modern & Minimalist 0.39 | Wrong at both stages. It is the sofa arm or a throw in the bottom-left corner, not a chair or a table. |

## Summary

**What works well**
- When a crop contains one clearly visible piece of furniture, such as the sofa in image 1 or the chair in image 2, the model predicts the type correctly with high confidence (0.73–1.00). This matches the type accuracy of 0.84 on the real-source test set.
- The system correctly filters out non-furniture objects such as plants.
- Comparing the detector with the classifier (red boxes) flags suspicious predictions for human review. Both wrong predictions in image 2 were already flagged by the system.

**Limitations**
1. **Style is not yet reliable.** Most style confidences are around 0.4–0.5, and the top-1 and top-2 scores are often close. This matches the style macro-F1 of only 0.29 on the real-source test set. Style is also subjective by nature, so it is better to show the top-2 or top-3 styles than a single answer.
2. **The detector is the bottleneck.** It is an off-the-shelf COCO YOLOv8n with no extra training, so it misses unusually shaped furniture (the stump table), produces duplicate boxes (tables #1 and #4), and makes false detections (#3).
3. **Overlapping objects.** The crop of a partly hidden object also contains other furniture. The classifier was trained on single-product images, so it answers with whichever object is more prominent.
4. **Unusual lighting.** Image 2 has strong orange lighting, very different from the training images, which lowers style confidence.
5. **Not a quantitative evaluation.** The room images do not yet have human-verified annotations, so this is a qualitative demonstration on two images only. Confidence values are uncalibrated softmax scores.

**Possible improvements:** train the detector on annotated room images; apply class-agnostic NMS to remove duplicate boxes; add lighting and colour augmentation; and add real room images with style labels to the training set.

## How to reproduce

```powershell
.venv-room\Scripts\python.exe -m scripts.detect_then_classify `
  --image data\room-independent-review-v1\images\final-27118108.jpg `
  --checkpoint artifacts\runs\furniture-v3-style5\densenet121-C6-123\best.pt `
  --config configs\inference\room-yolov8n-coco-baseline.yaml `
  --output artifacts\real-room-demo-v3s5\final-27118108 --device cpu --top-k 3 --save-crops
```

- Classifier SHA-256: `8cc56c4433a914cf1fde4f1291dd9ea9a065cac4e1cf5e8fbbf3a87433f66984`
- Detector SHA-256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`
- All outputs (`annotated.png`, `objects.csv`, `result.json`, crops): `artifacts/real-room-demo-v3s5/`
