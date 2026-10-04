# Project status — end of Week 2, Day 5

## Done
- EDA complete; 8 target classes (tinea dropped, SkinDisNet documented)
- Manifest built: 18,126 images
- pHash dedupe: removed 2,623 near-duplicates → 15,503 images
- Vitiligo pHash clustering (threshold ≤12): fixes patient-level leakage in SkinDiseaseBD/SkinDiseaseClassification
- Patient-level 70/15/15 split: 10,854 train / 2,328 val / 2,321 test. **Zero group leakage.**
- DVC + Google Drive remote works for teammate
- CI passing on GitHub Actions
- Augmentation pipeline: `get_train_transforms`, `get_train_transforms_aggressive` (vitiligo), `get_eval_transforms`, `get_tencrop_eval_transforms`
- Dataset class: `src/data/dataset.py` with per-class transform overrides

## Remaining before Week 3 model work
- [ ] Refactor `src/model.py` to two-backbone interface: `build_model(backbone, num_classes, pretrained=True)` supporting `"efficientnet_b0"` and `"resnet50"`
- [ ] Change `weights=None` → pretrained weights (critical)

## Week 3 plan
- Model 1: EfficientNet-B0 pretrained (primary)
- Model 4 ablation: ResNet50 pretrained (for report comparison table)
- Write `src/train.py` (training loop)

## Locked decisions
- No training from scratch — pretrained weights required
- Class names hardcoded alphabetically in `dataset.py` (do not derive per-file)
- Vitiligo: aggressive crop (scale 0.5–1.0) for training only; eval uses default transforms

## TODO for Week 4 (do NOT forget)
- [ ] Use `get_tencrop_eval_transforms` for FINAL test-set numbers
- [ ] Compare per-class accuracy: center crop vs TenCrop
- [ ] Specifically check vitiligo — it's the class most likely to benefit (train/eval distribution mismatch)
- [ ] DDI fairness analysis (skin tone accuracy gap) — Week 5

## Known limitations to document
- DermNet watermark visible on eczema images → potential short-cut learning
- Four source datasets have no patient IDs (acne, bcc, DermNet eczema, SkinDiseaseBD vitiligo). Vitiligo now clustered; others still carry residual leakage risk.
- Small class imbalance remains: nevus 1,708 vs acne 923 in train. Will evaluate with confusion matrix before deciding whether to rebalance.