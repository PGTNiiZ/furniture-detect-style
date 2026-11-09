import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch
import numpy as np

from .data import rgb, transform, read_rows
from .recommend import canonical_type, decide, replacements
from .train import choose_device, load_model


def classify_crop(model, payload, image, device):
    tensor = transform(payload["config"]["image_size"])(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor)
    return {key: {name: float(scores[i]) for name, i in mapping.items()}
            for key, mapping in payload["mapping"].items()
            for scores in [output[key].softmax(1)[0].cpu()]}


def classify_image(image, checkpoint, output, device='auto'):
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; choose a new file')
    device = choose_device(device)
    model, payload = load_model(checkpoint, device)
    with Image.open(image) as source:
        scores = classify_crop(model, payload, source, device)
    result = {'image':str(image), 'checkpoint':str(checkpoint), 'device':str(device), 'scores':scores,
              'predictions':{task:{'label':max(values,key=values.get), 'confidence':max(values.values())}
                             for task,values in scores.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result


def analyze(image, checkpoint, detector_weights, target, output, catalog=None, device="auto", confidence=.6, detector_confidence=.25, keep_threshold=.6, replace_threshold=.2, top_k=3):
    from ultralytics import YOLO
    output = Path(output)
    if output.exists():
        raise ValueError("Output exists; choose a new report directory")
    device = choose_device(device)
    model, payload = load_model(checkpoint, device)
    if set(payload['mapping']) != {'type', 'style'}:
        raise ValueError('Room inference requires a checkpoint with both type and style heads')
    if target not in payload["mapping"]["style"]:
        raise ValueError(f"Target style unavailable; choose from {list(payload['mapping']['style'])}")
    detector = YOLO(detector_weights)
    with Image.open(image) as im:
        original = rgb(im)
    # Ultralytics ndarray input is BGR; align boxes with the EXIF-corrected crop canvas.
    results = detector.predict(source=np.asarray(original)[:,:,::-1].copy(), conf=detector_confidence, device=str(device), verbose=False)
    canvas, objects = original.copy(), []
    drawer = ImageDraw.Draw(canvas)
    catalog_rows = read_rows(catalog) if catalog else []
    supported = {canonical_type(name) for name in payload["mapping"]["type"]}
    for i, box in enumerate(results[0].boxes):
        label = results[0].names[int(box.cls.item())]
        if canonical_type(label) not in supported:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().tolist()]
        x1, y1, x2, y2 = max(0,x1), max(0,y1), min(original.width,x2), min(original.height,y2)
        if x2 <= x1 or y2 <= y1:
            continue
        scores = classify_crop(model, payload, original.crop((x1,y1,x2,y2)), device)
        predicted_type = max(scores["type"], key=scores["type"].get)
        decision = decide(scores["style"], target, minimum_confidence=confidence, keep_threshold=keep_threshold, replace_threshold=replace_threshold)
        if canonical_type(predicted_type) != canonical_type(label) or scores["type"][predicted_type] < confidence:
            decision["action"] = "review"
            decision["reason"] = "Detector/classifier type disagreement or low type confidence"
        candidates = replacements(catalog_rows, label, target, top_k) if decision["action"] == "replace_candidate" else []
        item = {"object_id": i, "detector_type": label, "detector_score": float(box.conf.item()),
                "bbox_xyxy": [x1,y1,x2,y2], "attribute_scores": scores, **decision,
                "replacement_candidates": candidates}
        objects.append(item)
        color = {"keep_candidate":"green", "replace_candidate":"red", "review":"orange"}[decision["action"]]
        drawer.rectangle((x1,y1,x2,y2), outline=color, width=3)
        drawer.text((x1+3,y1+3), f"{i}: {label} / {decision['action']}", fill=color)
    objects.sort(key=lambda item: item["target_style_score"])
    report = {"target_style": target, "objects": objects,
              "settings":dict(detector_confidence=detector_confidence,minimum_confidence=confidence,
                              keep_threshold=keep_threshold,replace_threshold=replace_threshold,top_k=top_k),
              "status": "ok" if objects else "no_supported_furniture_detected",
              "limitations": ["Single-label style mismatch is not proof of incompatibility",
                              "Synthetic product images may not generalize to room crops",
                              "Catalog suggestions use labels; not a trained compatibility ranking",
                              "No physical size, price, room layout, repainting or material-change estimates"]}
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    canvas.save(output / "annotated.jpg")
    return report
