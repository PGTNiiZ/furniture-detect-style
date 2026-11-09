"""Pipeline-only artificial fixture, NEVER use scores as furniture evidence."""
import json
from pathlib import Path
from PIL import Image
from .data import image_hash, dhash, write_rows


def make_smoke(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Output exists")
    (output / "images").mkdir(parents=True)
    for s, offset in [("train",0),("val",40),("test",80)]:
        rows = []
        for i in range(8):
            image = Image.new("RGB", (32,32), (offset+i, 50+i*5,100))
            name = f"{s}-{i}"
            image.save(output / "images" / f"{name}.png")
            rows.append({"image_id":name,"image_path":f"images/{name}.png",
                         "type":"chair" if i%2 else "table", "style":"modern" if i%2 else "rustic",
                         "source_split":"ARTIFICIAL_SMOKE_ONLY", "pixel_hash":image_hash(image), "dhash":dhash(image)})
        write_rows(output / f"{s}.csv",rows)
    (output / "audit.json").write_text(json.dumps({"ARTIFICIAL_SMOKE_ONLY":True}))
    return {"status":"artificial smoke fixtures created", "output":str(output)}
