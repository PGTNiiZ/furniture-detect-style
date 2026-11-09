"""Generate deterministic representative images for preliminary visual label review."""
import argparse
import csv
import io
import json
import os
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf8'))
    archive=Path(os.path.expandvars(cfg['archive_path']))
    with Path(args.manifest).open(encoding='utf8',newline='') as f: rows=list(csv.DictReader(f))
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    styles=sorted({r['style'] for r in rows}); types=sorted({r['type'] for r in rows}); selected=[]
    for index,style in enumerate(styles):
        desired=types[index % len(types)]
        choices=[r for r in rows if r['style']==style and r['type']==desired] or [r for r in rows if r['style']==style]
        selected.append(sorted(choices,key=lambda r:r['image_id'])[0])
    with (out/'samples.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image_id','image_path','type','style'],extrasaction='ignore'); writer.writeheader(); writer.writerows(selected)
    with zipfile.ZipFile(archive) as z:
        for page in range((len(selected)+19)//20):
            chunk=selected[page*20:(page+1)*20]
            sheet=Image.new('RGB',(1000,800),'white'); draw=ImageDraw.Draw(sheet)
            for index,row in enumerate(chunk):
                x=(index%5)*200; y=(index//5)*200
                with Image.open(io.BytesIO(z.read(cfg['archive_prefix']+row['image_path']))) as im:
                    thumbnail=ImageOps.contain(im.convert('RGB'),(190,162))
                sheet.paste(thumbnail,(x,y+32)); draw.text((x+2,y+2),row['image_id'],fill='black')
                draw.text((x+2,y+16),f"{row['type']} | {row['style']}",fill='black')
            sheet.save(out/f'label-samples-{page+1}.png')
    print(json.dumps({'styles':len(styles),'types':len(types),'sample_images':len(selected),
                      'note':'one example per style, rotating type; style semantics need expert evidence'}))


if __name__ == '__main__': main()
