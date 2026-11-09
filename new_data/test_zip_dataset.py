import csv
import io
import json
import os
import zipfile

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from new_data.zip_dataset import ZipFurnitureDataset, loader_from_config


class TracedDataset(ZipFurnitureDataset):
    def __getitem__(self, index):
        tensor, labels = super().__getitem__(index)
        return index, os.getpid(), self._pid, tensor, labels


def make_data(tmp_path, corrupt=False):
    archive = tmp_path/'test.zip'
    rows = []
    with zipfile.ZipFile(archive, 'w') as z:
        for i in range(8):
            data = io.BytesIO()
            Image.new('RGB', (8, 8), (i*20, 0, 0)).save(data, format='PNG')
            z.writestr(f'p/{i}.png', b'bad image' if corrupt and i == 0 else data.getvalue())
            rows.append({'image_id': str(i), 'image_path': f'{i}.png', 'type':'bed', 'style':'modern'})
    return archive, rows


def test_shuffled_spawn_workers_own_handles_and_read_once(tmp_path):
    archive, rows = make_data(tmp_path)
    data = TracedDataset(rows, archive, {'type': {'bed':0}, 'style': {'modern':0}}, prefix='p/', size=16)
    loader = DataLoader(data, batch_size=2, num_workers=2, shuffle=True,
                        generator=torch.Generator().manual_seed(42), persistent_workers=True)
    for _ in range(2):
        batches = list(loader)
        indices = [n for batch in batches for n in batch[0].tolist()]
        pids = [n for batch in batches for n in batch[1].tolist()]
        owners = [n for batch in batches for n in batch[2].tolist()]
        assert sorted(indices) == list(range(8))
        assert indices != list(range(8))
        assert pids == owners and len(set(pids)) == 2 and os.getpid() not in pids


def test_corrupt_and_missing_archive_fail_with_sample(tmp_path):
    archive, rows = make_data(tmp_path, corrupt=True)
    data = ZipFurnitureDataset(rows, archive, {'type': {'bed':0}}, prefix='p/')
    with pytest.raises(RuntimeError, match='sample 0'):
        data[0]
    archive.unlink()
    with pytest.raises(FileNotFoundError, match='archive unavailable'):
        data[1]


def test_config_manifest_and_mapping(tmp_path):
    archive, rows = make_data(tmp_path)
    manifest = tmp_path/'rows.csv'
    with manifest.open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'storage_mode':'zip','corrupt_image_policy':'error',
        'archive_path':str(archive),'archive_prefix':'p/','manifest_path':str(manifest),
        'snapshot_id':'test-only','batch_size':3,'num_workers':0,'seed':42}), encoding='utf8')
    batches = list(loader_from_config(config, {'type':{'bed':0}, 'style':{'modern':0}}))
    assert sum(len(batch[0]) for batch in batches) == 8
    assert all(batch[1]['type'].eq(0).all() for batch in batches)


def test_manifest_style_not_zip_folder_name(tmp_path):
    archive=tmp_path/'source.zip'
    data=io.BytesIO(); Image.new('RGB',(8,8),'white').save(data,format='PNG')
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('p/bed/nordic/a.png',data.getvalue())
    manifest=[{'image_id':'example','image_path':'bed/nordic/a.png',
               'type':'bed','style':'scandinavian'}]
    dataset=ZipFurnitureDataset(manifest,archive,
        {'type':{'bed':0},'style':{'nordic':0,'scandinavian':1}},prefix='p/',size=16)
    _,labels=dataset[0]
    assert labels['style'].item()==1
    dataset.close()
