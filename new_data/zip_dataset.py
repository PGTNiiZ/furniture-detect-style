"""Experimental ZIP adapter for the existing FurnitureDataset tensor/label contract.

This module does not alter the frozen v2 training suite or approve a snapshot.
"""
import csv
import json
import os
import zipfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from furniture.data import FurnitureDataset


class ZipFurnitureDataset(FurnitureDataset):
    def __init__(self, rows, archive, mapping, prefix='', size=224, augment=False):
        super().__init__(rows, '.', mapping, size=size, augment=augment)
        self.archive = Path(archive).expanduser().resolve()
        if not self.archive.is_file():
            raise FileNotFoundError(f'Dataset archive unavailable: {self.archive}')
        self.prefix = prefix
        self._zip, self._pid = None, None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_zip'], state['_pid'] = None, None
        return state

    def close(self):
        if self._zip is not None:
            self._zip.close()
        self._zip, self._pid = None, None

    def __getitem__(self, index):
        row = self.rows[index]
        if not self.archive.is_file():
            raise FileNotFoundError(f'Dataset archive unavailable: {self.archive}')
        try:
            if self._pid != os.getpid():
                if self._zip is not None:
                    self._zip.close()
                self._zip = zipfile.ZipFile(self.archive)
                self._pid = os.getpid()
            with self._zip.open(self.prefix + row['image_path']) as stream, Image.open(stream) as image:
                tensor = self.transform(image)
            return tensor, {key: torch.tensor(self.mapping[key][row[key]]) for key in self.mapping}
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            self.close()
            raise RuntimeError(f"Cannot read sample {row.get('image_id', index)} from {self.archive}: {error}") from error


def loader_from_config(config_path, mapping):
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if cfg.get('storage_mode') != 'zip' or cfg.get('corrupt_image_policy') != 'error':
        raise ValueError('Expected storage_mode=zip and corrupt_image_policy=error')
    archive = os.path.expandvars(cfg['archive_path'])
    manifest = Path(os.path.expandvars(cfg['manifest_path']))
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [row['image_id'] for row in rows]
    if not ids or len(set(ids)) != len(ids) or not cfg.get('snapshot_id'):
        raise ValueError('Snapshot ID and unique sample IDs are required')
    dataset = ZipFurnitureDataset(rows, archive, mapping, cfg.get('archive_prefix', ''),
                                  cfg.get('image_size', 224), cfg.get('augment', False))
    workers = cfg.get('num_workers', 0)
    options = {'num_workers': workers, 'persistent_workers': workers > 0 and cfg.get('persistent_workers', False)}
    if workers > 0 and cfg.get('prefetch_factor') is not None:
        options['prefetch_factor'] = cfg['prefetch_factor']
    generator = torch.Generator().manual_seed(cfg.get('seed', 42))
    return DataLoader(dataset, batch_size=cfg.get('batch_size', 16), shuffle=cfg.get('shuffle', True),
                      drop_last=cfg.get('drop_last', False), generator=generator, **options)
