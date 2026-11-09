import hashlib
import json
import struct
import zipfile

import pytest
from new_data.inventory_local_sources import inventory


def archive_fixture(tmp_path):
    path=tmp_path/'sources.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('furniture_sources/A/image.jpg',b'image contents')
        z.writestr('furniture_sources/A/labels.json',b'{}')
        z.writestr('furniture_sources/B/nested.zip',b'not opened')
    return path


def test_counts_file_names_without_claiming_independent_images(tmp_path):
    archive=archive_fixture(tmp_path)
    result=inventory(archive,tmp_path/'out',archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest())
    assert result['full_member_crc_verified'] and result['trusted_sha256_supplied']
    assert result['file_entries']==3 and result['image_extension_entries']==1
    assert result['sources']['A']['annotation_extension_entries']==1
    assert result['sources']['B']['nested_archives']==1
    saved=json.loads((tmp_path/'out'/'inventory.json').read_text(encoding='utf8'))
    assert any('not independent-image' in line for line in saved['limitations'])


def test_rejects_wrong_size_and_corrupt_member(tmp_path):
    archive=archive_fixture(tmp_path)
    with pytest.raises(ValueError,match='size mismatch'):
        inventory(archive,tmp_path/'badsize',1)
    data=bytearray(archive.read_bytes())
    name_len,extra_len=struct.unpack_from('<HH',data,26)
    data[30+name_len+extra_len]^=1
    archive.write_bytes(data)
    with pytest.raises(zipfile.BadZipFile,match='CRC'):
        inventory(archive,tmp_path/'badcrc')
    assert not (tmp_path/'badcrc').exists()
