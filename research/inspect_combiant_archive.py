"""Inventory the public replication ZIP using ordinary HTTP range requests.

Only inventory bytes are fetched here. No remote source code is executed.
"""
import ncs_paths  # portable I/O paths; original computational rules preserved
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

OUT = ncs_paths.data_path('experiments/combiant_public_v1')
API = 'https://zenodo.org/api/records/13893288'


class RemoteArchive(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.requests = []
        self.cache = []
        self.disk_cache = []

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError('negative seek')
        return self.position

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.position
        n = min(n, self.size - self.position)
        if n <= 0:
            return b''
        start, end = self.position, self.position + n - 1
        for base, length, path in self.disk_cache:
            if base <= start and end < base + length:
                with path.open('rb') as stream:
                    stream.seek(start - base)
                    data = stream.read(n)
                if len(data) != n:
                    raise RuntimeError('Truncated local archive span')
                self.position += n
                return data
        for base, data in self.cache:
            if base <= start and end < base + len(data):
                self.position += n
                return data[start - base:end - base + 1]
        pieces = []
        cursor = start
        while cursor <= end:
            piece_end = min(end, cursor + 524287)
            req = urllib.request.Request(self.url, headers={'Range': f'bytes={cursor}-{piece_end}'})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=60) as response:
                        expected = f'bytes {cursor}-{piece_end}/{self.size}'
                        if response.status != 206 or response.headers.get('Content-Range') != expected:
                            raise RuntimeError('Server did not honor exact public archive byte range')
                        piece = response.read()
                    if len(piece) != piece_end - cursor + 1:
                        raise RuntimeError('Truncated byte range')
                    pieces.append(piece)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            cursor = piece_end + 1
        data = b''.join(pieces)
        self.requests.append({'start': start, 'end': end, 'sha256': hashlib.sha256(data).hexdigest()})
        self.cache.append((start, data))
        self.cache = self.cache[-4:]
        self.position += n
        return data


def main():
    OUT.mkdir(exist_ok=True)
    with urllib.request.urlopen(API, timeout=45) as response:
        metadata = json.load(response)
    (OUT / 'repository_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    item = metadata['files'][0]
    remote = RemoteArchive(item['links']['self'], item['size'])
    with zipfile.ZipFile(remote) as archive:
        entries = [{'name': e.filename, 'size': e.file_size, 'compressed_size': e.compress_size,
                    'crc32': f'{e.CRC:08x}', 'header_offset': e.header_offset,
                    'compression': e.compress_type, 'directory': e.is_dir()}
                   for e in archive.infolist()]
    inventory = {'status': 'inventory_only', 'record_id': metadata['id'], 'doi': metadata['doi'],
                 'license': metadata['metadata']['license'], 'archive': item, 'entries': entries,
                 'inventory_byte_requests': remote.requests,
                 'limitations': ['Archive MD5 is not verified until the entire archive is downloaded.',
                                 'Inventory does not establish image-label pairing or isolate independence.']}
    (OUT / 'archive_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    print(json.dumps({'entries': len(entries), 'uncompressed_bytes': sum(e['size'] for e in entries),
                      'extensions': dict(Counter(Path(e['name']).suffix.lower() for e in entries)),
                      'directories': dict(Counter('/'.join(e['name'].split('/')[:-1]) for e in entries)),
                      'small_text_files': [e['name'] for e in entries if Path(e['name']).suffix.lower() in
                                           ('.txt', '.md', '.csv', '.xlsx', '.json', '.yaml', '.yml')]}, indent=2))


if __name__ == '__main__':
    main()
