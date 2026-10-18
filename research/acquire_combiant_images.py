"""Resume public ZIP member acquisition; no model or foreign code execution."""
import argparse
import hashlib
import json
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from inspect_combiant_archive import OUT, RemoteArchive
from acquire_combiant_public import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    inventory = json.loads((OUT / 'archive_inventory.json').read_text())
    lock = OUT / 'image_acquisition.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    path = OUT / 'image_acquisition_manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {
        'doi': inventory['doi'], 'license': inventory['license'], 'members': {},
        'limitations': ['Per-member CRC verified, full archive MD5 unverified.',
                        'Published inputs may include original author preprocessing.']}
    try:
        entries = [e for e in inventory['entries'] if e['name'].startswith('inference/input/') and not e['directory']]
        pairs = json.loads((OUT / 'pairing_audit.json').read_text())
        previews = set(pairs['example_image_members'])
        entries.sort(key=lambda e: (e['name'] not in previews, e['name']))
        if args.preview_only:
            entries = [e for e in entries if e['name'] in previews]
        pending = []
        for e in entries:
            dst = OUT / 'replication_subset' / e['name']
            old = manifest['members'].get(e['name'])
            if old:
                if not dst.exists() or hashlib.sha256(dst.read_bytes()).hexdigest() != old['sha256']:
                    raise RuntimeError('Previously acquired image changed: ' + e['name'])
            else:
                pending.append(e)
        manifest.update(status='acquiring', pid=os.getpid(), expected_images=299,
                        scope='preview' if args.preview_only else 'all_evaluation_inputs')
        write(path, manifest)

        def acquire(e):
            remote = RemoteArchive(inventory['archive']['links']['self'], inventory['archive']['size'])
            remote.seek(e['header_offset'])
            header = remote.read(30)
            fields = struct.unpack('<4s5H3I2H', header)
            if fields[0] != b'PK\x03\x04' or fields[3] != e['compression']:
                raise RuntimeError('ZIP local header mismatch')
            name_length, extra_length = fields[-2:]
            metadata = remote.read(name_length + extra_length)
            name = metadata[:name_length].decode('utf-8' if fields[2] & 0x800 else 'cp437')
            if name != e['name']:
                raise RuntimeError('ZIP local filename mismatch')
            payload = remote.read(e['compressed_size'])
            raw = payload if e['compression'] == 0 else zlib.decompress(payload, -15) if e['compression'] == 8 else None
            if raw is None or len(raw) != e['size'] or f'{zlib.crc32(raw) & 0xffffffff:08x}' != e['crc32']:
                raise RuntimeError('Image size or ZIP CRC mismatch: ' + name)
            dst = OUT / 'replication_subset' / name
            if not dst.resolve().is_relative_to((OUT / 'replication_subset').resolve()):
                raise RuntimeError('Unsafe member path')
            dst.parent.mkdir(parents=True, exist_ok=True)
            partial = dst.with_suffix(dst.suffix + '.partial'); partial.write_bytes(raw); partial.replace(dst)
            return name, {'status': 'downloaded_crc_verified', 'size': len(raw),
                          'sha256': hashlib.sha256(raw).hexdigest(), 'crc32': e['crc32']}

        errors = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(acquire, e): e for e in pending}
            for future in as_completed(futures):
                e = futures[future]
                try:
                    name, row = future.result(); manifest['members'][name] = row
                except Exception as exc:
                    errors.append({'member': e['name'], 'error': repr(exc)})
                manifest.update(updated_utc=datetime.now(timezone.utc).isoformat(), failures=errors)
                write(path, manifest)
                print(json.dumps({'acquired_images': len(manifest['members']), 'failures': len(errors),
                                  'member': e['name']}), flush=True)
        manifest.update(status='failed' if errors else 'preview_acquired' if args.preview_only else 'evaluation_inputs_acquired_pending_audit')
        write(path, manifest)
        if errors:
            raise RuntimeError(f'{len(errors)} members failed; preserve and resume only missing images')
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
