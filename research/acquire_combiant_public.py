"""Fetch declared public replication members without downloading its container.

ZIP CRC checks verify individual members; full archive MD5 remains unverified.
"""
import argparse
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from inspect_combiant_archive import OUT, RemoteArchive


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('metadata', 'evaluation_images'), default='metadata')
    args = parser.parse_args()
    inventory = json.loads((OUT / 'archive_inventory.json').read_text(encoding='utf-8'))
    lock = OUT / 'acquisition.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    path = OUT / 'acquisition_manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {
        'doi': inventory['doi'], 'record_id': inventory['record_id'],
        'license': inventory['license'], 'archive': inventory['archive'], 'members': {},
        'limitations': ['Selective extraction with each member ZIP CRC verified.',
                        'Full archive MD5 is unverified; no full archive acquisition claim.',
                        'Existing published outputs are reference data, not our new model results.']}
    try:
        manifest.update(status='acquiring', pid=os.getpid(), scope=args.scope)
        write(path, manifest)
        remote = RemoteArchive(inventory['archive']['links']['self'], inventory['archive']['size'])
        remote.seek(remote.size - 1048576)
        remote.read(1048576)  # one contiguous span contains the ZIP64 directory.
        with zipfile.ZipFile(remote) as archive:
            infos = sorted(archive.infolist(), key=lambda e: e.header_offset)
            selected = []
            for index, info in enumerate(infos):
                name = info.filename
                if info.is_dir():
                    continue
                take = (name in ('Readme.md', 'requirements.txt',
                                 'CombiANT_raw_data/combiANT_user_measurements.csv', 'inference/output.csv')
                        or name.startswith('scripts/') or name.endswith('.ipynb'))
                if args.scope == 'evaluation_images':
                    take = name.startswith('inference/input/') and Path(name).suffix.lower() in ('.png', '.jpg', '.jpeg')
                if take:
                    selected.append((info, infos[index + 1].header_offset if index + 1 < len(infos) else archive.start_dir))
            manifest['scope_expected_members'] = len(selected)
            for info, end in selected:
                target = OUT / 'replication_subset' / info.filename
                if not target.resolve().is_relative_to((OUT / 'replication_subset').resolve()):
                    raise RuntimeError('Unsafe archive member path')
                old = manifest['members'].get(info.filename)
                if old and target.exists():
                    if sha(target) != old['sha256'] or target.stat().st_size != info.file_size:
                        raise RuntimeError('Changed acquired member: ' + info.filename)
                    continue
                remote.seek(info.header_offset)
                remote.read(end - info.header_offset)
                data = archive.read(info)  # zipfile checks the recorded member CRC.
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(target.suffix + '.partial')
                partial.write_bytes(data)
                partial.replace(target)
                manifest['members'][info.filename] = {'size': len(data), 'sha256': sha(target),
                    'crc32': f'{info.CRC:08x}', 'status': 'downloaded_crc_verified'}
                manifest['updated_utc'] = datetime.now(timezone.utc).isoformat()
                write(path, manifest)
                print(json.dumps({'member': info.filename, 'bytes': len(data),
                                  'acquired_members': len(manifest['members'])}), flush=True)
        manifest.update(status=args.scope + '_acquired_pending_audit',
                        updated_utc=datetime.now(timezone.utc).isoformat())
        write(path, manifest)
        print(json.dumps({'status': manifest['status'], 'members': len(manifest['members'])}), flush=True)
    except Exception as error:
        manifest.update(status='failed', error=repr(error), updated_utc=datetime.now(timezone.utc).isoformat())
        write(path, manifest)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
