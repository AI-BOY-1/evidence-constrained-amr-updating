"""Verify delivered bytes, source bindings and Python syntax without datasets."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    m=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    assert m['status']=='passed'
    for rel,rec in m['files'].items():assert sha(ROOT/rel)==rec['sha256'],rel
    for p in ROOT.rglob('*.py'):
        if '__pycache__' not in p.parts and '_runtime' not in p.parts:ast.parse(p.read_text(encoding='utf-8'),filename=str(p.relative_to(ROOT)))
    original=json.loads((ROOT/'historical_frozen/ANI_SOURCE_BINDING.json').read_text(encoding='utf-8'))
    for name,h in original['code'].items():assert sha(ROOT/'historical_frozen/source_snapshots'/name)==h,name
    print(json.dumps({'status':'passed','files':len(m['files']),'ANI_core_hashes':len(original['code'])}))
if __name__=='__main__':main()
