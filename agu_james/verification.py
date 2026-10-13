"""Source integrity, import-safety and dependency checks (no data execution)."""
import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from agu_james import runtime

def check(import_modules=True):
    root=Path(__file__).resolve().parents[1]
    manifest_path=root/'provenance/sources.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else []
    issues=[];unchanged=0
    release_verified=0;control_verified=0
    release_manifest=root/'provenance/files.json'
    if release_manifest.is_file():
        for item in json.loads(release_manifest.read_text(encoding='utf-8'))['files']:
            file=root/item['file']
            if hashlib.sha256(file.read_bytes()).hexdigest()!=item['sha256']:issues.append('release file changed: '+item['file'])
            else:release_verified+=1
    controls=root/'provenance/reference_sources.json'
    if controls.is_file():
        for item in json.loads(controls.read_text(encoding='utf-8')):
            file=Path(item['source'])
            if file.exists():
                if hashlib.sha256(file.read_bytes()).hexdigest()!=item['source_sha256']:issues.append('control reference changed: '+str(file))
                else:control_verified+=1
    for item in manifest:
        source=Path(item['source'])
        if source.exists():
            if hashlib.sha256(source.read_bytes()).hexdigest()!=item['source_sha256']:issues.append('source changed: '+str(source))
            else:unchanged+=1
        target=root/item['destination']
        if hashlib.sha256(target.read_bytes()).hexdigest()!=item['output_sha256']:issues.append('curated hash changed: '+str(target))
    imported=[];missing=[]
    for p in sorted((root/'agu_james').rglob('*.py')):
        ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,10))
        if not import_modules or p.name=='__init__.py':continue
        name='.'.join(p.relative_to(root).with_suffix('').parts)
        try:importlib.import_module(name);imported.append(name)
        except ModuleNotFoundError as exc:missing.append(dict(module=name,dependency=exc.name))
        except Exception as exc:issues.append(f'{name}: {type(exc).__name__}: {exc}')
    result=dict(source_records=len(manifest),verified_unchanged_sources=unchanged,imported_modules=len(imported),
                verified_release_files=release_verified,verified_control_references=control_verified,
                source_manifest_status='available in source checkout' if manifest else 'not included in installed wheel; see source release provenance',
                unavailable_dependencies=missing,issues=issues,python=sys.version,
                validation_scope='syntax/import/source checks; not full training or HPC reproduction')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    runtime.write_json(runtime.artifact_root()/'verification/check.json',result)
    if issues:raise RuntimeError('Source/import verification failed; inspect check.json')
    return result
