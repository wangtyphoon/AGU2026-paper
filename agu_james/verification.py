"""Syntax, import-safety and dependency checks (no data execution)."""
import ast
import importlib
import json
from pathlib import Path
import sys
from agu_james import runtime

def check(import_modules=True):
    root=Path(__file__).resolve().parents[1]
    issues=[];imported=[];missing=[];syntax_checked=0
    for p in sorted((root/'agu_james').rglob('*.py')):
        ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,10))
        syntax_checked+=1
        if not import_modules or p.name=='__init__.py':continue
        name='.'.join(p.relative_to(root).with_suffix('').parts)
        try:importlib.import_module(name);imported.append(name)
        except ModuleNotFoundError as exc:missing.append(dict(module=name,dependency=exc.name))
        except Exception as exc:issues.append(f'{name}: {type(exc).__name__}: {exc}')
    result=dict(syntax_checked_modules=syntax_checked,imported_modules=len(imported),
                unavailable_dependencies=missing,issues=issues,python=sys.version,
                validation_scope='syntax/import/dependency checks; not full training or HPC reproduction')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    runtime.write_json(runtime.artifact_root()/'verification/check.json',result)
    if issues:raise RuntimeError('Import verification failed; inspect check.json')
    return result
