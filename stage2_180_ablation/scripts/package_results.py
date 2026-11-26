import json, zipfile, hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]

def main():
    checks=json.loads((OUT/'validation_report.json').read_text(encoding='utf-8'))
    if not checks['completed'] or not checks['all_26_configuration_checks_passed']:
        raise RuntimeError('Refusing a final package with incomplete core results')
    if not checks.get('figure_manual_review_passed') or not checks.get('all_requested_core_outputs_present'):
        raise RuntimeError('Final output and figure verification must pass before packaging')
    entries=[]
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and 'scratch' not in path.relative_to(OUT).parts and path.name!='file_checksums.sha256':
            entries.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.relative_to(OUT).as_posix())
    (OUT/'file_checksums.sha256').write_text('\n'.join(entries)+'\n',encoding='utf-8')
    target=OUT.parent/'Stage2_Final_Results.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and 'scratch' not in path.relative_to(OUT).parts:
                z.write(path,Path(OUT.name)/path.relative_to(OUT))
    with zipfile.ZipFile(target) as z:
        bad=z.testzip()
        if bad: raise RuntimeError('ZIP integrity failed: '+bad)
        if any('/data/' in n and n.endswith('.csv') and 'reference' not in n for n in z.namelist()):
            raise RuntimeError('Raw data accidentally included')
    print(str(target),target.stat().st_size,'bytes',flush=True)

if __name__=='__main__':main()
