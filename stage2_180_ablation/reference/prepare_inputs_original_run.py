import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import json, hashlib, zipfile, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root/'outputs/Stage2_Final_Results'
files = pd.read_csv(out/'TSB-AD-M-Eva.csv').file_name.tolist()
data = root/'work/data'
data.mkdir(exist_ok=True)
z = zipfile.ZipFile('D:/ms_cmesc/TSB-AD-M.zip')
rows = []
for i, name in enumerate(files):
    info = z.getinfo('TSB-AD-M/'+name)
    dest = data/name
    if not dest.exists():
        with z.open(info) as src, dest.open('wb') as dst:
            shutil.copyfileobj(src, dst)
    rows.append({'file':name,'bytes':info.file_size,'zip_crc32':f'{info.CRC:08x}',
                 'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
pd.DataFrame(rows).to_csv(out/'input_manifest.csv',index=False)
refzip = zipfile.ZipFile('D:/Downloads/Google/DB_ESC_Complete_Materials_Package.zip')
(out/'reference').mkdir(exist_ok=True)
for name in ['03_Code_and_Verification/data/eva180_final_results.csv','03_Code_and_Verification/run_tsb_ad_m.py']:
    (out/'reference'/Path(name).name).write_bytes(refzip.read(name))
shutil.copy2('D:/Downloads/Google/DBESCNeurocomputingManuscript.pdf',out/'DBESCNeurocomputingManuscript_provided.pdf')
official = out/'scripts/official_evaluation'
official.mkdir(exist_ok=True)
for name in ['evaluation/basic_metrics.py','utils/slidingWindows.py']:
    source=Path('D:/ms_cmesc/TSB-AD/TSB_AD')/name
    shutil.copy2(source,official/source.name)
print('EXTRACTED',len(rows),'files',sum(r['bytes'] for r in rows),'bytes',flush=True)
