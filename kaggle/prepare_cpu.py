"""Kaggle CPU stage: official MSD archive -> validated nnU-Net 2D preprocessing."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

WORK = Path('/kaggle/working')
SCRATCH = Path('/tmp/brain-mri-prepare')
NAME = 'Dataset501_BrainTumour'
URL = 'https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar'
MD5 = '240a19d752f0d9e9101544901065d872'


def main():
    SCRATCH.mkdir(exist_ok=True)
    os.environ.update(nnUNet_raw=str(SCRATCH/'raw'), nnUNet_preprocessed=str(WORK/'preprocessed'),
                      nnUNet_results=str(SCRATCH/'results'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    archive = SCRATCH/'Task01_BrainTumour.tar'
    subprocess.run(['curl', '-fL', '--retry', '5', '-C', '-', URL, '-o', str(archive)], check=True)
    digest = hashlib.md5()
    with archive.open('rb') as stream:
        while block := stream.read(8*1024*1024): digest.update(block)
    if digest.hexdigest() != MD5: raise ValueError('Archive checksum mismatch')
    print('Official archive MD5 verified', flush=True)
    with tarfile.open(archive) as source:
        members = [m for m in source.getmembers()
                   if not any(part.startswith('._') for part in Path(m.name).parts)
                   and (m.name.endswith('/dataset.json') or '/imagesTr/' in m.name or '/labelsTr/' in m.name)]
        source.extractall(SCRATCH, members=members, filter='data')
    archive.unlink()
    task = SCRATCH/'Task01_BrainTumour'
    (task/'imagesTs').mkdir(exist_ok=True)
    metadata = json.loads((task/'dataset.json').read_text())
    assert metadata['modality'] == {'0':'FLAIR','1':'T1w','2':'t1gd','3':'T2w'}
    assert metadata['labels'] == {'0':'background','1':'edema','2':'non-enhancing tumor','3':'enhancing tumour'}
    assert len(list((task/'imagesTr').glob('*.nii.gz'))) == 484
    assert len(list((task/'labelsTr').glob('*.nii.gz'))) == 484
    subprocess.run(['nnUNetv2_convert_MSD_dataset', '-i', str(task), '-overwrite_id', '501', '-np', '2'], check=True)
    shutil.rmtree(task)
    subprocess.run(['nnUNetv2_plan_and_preprocess', '-d', '501', '-c', '2d',
                    '-npfp', '2', '-np', '2', '--verify_dataset_integrity'], check=True)
    target = WORK/'preprocessed'/NAME
    # Generate deterministic case-level folds once on CPU, then reuse unchanged.
    from nnunetv2.utilities.crossval_split import generate_crossval_split
    cases = sorted(p.name.removesuffix('.nii.gz') for p in (SCRATCH/'raw'/NAME/'labelsTr').glob('*.nii.gz'))
    folds = generate_crossval_split(cases, seed=12345, n_splits=5)
    (target/'splits_final.json').write_text(json.dumps(folds, indent=2))
    plans = json.loads((target/'nnUNetPlans.json').read_text())
    identifier = plans['configurations']['2d']['data_identifier']
    assert (target/identifier).is_dir()
    report = {'stage':'cpu_preprocessing_complete','nnunet_version':'2.8.1','dataset':NAME,
              'source':URL,'source_md5':MD5,'labeled_cases':len(cases),'configuration':'2d',
              'fold_seed':12345,'train_cases_fold0':len(folds[0]['train']),
              'val_cases_fold0':len(folds[0]['val']),'data_identifier':identifier,
              'license':'CC-BY-SA 4.0','attribution':'Medical Segmentation Decathlon / BraTS 2016-2017 contributors'}
    (WORK/'prepare-report.json').write_text(json.dumps(report, indent=2))
    shutil.rmtree(SCRATCH)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
