"""Build reviewable, self-contained Kaggle notebooks from the checked-in scripts."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = 'hichandut'


def _existing_dataset_sources(folder):
    """Preserve a hand-added resume-checkpoint dataset_sources entry across rebuilds."""
    path = HERE/folder/'kernel-metadata.json'
    if path.is_file():
        return json.loads(path.read_text()).get('dataset_sources', [])
    return []


def notebook(cells):
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
            'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code.splitlines(keepends=True)} for code in cells]}


def build():
    install = "import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'nnunetv2==2.8.1'], check=True)\n"
    prepare = (HERE/'prepare_cpu.py').read_text()
    (HERE/'prepare/prepare.ipynb').write_text(json.dumps(notebook([install,prepare]),indent=1))
    train_source = (HERE/'train_budget.py').read_text()
    stage = '''import os, json, shutil
from pathlib import Path
work = Path('/kaggle/working')
inputs = list(Path('/kaggle/input').rglob('prepare-report.json'))
if len(inputs) != 1:
    raise RuntimeError(f'Expected exactly one completed CPU preparation output, found {inputs}')
source = inputs[0].parent
report = json.loads(inputs[0].read_text())
if report['stage'] != 'cpu_preprocessing_complete' or report['labeled_cases'] != 484:
    raise RuntimeError('CPU preparation is incomplete')
shutil.copytree(source/'preprocessed', work/'preprocessed')
shutil.copy2(inputs[0], work/'prepare-report.json')
os.environ.update(nnUNet_raw=str(work/'raw'), nnUNet_preprocessed=str(work/'preprocessed'),
                  nnUNet_results=str(work/'results'), nnUNet_compile='false', nnUNet_n_proc_DA='2')
(work/'raw').mkdir(exist_ok=True)
'''
    launch = f"from pathlib import Path\nPath('/kaggle/working/train_budget.py').write_text({train_source!r})\n"
    launch += "subprocess.run([sys.executable, '/kaggle/working/train_budget.py', '--hours', '2', '--fold', '0'], check=True)\n"
    launch += "shutil.rmtree('/kaggle/working/preprocessed')\nprint(Path('/kaggle/working/train-report.json').read_text())\n"
    (HERE/'train/train.ipynb').write_text(json.dumps(notebook([install,stage,launch]),indent=1))
    for folder,slug,gpu in [('prepare','brain-mri-prepare-cpu',False),('train','brain-mri-2d-100ep',True)]:
        metadata={'id':f'{OWNER}/{slug}','title':slug.replace('-',' ').title(),
                  'code_file':'prepare.ipynb' if folder=='prepare' else 'train.ipynb',
                  'language':'python','kernel_type':'notebook','is_private':True,
                  'enable_gpu':gpu,'enable_internet':True,'dataset_sources':_existing_dataset_sources(folder),
                  'kernel_sources':[f'{OWNER}/brain-mri-prepare-cpu'] if gpu else [],'competition_sources':[]}
        if gpu: metadata['machine_shape']='NvidiaL4'
        (HERE/folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


def build_liver():
    install = "import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'nnunetv2==2.8.1'], check=True)\n"
    prepare = (HERE/'liver_prepare_cpu.py').read_text()
    (HERE/'liver_prepare/prepare.ipynb').write_text(json.dumps(notebook([install,prepare]),indent=1))
    train_source = (HERE/'liver_train_budget.py').read_text()
    stage = '''import os, json, shutil
from pathlib import Path
work = Path('/kaggle/working')
inputs = list(Path('/kaggle/input').rglob('prepare-report.json'))
if len(inputs) != 1:
    raise RuntimeError(f'Expected exactly one completed CPU preparation output, found {inputs}')
source = inputs[0].parent
report = json.loads(inputs[0].read_text())
if report['stage'] != 'cpu_preprocessing_complete' or report['labeled_cases'] != 131:
    raise RuntimeError('CPU preparation is incomplete')
shutil.copytree(source/'preprocessed', work/'preprocessed')
shutil.copy2(inputs[0], work/'prepare-report.json')
os.environ.update(nnUNet_raw=str(work/'raw'), nnUNet_preprocessed=str(work/'preprocessed'),
                  nnUNet_results=str(work/'results'), nnUNet_compile='false', nnUNet_n_proc_DA='2')
(work/'raw').mkdir(exist_ok=True)
'''
    launch = f"from pathlib import Path\nPath('/kaggle/working/train_budget.py').write_text({train_source!r})\n"
    launch += "subprocess.run([sys.executable, '/kaggle/working/train_budget.py', '--hours', '2', '--fold', '0'], check=True)\n"
    launch += "shutil.rmtree('/kaggle/working/preprocessed')\nprint(Path('/kaggle/working/train-report.json').read_text())\n"
    (HERE/'liver_train/train.ipynb').write_text(json.dumps(notebook([install,stage,launch]),indent=1))
    for folder,slug,gpu in [('liver_prepare','liver-ct-prepare-cpu',False),('liver_train','liver-ct-3dlowres-100ep',True)]:
        metadata={'id':f'{OWNER}/{slug}','title':slug.replace('-',' ').title(),
                  'code_file':'prepare.ipynb' if folder=='liver_prepare' else 'train.ipynb',
                  'language':'python','kernel_type':'notebook','is_private':True,
                  'enable_gpu':gpu,'enable_internet':True,'dataset_sources':_existing_dataset_sources(folder),
                  'kernel_sources':[f'{OWNER}/liver-ct-prepare-cpu'] if gpu else [],'competition_sources':[]}
        if gpu: metadata['machine_shape']='NvidiaL4'
        (HERE/folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__ == '__main__':
    build()
    build_liver()
