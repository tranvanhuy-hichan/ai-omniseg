import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'import_msd', Path(__file__).resolve().parents[1] / 'scripts/import_msd.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_source(tmp_path, folder, dataset_json):
    source = tmp_path / folder
    (source / 'imagesTr').mkdir(parents=True)
    (source / 'labelsTr').mkdir(parents=True)
    (source / 'dataset.json').write_text(json.dumps(dataset_json))
    return source


def test_verify_accepts_matching_liver_metadata(tmp_path):
    task = module.get_task('liver_tumour_ct')
    source = make_source(tmp_path, 'Task03_Liver', {
        'modality': {'0': 'CT'}, 'labels': {'0': 'background', '1': 'liver', '2': 'cancer'},
    })
    metadata = module.verify(source, task)
    assert metadata['labels']['2'] == 'cancer'


def test_verify_rejects_wrong_folder_name(tmp_path):
    task = module.get_task('liver_tumour_ct')
    source = make_source(tmp_path, 'Task01_BrainTumour', {
        'modality': {'0': 'CT'}, 'labels': {'0': 'background', '1': 'liver', '2': 'cancer'},
    })
    with pytest.raises(ValueError, match='Task03_Liver'):
        module.verify(source, task)


def test_verify_rejects_mismatched_labels(tmp_path):
    task = module.get_task('liver_tumour_ct')
    source = make_source(tmp_path, 'Task03_Liver', {
        'modality': {'0': 'CT'}, 'labels': {'0': 'background', '1': 'liver', '2': 'tumor'},
    })
    with pytest.raises(ValueError, match='do not guess or remap'):
        module.verify(source, task)


def test_verify_rejects_unknown_task(tmp_path):
    source = make_source(tmp_path, 'Task99_Unknown', {'modality': {}, 'labels': {}})
    with pytest.raises(ValueError, match='No known MSD source fingerprint'):
        module.verify(source, {'id': 'not_a_real_task'})
