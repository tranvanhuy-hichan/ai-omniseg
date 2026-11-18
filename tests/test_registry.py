import json

from omniseg import registry
from omniseg.core import inference
from omniseg.core.volume import color_table


def test_label_entries_use_display_names_and_fall_back_to_label_names():
    brain = registry.get_task('brain_tumour_mri')
    assert [(i, n) for i, n, _ in registry.label_entries(brain)] == [
        (1, 'Phù nề'), (2, 'U không tăng tương phản'), (3, 'U tăng tương phản')]
    assert registry.label_entries(brain)[0][2] == (245, 180, 69)
    liver = registry.get_task('liver_tumour_ct')
    assert [(i, n) for i, n, _ in registry.label_entries(liver)] == [(1, 'Gan'), (2, 'U gan')]
    assert registry.label_entries(liver)[0][2] != registry.label_entries(liver)[1][2]
    bare = {'labels': {'background': 0, 'non_enhancing_tumor': 2, 'edema': 1}}
    entries = registry.label_entries(bare)
    assert [(i, n) for i, n, _ in entries] == [(1, 'Edema'), (2, 'Non enhancing tumor')]
    assert entries[0][2] == registry.FALLBACK_COLORS[0]


def test_color_table_is_indexed_by_label_id():
    table = color_table([(1, 'a', (10, 20, 30)), (3, 'c', (1, 2, 3))])
    assert table.shape == (4, 3) and tuple(table[1]) == (10, 20, 30) and tuple(table[2]) == (0, 0, 0)


def test_checkpoint_path_and_existence_follow_nnunet_results(tmp_path, monkeypatch):
    task = registry.get_task('liver_tumour_ct')
    monkeypatch.setenv('nnUNet_results', str(tmp_path))
    path = inference.checkpoint_path(task)
    assert path == tmp_path / 'Dataset502_Liver' / 'nnUNetTrainer_100epochs__nnUNetPlans__3d_lowres' / 'fold_0' / 'checkpoint_final.pth'
    assert not inference.checkpoint_exists(task)
    path.parent.mkdir(parents=True); path.write_bytes(b'x')
    assert inference.checkpoint_exists(task)


def test_every_task_json_labels_have_matching_display_entries():
    for task in registry.list_tasks():
        foreground = {v for v in task['labels'].values() if v != 0}
        assert {i for i, _, _ in registry.label_entries(task)} == foreground
        assert set(task.get('label_display', {})) <= {str(v) for v in foreground}
