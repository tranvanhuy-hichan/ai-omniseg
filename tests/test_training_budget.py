import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('budget', Path(__file__).resolve().parents[1]/'kaggle/train_budget.py')
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def test_resume_checkpoint_keeps_next_epoch_without_skipping(tmp_path):
    class Trainer:
        current_epoch = 17
        def save_checkpoint(self, path):
            self.saved_epoch = self.current_epoch + 1  # actual nnU-Net convention
    trainer = Trainer()
    budget.save_for_next_epoch(trainer, tmp_path/'checkpoint.pth')
    assert trainer.saved_epoch == 17
    assert trainer.current_epoch == 17


def test_checkpoint_failure_does_not_change_epoch(tmp_path):
    class Trainer:
        current_epoch = 17
        def save_checkpoint(self, path): raise OSError('disk full')
    trainer = Trainer()
    with pytest.raises(OSError): budget.save_for_next_epoch(trainer,tmp_path/'checkpoint.pth')
    assert trainer.current_epoch == 17


def test_benchmark_excludes_first_two_warmup_epochs():
    result=budget.estimate([300,200,60,60,60],5,100)
    assert result['seconds_per_epoch_median'] == 60
    assert result['estimated_remaining_hours'] == pytest.approx(95/60)
