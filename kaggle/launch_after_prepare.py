"""Wait for CPU preprocessing, then submit the two-hour pilot exactly once."""
import fcntl
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'artifacts/kaggle-launcher.json'


def command(*args):
    return subprocess.run(['kaggle','-W',*args],cwd=ROOT,text=True,capture_output=True,timeout=180)


def state(status, **extra):
    result={'status':status,'updated_unix':time.time(),**extra}
    STATE.parent.mkdir(exist_ok=True)
    temporary=STATE.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2));temporary.replace(STATE)
    print(json.dumps(result),flush=True)


def main():
    STATE.parent.mkdir(exist_ok=True)
    with (STATE.parent/'kaggle-launcher.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if STATE.exists() and json.loads(STATE.read_text()).get('status')=='gpu_submitted':
            print('GPU job already submitted; refusing duplicate launch',flush=True);return
        deadline=time.monotonic()+4.5*3600
        while time.monotonic()<deadline:
            result=command('kernels','status','hichandut/brain-mri-prepare-cpu')
            text=result.stdout+result.stderr
            if result.returncode:
                state('waiting_api_retry',detail=text[-1000:]);time.sleep(60);continue
            if 'KernelWorkerStatus.COMPLETE' in text:
                break
            if any(token in text for token in ['KernelWorkerStatus.ERROR','KernelWorkerStatus.CANCELLED']):
                state('prepare_failed',detail=text);return
            state('waiting_cpu_preparation',detail=text.strip());time.sleep(60)
        else:
            state('prepare_wait_timeout');return
        # Read quota again immediately before consuming GPU time.
        quota=command('quota','--format','json')
        if quota.returncode:
            state('quota_check_failed',detail=quota.stderr);return
        resources=json.loads(quota.stdout)
        gpu=next(item for item in resources if item['resource']=='GPU')
        if float(gpu['remaining'].removesuffix('h')) < 3:
            state('insufficient_quota',quota=gpu);return
        metadata_path=ROOT/'kaggle/train/kernel-metadata.json'
        metadata=json.loads(metadata_path.read_text())
        result=command('kernels','push','-p','kaggle/train','-t','10800')
        text=result.stdout+result.stderr
        # Some accounts do not expose L4. Fall back only for an explicit
        # hardware-selection rejection, not quota/auth/data failures.
        hardware_rejected = (any(term in text.lower() for term in ['accelerator', 'machine shape'])
                             and any(term in text.lower() for term in ['invalid', 'not available', 'not found', 'not supported', 'unsupported'])
                             and 'quota' not in text.lower())
        if result.returncode and hardware_rejected:
            metadata['machine_shape']='NvidiaTeslaT4'
            metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
            result=command('kernels','push','-p','kaggle/train','-t','10800');text=result.stdout+result.stderr
        if result.returncode:
            state('gpu_submit_failed',detail=text);return
        state('gpu_submitted',kernel=metadata['id'],machine_shape=metadata['machine_shape'],
              quota_before=gpu,training_budget_hours=2,kernel_hard_limit_hours=3,detail=text)


if __name__=='__main__':main()
