import argparse
import json
import socket
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['pilot', 'first', 'finish'])
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    root = Path('${DEV_ROOT}/phace-atomic')
    logs = root / 'matcloud-upload/mad-cosmo-v1/logs'
    if args.status:
        record = json.loads((logs / (args.mode + '-process.json')).read_text())
        print(json.dumps(record))
        if record['host'] != socket.gethostname():
            raise SystemExit('Status check is on a different host; no process conclusion')
        subprocess.run(['ps', '-p', str(record['pid']), '--ppid', str(record['pid']),
                        '-o', 'pid,ppid,stat,etime,time,wchan:28,args'], check=False)
        child_path = Path(f'/proc/{record["pid"]}/task/{record["pid"]}/children')
        if child_path.exists():
            for child in child_path.read_text().split():
                process_path = Path('/proc') / child
                print((process_path / 'io').read_text())
                for descriptor in (process_path / 'fd').iterdir():
                    try:
                        target = str(descriptor.readlink())
                    except FileNotFoundError:
                        continue
                    if target.startswith('${STORE_ROOT}/mad-core/release/') and target.endswith(('na.npy', 'source_reference_offsets.npy')):
                        print(target)
        subprocess.run(['tail', '-n', '8', record['log']], check=False)
        return
    if args.mode == 'finish':
        raise SystemExit('Use finish.py to start the completion workflow')
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f'{args.mode}-{int(time.time())}.log'
    with log.open('xb') as stream:
        process = subprocess.Popen(['bash', str(Path(__file__).with_name('submit.sh')), args.mode],
                                   cwd=root, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    status = dict(pid=process.pid, host=socket.gethostname(), log=str(log), mode=args.mode)
    (logs / (args.mode + '-process.json')).write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status))


if __name__ == '__main__':
    main()
