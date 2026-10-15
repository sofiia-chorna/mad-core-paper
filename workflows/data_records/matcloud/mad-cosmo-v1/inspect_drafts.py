import json
import urllib.error
import urllib.request
from pathlib import Path


def get_json(url, token):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    token_path = Path('${HOME}/.mcloud_token')
    if token_path.stat().st_mode & 0o077:
        raise SystemExit('Token file permissions are not private')
    token = token_path.read_text().strip()
    base = 'https://archive.materialscloud.org'
    records = get_json(base + '/api/user/records?size=100', token)
    print(json.dumps({'total': records.get('hits', {}).get('total')}, indent=2))
    for record in records.get('hits', {}).get('hits', []):
        summary = {key: record.get(key) for key in ('id', 'status', 'is_published')}
        summary['title'] = record.get('metadata', {}).get('title')
        if not record.get('is_published'):
            try:
                draft = get_json(base + '/api/records/' + record['id'] + '/draft', token)
                summary['files'] = {key: value for key, value in draft.get('files', {}).items() if key != 'entries'}
                summary['quota'] = {key: value for key, value in draft.items() if 'quota' in key}
                files = get_json(base + '/api/records/' + record['id'] + '/draft/files', token)
                summary['file_listing'] = {key: value for key, value in files.items() if key != 'entries'}
                summary['file_count'] = len(files.get('entries', []))
                summary['file_bytes'] = sum(entry.get('size', 0) or 0 for entry in files.get('entries', []))
            except urllib.error.HTTPError as error:
                summary['read_error'] = error.code
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
