import argparse
import json
import os
import sys
import urllib.request

# uploads the release files to an existing Zenodo draft and sets its met

API = "https://zenodo.org/api"


def token():
    value = os.environ.get("ZENODO_TOKEN")

    if not value:
        path = os.path.expanduser("~/.zenodo_token")

        if os.path.exists(path):
            value = open(path).read().strip()

    if not value:
        sys.exit("no token, set ZENODO_TOKEN or write it to ~/.zenodo_token")

    return value


def request(method, url, data=None, content_type="application/json"):
    headers = {"Authorization": f"Bearer {token()}"}

    if data is not None:
        headers["Content-Type"] = content_type

    if hasattr(data, "read"):
        headers["Content-Length"] = str(os.fstat(data.fileno()).st_size)

    req = urllib.request.Request(url, data=data, method=method, headers=headers)

    with urllib.request.urlopen(req) as reply:
        body = reply.read()

    return json.loads(body) if body else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("deposition", help="id of the Zenodo draft, the number in the reserved DOI")
    parser.add_argument("folder", help="folder with the archives and SHA256SUMS")
    parser.add_argument("metadata", help="json with the deposit metadata")
    args = parser.parse_args()

    draft = request("GET", f"{API}/deposit/depositions/{args.deposition}")

    if draft.get("submitted"):
        sys.exit("this deposition is already published, make a new version first")

    bucket = draft["links"]["bucket"]

    for name in sorted(os.listdir(args.folder)):
        path = os.path.join(args.folder, name)
        print("uploading", name, os.path.getsize(path), "bytes")

        with open(path, "rb") as fh:
            request("PUT", f"{bucket}/{name}", data=fh, content_type="application/octet-stream")

    metadata = json.load(open(args.metadata))
    metadata = {key: value for key, value in metadata.items() if not key.startswith("_")}

    request("PUT", f"{API}/deposit/depositions/{args.deposition}", data=json.dumps({"metadata": metadata}).encode())
    print("draft updated, review it on zenodo.org and publish it there")


if __name__ == "__main__":
    main()
