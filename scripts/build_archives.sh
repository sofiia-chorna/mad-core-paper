#!/bin/bash
# packs the release from the current commit into dist/<version>
# usage: bash scripts/build_archives.sh 1.0.0
root=$(cd "$(dirname "$0")/.." && pwd)
version=${1:?version}
name=mad-core-paper-$version
out=$root/dist/$version
mkdir -p $out

git -C $root archive --prefix=$name/ HEAD | gzip -n > $out/$name-src.tar.gz
git -C $root rev-parse HEAD > $out/$name-src.commit
tar -C $root --transform "s|^results_archive|$name-results|" -czf $out/$name-results.tar.gz results_archive
tar -C $root --transform "s|^models_archive|$name-models|" -czf $out/$name-models.tar.gz models_archive
cp $root/README.md $out

cd $out
sha256sum *.tar.gz README.md > SHA256SUMS
cat SHA256SUMS
