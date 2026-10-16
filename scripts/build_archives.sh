#!/bin/bash
# packs the release from the current commit into dist/<version>
# usage  bash scripts/build_archives.sh 1.0.0

cd "$(dirname "$0")/.."
version=${1:?give a version, for example 1.0.0}
name=mad-core-paper-$version
out=dist/$version

mkdir -p $out

git archive --prefix=$name/ HEAD | gzip -n > $out/$name-src.tar.gz
git rev-parse HEAD > $out/$name-src.commit

tar --transform "s|^results_archive|$name-results|" -czf $out/$name-results.tar.gz results_archive
tar --transform "s|^models_archive|$name-models|" -czf $out/$name-models.tar.gz models_archive

cp README.md docs/reproduction.md docs/verification.md docs/zenodo.json $out

cd $out
sha256sum *.tar.gz *.md > SHA256SUMS
cat SHA256SUMS
