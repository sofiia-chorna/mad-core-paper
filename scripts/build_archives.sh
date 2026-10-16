#!/bin/bash
# packs the release from the current commit into dist/ version

cd "$(dirname "$0")/.."
version=$1
name=mad-core-paper-$version
out=dist/$version

mkdir -p $out

git archive --prefix=$name/ HEAD | gzip -n > $out/$name-src.tar.gz
git rev-parse HEAD > $out/$name-src.commit

tar --transform "s|^results_archive|$name-results|" -czf $out/$name-results.tar.gz results_archive
tar --transform "s|^models_archive|$name-models|" -czf $out/$name-models.tar.gz models_archive

cp README.md $out

cd $out
sha256sum *.tar.gz *.md > SHA256SUMS
cat SHA256SUMS
