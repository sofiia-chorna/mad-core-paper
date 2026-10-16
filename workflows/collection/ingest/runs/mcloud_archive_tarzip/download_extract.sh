#!/bin/bash

# downloads the archive files of each record and keeps the structure files
records=$1
work=${WORK}/mcloud_archive

for id in $(cut -f1 $records | sort -u); do

    mkdir -p $work/raw/$id $work/extract/$id

    grep -P "^$id\t" $records | while IFS=$'\t' read -r record size url name; do
        curl -sL $url -o $work/raw/$id/$name
    done

    for f in $work/raw/$id/*; do
        case ${f,,} in
            *.zip)
                unzip -qoC $f '*.db' '*.xyz' '*.extxyz' '*.cif' '*.traj' '*POSCAR*' '*CONTCAR*' '*XDATCAR*' -d $work/extract/$id
                ;;
            *.tar*|*.tgz)
                tar -xaf $f -C $work/extract/$id --wildcards --no-anchored '*.db' '*.xyz' '*.extxyz' '*.cif' '*.traj' '*POSCAR*' '*CONTCAR*' '*XDATCAR*'
                ;;
        esac
    done

    rm -r $work/raw/$id
done
