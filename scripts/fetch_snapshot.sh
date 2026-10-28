#!/usr/bin/env bash
# Download the frozen raw-data snapshot (a GitHub Release asset) into data/raw and verify
# its checksum. Afterwards `make data` builds every table from the cache without downloading.
# Usage: scripts/fetch_snapshot.sh [tag]   (default: the snapshot the results were made with)
set -euo pipefail

tag=${1:-data-2026-10-25}
name="wpforecast-raw-${tag#data-}.tar.gz"
url="https://github.com/sneglen/wind-power-forecast/releases/download/${tag}"
dest=data/raw

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fL --retry 3 -o "$tmp/$name" "$url/$name"
curl -fL --retry 3 -o "$tmp/$name.sha256" "$url/$name.sha256"
(cd "$tmp" && sha256sum --check "$name.sha256")

mkdir -p "$dest"
tar -xzf "$tmp/$name" -C "$dest"
echo "Snapshot $tag unpacked into $dest. Next: make data (builds the tables, no download)."
