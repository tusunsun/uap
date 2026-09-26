#!/usr/bin/env bash
# Rebuild the full (unsubsetted) CJK fonts from npm webfont packages.
# The repo ships fonts subset to the characters the film uses; run this if you
# change on-screen text and need characters that are not in the subset.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
pip install -q fonttools brotli
(cd "$TMP" && npm pack noto-serif-sc noto-sans-sc lxgw-wenkai-webfont >/dev/null && for f in *.tgz; do tar xzf "$f" --one-top-level="${f%.tgz}"; done)
merge() {  # merge <glob> <out.ttf>
python3 - "$1" "$2" <<'PY'
import glob, os, sys
from fontTools.ttLib import TTFont
from fontTools.merge import Merger, Options
files = sorted(glob.glob(sys.argv[1]))
tmp = []
for i, f in enumerate(files):
    t = TTFont(f); t.flavor = None
    p = f"/tmp/_sub_{os.getpid()}_{i}.ttf"; t.save(p); tmp.append(p)
m = Merger(options=Options(drop_tables=['vmtx', 'vhea', 'MATH', 'GSUB', 'GPOS', 'GDEF']))
m.merge(tmp).save(sys.argv[2])
for p in tmp: os.remove(p)
print("built", sys.argv[2])
PY
}
S="$TMP/noto-serif-sc-"*/package; N="$TMP/noto-sans-sc-"*/package; K="$TMP/lxgw-wenkai-webfont-"*/package/files
merge "$(echo $S)/noto_serif_sc_black/*.woff2" assets/fonts/NotoSerifSC-Black.ttf
merge "$(echo $S)/noto_serif_sc_bold/*.woff2" assets/fonts/NotoSerifSC-Bold.ttf
merge "$(echo $S)/noto_serif_sc_light/*.woff2" assets/fonts/NotoSerifSC-Light.ttf
merge "$(echo $N)/noto_sans_sc_black/*.woff2" assets/fonts/NotoSansSC-Black.ttf
merge "$(echo $N)/noto_sans_sc_bold/*.woff2" assets/fonts/NotoSansSC-Bold.ttf
merge "$(echo $N)/noto_sans_sc_medium/*.woff2" assets/fonts/NotoSansSC-Medium.ttf
merge "$(echo $K)/lxgwwenkai-regular-subset-*.woff2" assets/fonts/LXGWWenKai-Regular.ttf
merge "$(echo $K)/lxgwwenkai-bold-subset-*.woff2" assets/fonts/LXGWWenKai-Bold.ttf
rm -rf "$TMP"
