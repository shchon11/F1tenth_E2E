#!/bin/bash
# usage: get.sh <name> <image-url> <source-page>
# downloads to refs/raw/<name>.<ext>, writes 1400px-wide copy to refs/<name>.jpg, logs to refs/sources.tsv
R=/home/shchon11/F1tenth_E2E/docs/pm/sim-design-2026-11-28/round2/refs
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
name="$1"; url="$2"; page="$3"
ext="${url##*.}"; ext="${ext%%\?*}"; case "$ext" in jpg|jpeg|png|webp) ;; *) ext=jpg;; esac
code=$(curl -sL -m 60 -A "$UA" -o "$R/raw/$name.$ext" -w "%{http_code}" "$url")
if [ "$code" != "200" ]; then echo "FAIL $code $name $url"; rm -f "$R/raw/$name.$ext"; printf "%s\t%s\t%s\tFAIL %s\n" "$name" "$url" "$page" "$code" >> "$R/sources.tsv"; exit 1; fi
python3 -I - "$R/raw/$name.$ext" "$R/$name.jpg" <<'PY'
import sys
from PIL import Image
im=Image.open(sys.argv[1]).convert('RGB')
w,h=im.size
if w>1400: im=im.resize((1400,round(h*1400/w)),Image.LANCZOS)
im.save(sys.argv[2],quality=88)
print(sys.argv[2].split('/')[-1],(w,h),'->',im.size)
PY
printf "%s\t%s\t%s\tOK\n" "$name" "$url" "$page" >> "$R/sources.tsv"
