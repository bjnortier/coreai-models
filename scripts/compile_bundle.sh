#!/usr/bin/env bash
#
# Pre-compile every .aimodel in an exported model bundle, per platform, and
# package one downloadable bundle per platform + architecture.
#
# Usage:
#   scripts/compile_bundle.sh [BUNDLE_DIR] [OUTPUT_DIR]
#
#   BUNDLE_DIR  exported bundle containing metadata.json and *.aimodel
#               (default: exports/parakeet-tdt-0.6b-v3_float16_static)
#   OUTPUT_DIR  where to write results (default: <BUNDLE_DIR>_compiled)
#
# Environment:
#   PLATFORMS               space-separated platforms (default: "iOS macOS")
#   MIN_DEPLOYMENT_VERSION  passed to coreai-build (default: 27.0)
#   PREFERRED_COMPUTE       passed to coreai-build: gpu, neural-engine or none
#                           (default: gpu)
#
# Output layout:
#   OUTPUT_DIR/compiled.iOS/<model>.<arch>.aimodelc     raw compiler output,
#   OUTPUT_DIR/compiled.macOS/<model>.<arch>.aimodelc   deleted once zipped
#   OUTPUT_DIR/<bundle>.<platform>.<arch>/               client bundle
#       metadata.json, processor/, ...                   copied from BUNDLE_DIR
#       <model>.aimodelc                                 compiled for <arch>
#   OUTPUT_DIR/<bundle>.<platform>.<arch>.zip            what the client downloads
#
# In each client bundle metadata.json's "assets" point at the .aimodelc names.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE_DIR="${1:-$REPO_ROOT/exports/parakeet-tdt-0.6b-v3_float16_static}"
BUNDLE_DIR="$(cd "$BUNDLE_DIR" && pwd)"
BUNDLE_NAME="$(basename "$BUNDLE_DIR")"
OUTPUT_DIR="${2:-${BUNDLE_DIR}_compiled}"
PLATFORMS="${PLATFORMS:-iOS macOS}"
MIN_DEPLOYMENT_VERSION="${MIN_DEPLOYMENT_VERSION:-27.0}"
PREFERRED_COMPUTE="${PREFERRED_COMPUTE:-gpu}"

if [[ ! -f "$BUNDLE_DIR/metadata.json" ]]; then
    echo "error: $BUNDLE_DIR/metadata.json not found" >&2
    exit 1
fi

models=("$BUNDLE_DIR"/*.aimodel)
if [[ ! -e "${models[0]}" ]]; then
    echo "error: no .aimodel found in $BUNDLE_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

for platform in $PLATFORMS; do
    compiled_dir="$OUTPUT_DIR/compiled.$platform"
    rm -rf "$compiled_dir"
    mkdir -p "$compiled_dir"

    # 1. Compile each model for every architecture the platform supports.
    for model in "${models[@]}"; do
        echo "==> Compiling $(basename "$model") for $platform"
        xcrun coreai-build compile "$model" \
            --platform "$platform" \
            --min-deployment-version "$MIN_DEPLOYMENT_VERSION" \
            --preferred-compute "$PREFERRED_COMPUTE" \
            --output "$compiled_dir"
    done

    # 2. Collect the architectures produced (<model>.<arch>.aimodelc).
    archs=$(find "$compiled_dir" -maxdepth 1 -name '*.aimodelc' \
        | sed -E 's/.*\.([^.]+)\.aimodelc$/\1/' | sort -u)

    # 3. Assemble and zip one client bundle per architecture.
    for arch in $archs; do
        name="$BUNDLE_NAME.$platform.$arch"
        dest="$OUTPUT_DIR/$name"
        echo "==> Assembling $name"
        rm -rf "$dest" "$dest.zip"
        mkdir -p "$dest"

        # Everything except the source models (metadata, processor, ...).
        rsync -a --exclude '*.aimodel' --exclude '.DS_Store' "$BUNDLE_DIR/" "$dest/"

        for model in "${models[@]}"; do
            base="$(basename "$model" .aimodel)"
            src="$compiled_dir/$base.$arch.aimodelc"
            if [[ ! -d "$src" ]]; then
                echo "error: $base was not compiled for $platform/$arch" >&2
                exit 1
            fi
            # Move rather than copy so the raw output doesn't sit on disk twice.
            mv "$src" "$dest/$base.aimodelc"
        done

        # Point the metadata's assets at the compiled models.
        /usr/bin/python3 - "$dest/metadata.json" <<'EOF'
import json, sys
path = sys.argv[1]
with open(path) as f:
    meta = json.load(f)
meta["assets"] = {
    k: v + "c" if v.endswith(".aimodel") else v
    for k, v in meta.get("assets", {}).items()
}
with open(path, "w") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
EOF

        # Zip without AppleDouble (._*) or .DS_Store files, and check for them.
        find "$dest" \( -name '._*' -o -name '.DS_Store' \) -delete
        (cd "$OUTPUT_DIR" && ditto -c -k --norsrc --noextattr --keepParent "$name" "$name.zip")
        if zipinfo -1 "$dest.zip" | grep -qE '(^|/)(\._|\.DS_Store$)'; then
            echo "error: $name.zip contains ._ or .DS_Store entries" >&2
            exit 1
        fi
    done

    # Every compiled model has been moved into a zipped bundle.
    rm -rf "$compiled_dir"
done

echo
echo "Done. Client bundles in $OUTPUT_DIR:"
ls -1 "$OUTPUT_DIR"/*.zip
