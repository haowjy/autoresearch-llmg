#!/usr/bin/env bash
# Install ripgrep (rg) for P0-TW-03 agent shell + harness_rg. No sudo required.
set -euo pipefail
VERSION="${RIPGREP_VERSION:-15.1.0}"
ARCH="${RIPGREP_ARCH:-x86_64-unknown-linux-musl}"
DEST="${HOME}/.local/bin"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$DEST"
TARBALL="ripgrep-${VERSION}-${ARCH}.tar.gz"
curl -fsSL -o "$TMP/$TARBALL" \
  "https://github.com/BurntSushi/ripgrep/releases/download/${VERSION}/${TARBALL}"
tar -xzf "$TMP/$TARBALL" -C "$TMP"
cp "$TMP/ripgrep-${VERSION}-${ARCH}/rg" "$DEST/rg"
chmod +x "$DEST/rg"
echo "Installed: $("$DEST/rg" --version | head -1) -> $DEST/rg"
echo "Ensure PATH includes: export PATH=\"\$HOME/.local/bin:\$PATH\""
