#!/bin/bash
# make_opencv_deb.bash — Package the locally built OpenCV (CUDA) into a .deb
#
# Installs to /usr/local so find_package(OpenCV) works without hints
# and LD_LIBRARY_PATH is not needed (ldconfig handles the .so files).
#
# Run from: anywhere — paths are relative to this script
# Output:   limef-opencv-cuda_4.10.0_amd64.deb  (next to this script)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENCV_INSTALL="$SCRIPT_DIR/opencv/install"
VERSION="4.10.0"
PKG_NAME="limef-opencv-cuda"
ARCH="amd64"
DEB_NAME="${PKG_NAME}_${VERSION}_${ARCH}.deb"
STAGING="$SCRIPT_DIR/opencv-deb-staging"

echo "==> Packaging OpenCV $VERSION (CUDA) → $DEB_NAME"
echo "    Source: $OPENCV_INSTALL"

if [[ ! -d "$OPENCV_INSTALL" ]]; then
    echo "ERROR: $OPENCV_INSTALL not found"
    exit 1
fi

# ── Staging tree ──────────────────────────────────────────────────────────────

rm -rf "$STAGING"
mkdir -p "$STAGING/DEBIAN"
mkdir -p "$STAGING/usr/local"

echo "    Copying files..."
cp -rP "$OPENCV_INSTALL/lib"     "$STAGING/usr/local/lib"
cp -rP "$OPENCV_INSTALL/include" "$STAGING/usr/local/include"
# bin/ (opencv_annotation etc.) and share/ are skipped — not needed for library users

# Strip debug symbols from real .so files (not symlinks) to reduce package size
echo "    Stripping debug symbols..."
find "$STAGING/usr/local/lib" -name "*.so.*" -type f \
    | xargs --no-run-if-empty strip --strip-unneeded 2>/dev/null || true

# ── Detect CUDA runtime dependencies ─────────────────────────────────────────
#
# Scan all cuda .so files for NEEDED entries that look like CUDA components
# (npp*, cublas*, cufft*, cudart*, curand*, etc.), then resolve each soname
# to the installed deb package via dpkg -S.

echo "    Detecting CUDA dependencies..."

CUDA_SONAME_PATTERN="lib(npp|cublas|cufft|curand|cusolver|cusparse|cudart|cuda)[^.]*\.so\.[0-9]+"

CUDA_PKGS=()
while IFS= read -r soname; do
    pkg=$(dpkg -S "$soname" 2>/dev/null | head -1 | cut -d: -f1 | tr -d ' ')
    if [[ -n "$pkg" ]]; then
        CUDA_PKGS+=("$pkg")
    else
        echo "    WARNING: no deb found for $soname"
    fi
done < <(
    find "$OPENCV_INSTALL/lib" -name "libopencv_cuda*.so.*" -type f \
        | xargs readelf -d 2>/dev/null \
        | grep -oP "(?<=\[)${CUDA_SONAME_PATTERN}(?=\])" \
        | sort -u
)

# Deduplicate and build comma-separated Depends string
DEPENDS=$(printf '%s\n' "${CUDA_PKGS[@]}" | sort -u | paste -sd, | sed 's/,/, /g')
echo "    Depends: $DEPENDS"

# ── DEBIAN/control ────────────────────────────────────────────────────────────

MAINTAINER="$(git -C "$SCRIPT_DIR" config user.name 2>/dev/null || echo "Limef Project")"
MAINTAINER_EMAIL="$(git -C "$SCRIPT_DIR" config user.email 2>/dev/null || echo "noreply@example.com")"

cat > "$STAGING/DEBIAN/control" <<EOF
Package: $PKG_NAME
Version: $VERSION
Architecture: $ARCH
Maintainer: $MAINTAINER <$MAINTAINER_EMAIL>
Depends: $DEPENDS
Description: OpenCV $VERSION with CUDA support (Limef build)
 OpenCV built from source with CUDA modules: core, imgproc, cudaarithm,
 cudafilters, cudaimgproc, cudawarping, cudev.
 Installed to /usr/local — find_package(OpenCV) works without any hints.
EOF

# ── DEBIAN/postinst & postrm ─────────────────────────────────────────────────

cat > "$STAGING/DEBIAN/postinst" <<'EOF'
#!/bin/bash
set -e
ldconfig
EOF
chmod 0755 "$STAGING/DEBIAN/postinst"

cat > "$STAGING/DEBIAN/postrm" <<'EOF'
#!/bin/bash
set -e
ldconfig
EOF
chmod 0755 "$STAGING/DEBIAN/postrm"

# ── Build ─────────────────────────────────────────────────────────────────────

echo "    Building deb..."
dpkg-deb --build --root-owner-group "$STAGING" "$SCRIPT_DIR/$DEB_NAME"
rm -rf "$STAGING"

DEB_SIZE=$(du -sh "$SCRIPT_DIR/$DEB_NAME" | cut -f1)
echo ""
echo "==> Done: $SCRIPT_DIR/$DEB_NAME  ($DEB_SIZE)"
echo ""
echo "    Install:  sudo dpkg -i $SCRIPT_DIR/$DEB_NAME"
echo "    Remove:   sudo dpkg -r $PKG_NAME"
