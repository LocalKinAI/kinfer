#!/usr/bin/env python3
"""pypi-wheel.py — a release of kinfer as a wheel for PyPI, so `pip install kinfer-ai`
puts the `kinfer` binary on the PATH, as ruff's and uv's wheels put theirs.

    scripts/pypi-wheel.py dist/kinfer-v0.2.4-darwin-arm64.tar.gz [--name kinfer-ai] [--out dist]

The wheel holds the release's binary as it is, under .data/scripts, which pip installs
into the environment's bin; a small module, `python -m kinfer_ai`, runs the same
binary. Nothing is built here and nothing is downloaded: the tarball is release.sh's.
The name is not kinfer's own because "kinfer" on PyPI belongs to another project.

The binary needs macOS 26 on Apple Silicon (release builds link against SDK 26.2),
which is what the wheel's tag says: macosx_26_0. PyPI refuses a wheel that also says
macosx_16_0, the version a Python built with an older SDK reads macOS 26 as, since no
macOS 16 was released. Only the standard library is used.
"""
import argparse
import base64
import hashlib
import io
import os
import re
import sys
import tarfile
import zipfile

TAGS = ["py3-none-macosx_26_0_arm64"]
REPO = "https://github.com/LocalKinAI/kinfer"


def record_hash(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def absolute_links(readme, tag):
    """PyPI shows the README away from the repository: relative links point into it."""
    def fix(m):
        target = m.group(2)
        if re.match(r"[a-z]+:|#", target):
            return m.group(0)
        return f"{m.group(1)}({REPO}/blob/{tag}/{target})"
    return re.sub(r"(\]\s*)\(([^)\s]+)\)", fix, readme)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tarball", help="dist/kinfer-vX.Y.Z-darwin-arm64.tar.gz, as release.sh made it")
    ap.add_argument("--name", default="kinfer-ai", help="the distribution's name on PyPI (default kinfer-ai)")
    ap.add_argument("--out", default="dist", help="where the wheel goes (default dist)")
    args = ap.parse_args()

    m = re.search(r"kinfer-(v(\d+\.\d+\.\d+))-darwin-arm64\.tar\.gz$", os.path.basename(args.tarball))
    if not m:
        sys.exit(f"pypi-wheel: {args.tarball} is not named kinfer-vX.Y.Z-darwin-arm64.tar.gz")
    tag, version = m.group(1), m.group(2)
    stem = f"kinfer-{tag}-darwin-arm64"
    with tarfile.open(args.tarball) as t:
        binary = t.extractfile(f"{stem}/kinfer").read()
        license_text = t.extractfile(f"{stem}/LICENSE").read()
        readme = t.extractfile(f"{stem}/README.md").read().decode()

    dist = re.sub(r"[-_.]+", "_", args.name).lower()  # the name as wheel filenames and modules spell it
    module = dist
    info = f"{dist}-{version}.dist-info"
    data = f"{dist}-{version}.data"

    summary = ("A single-file local inference runtime for agent fleets on Apple Silicon: "
               "Ollama, OpenAI and Anthropic APIs, many agents at once, models bigger than the machine")
    metadata = "\n".join([
        "Metadata-Version: 2.1",
        f"Name: {args.name}",
        f"Version: {version}",
        f"Summary: {summary}",
        f"Home-page: {REPO}",
        "Author: LocalKinAI",
        "License: Apache-2.0",
        f"Project-URL: Source, {REPO}",
        f"Project-URL: Changelog, {REPO}/blob/main/CHANGELOG.md",
        f"Project-URL: Releases, {REPO}/releases",
        "Keywords: llm,inference,llama.cpp,mlx,ollama,apple-silicon,agents",
        "Classifier: Development Status :: 4 - Beta",
        "Classifier: Environment :: Console",
        "Classifier: Intended Audience :: Developers",
        "Classifier: License :: OSI Approved :: Apache Software License",
        "Classifier: Operating System :: MacOS",
        "Classifier: Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Requires-Python: >=3.8",
        "Description-Content-Type: text/markdown",
        "",
        absolute_links(readme, tag),
    ])
    wheel = "\n".join(["Wheel-Version: 1.0", "Generator: kinfer scripts/pypi-wheel.py", "Root-Is-Purelib: false"]
                      + [f"Tag: {t}" for t in TAGS]) + "\n"
    init = f'''"""kinfer {version}, the binary pip put on the PATH, and where it is."""
import os
import shutil
import sys
import sysconfig

__version__ = "{version}"


def binary():
    """The path of the kinfer binary this package installed."""
    schemes = [sysconfig.get_path("scripts")]
    if sys.version_info >= (3, 10):
        schemes.append(sysconfig.get_path("scripts", sysconfig.get_preferred_scheme("user")))
    for d in schemes:
        p = os.path.join(d, "kinfer")
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    p = shutil.which("kinfer")
    if p:
        return p
    raise FileNotFoundError("the kinfer binary is not where pip installed it; reinstall {args.name}")
'''
    main_py = f'''"""python -m {module}: the kinfer binary, with these arguments."""
import os
import sys

from {module} import binary

if __name__ == "__main__":
    b = binary()
    os.execv(b, [b] + sys.argv[1:])
'''

    files = [  # (path in the wheel, bytes, mode)
        (f"{module}/__init__.py", init.encode(), 0o644),
        (f"{module}/__main__.py", main_py.encode(), 0o644),
        (f"{data}/scripts/kinfer", binary, 0o755),
        (f"{info}/METADATA", metadata.encode(), 0o644),
        (f"{info}/WHEEL", wheel.encode(), 0o644),
        (f"{info}/LICENSE", license_text, 0o644),
    ]
    record = "".join(f"{p},{record_hash(b)},{len(b)}\n" for p, b, _ in files) + f"{info}/RECORD,,\n"
    files.append((f"{info}/RECORD", record.encode(), 0o644))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{dist}-{version}-py3-none-" + ".".join(t.split("-")[2] for t in TAGS) + ".whl")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p, b, mode in files:
            zi = zipfile.ZipInfo(p, date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (0o100000 | mode) << 16
            z.writestr(zi, b)
    print(path)


if __name__ == "__main__":
    main()
