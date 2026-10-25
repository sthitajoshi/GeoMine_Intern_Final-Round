"""Download every dataset the project needs into data/raw/.

Sources (all public, no account):
  * EHU Hyperspectral Remote Sensing Scenes page (Indian Pines, Salinas, Cuprite .mat).
    The live EHU server returns HTTP 403 to scripted clients from some networks, so we fall
    back to the Internet Archive's byte-exact capture of the same official URL (``id_`` mode
    returns the original bytes). SHA-256 checksums below were recorded on first download and
    are verified on every run.
  * USGS Spectral Library v7 (splib07, DOI 10.5066/F7RR1WDJ). The full release is a single
    5.5 GB zip behind a bot-check on ScienceBase. The USGS Spectroscopy Lab also served the
    individual ASCII records under crustal.usgs.gov/speclab/data/ASCIIdata/... ; those are
    fetched (directly, else via the Internet Archive capture of the same USGS URL). Only the
    AVIRIS-Classic-1998-convolved set (splib07b_cvAVIRISc1998) is retrievable this way.
  * 92AV3C.spc: the Purdue/MultiSpec AVIRIS 1992 channel calibration (centre, FWHM) that ships
    with the original 220-band Indian Pines image, mirrored in the spectralpython sample-data repo.

Usage:  python scripts/download_data.py
"""
from __future__ import annotations

import hashlib
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LIB = RAW / "splib07_AV98"

EHU = "https://www.ehu.eus/ccwintco/uploads/"
MAT_FILES = {
    "Indian_pines_corrected.mat": ("6/67/", "ec2f8808710919d566f70f0d4aa885aae1ddfd42b734aba71c5e12ca65450939"),
    "Indian_pines_gt.mat": ("c/c4/", "65c4687a8ab04f6da4789799bc3bc4f6e88bccac3ed6a2e6ae367e5e6b9e429c"),
    "Salinas_corrected.mat": ("a/a3/", "5ec1c0d22f56d18ecd336f8e35735863c0f160682e04e0c18ef3f89a3334d87d"),
    "Salinas_gt.mat": ("f/fa/", "ecfab4d31ef5553f097943235d8ea502038eb4a2067b2ad10b33e37c949955e2"),
    "Cuprite_f970619t01p02_r02_sc03.a.rfl.mat": ("7/7d/", "2c9fa8b09a7587e5ae9680ea0cb5763b214e9042c9950cdbadbdfeee6cff5086"),
}

SPECLAB = "https://crustal.usgs.gov/speclab/data/ASCIIdata/ASCIIdata_splib07b_cvAVIRISc1998/"
LIB_FILES = [
    "s07_AV98_AVIRIS_1998_Wavelengths_(um)_224_chans.txt",
    "ChapterM_Minerals/s07_AV98_Alunite_GDS84_Na03_BECKa_AREF.txt",
    "ChapterM_Minerals/s07_AV98_Alunite_HS295_ASDFRa_AREF.txt",
    "ChapterM_Minerals/s07_AV98_Kaolinite_CM9_BECKb_AREF.txt",
    "ChapterM_Minerals/s07_AV98_Kaolinite_KGa-1_(wxl)_BECKb_AREF.txt",
    "ChapterM_Minerals/s07_AV98_Muscovite_GDS107_BECKa_AREF.txt",
    "ChapterM_Minerals/s07_AV98_Buddingtonite_NHB2301_BECKb_AREF.txt",
    "ChapterM_Minerals/s07_AV98_Calcite_WS272_BECKa_AREF.txt",
    "ChapterM_Minerals/s07_AV98_Chalcedony_CU91-6A_ASDNGa_AREF.txt",
    # distractors used only to build the "unknown / other" synthetic class
    "ChapterM_Minerals/s07_AV98_Montmorillonite_SWy-1_BECKb_AREF.txt",
    "ChapterM_Minerals/s07_AV98_Hematite_GDS27_BECKa_AREF.txt",
    "ChapterM_Minerals/s07_AV98_Jarosite_GDS99_K_200C_Syn_BECKa_AREF.txt",
    "ChapterM_Minerals/s07_AV98_Dickite_NMNH106242_BECKb_AREF.txt",
    "ChapterM_Minerals/s07_AV98_Opal_WS732_BECKa_AREF.txt",
]

# Native-resolution (splib07a) records: one zip per spectrum, each containing the spectrum, its
# instrument wavelength table and bandpass (FWHM) table. Used for our own Gaussian convolution.
SPECLAB_ZIP = "https://crustal.usgs.gov/speclab/data/ZIPrecords/ZIPdata_splib07a/ChapterM_Minerals/"
NATIVE_RECORDS = [
    "Alunite_GDS84_Na03_BECKa_AREF", "Alunite_HS295_ASDFRa_AREF", "Kaolinite_CM9_BECKb_AREF",
    "Kaolinite_KGa-1_(wxl)_BECKb_AREF", "Muscovite_GDS107_BECKa_AREF", "Buddingtonite_NHB2301_BECKb_AREF",
    "Calcite_WS272_BECKa_AREF", "Chalcedony_CU91-6A_ASDNGa_AREF", "Montmorillonite_SWy-1_BECKb_AREF",
    "Hematite_GDS27_BECKa_AREF", "Jarosite_GDS99_K_200C_Syn_BECKa_AREF", "Dickite_NMNH106242_BECKb_AREF",
    "Opal_WS732_BECKa_AREF",
]
NATIVE = RAW / "splib07a"

SPC_URL ="https://raw.githubusercontent.com/spectralpython/sample-data/master/92AV3C.spc"
CUPRITE_MAP_PAGE = "https://www.usgs.gov/media/images/mineral-map-derived-aviris-data-obtained-over-cuprite-nevada"
# "Original" image linked from that page (USGS asset bucket); the page itself has no colour key.
CUPRITE_MAP_IMG = ("https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/"
                   "thumbnails/image/HyperApp.png")


def _get(url: str, timeout: int = 600) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "spectral-earth-mini/1.0 (research download)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status != 200:
            raise IOError(f"HTTP {r.status}")
        return r.read()


_DEAD_HOSTS: set[str] = set()  # live hosts that already refused us: skip straight to the archive


def fetch(url: str, dest: Path, sha256: str | None = None, archive_ts: str = "2020") -> str:
    """Try the original URL (short timeout, once per host), then the Internet Archive raw capture.
    Returns the URL used."""
    if dest.exists() and (sha256 is None or _sha(dest) == sha256):
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    quoted = urllib.parse.quote(url, safe=":/")
    host = urllib.parse.urlparse(url).netloc
    candidates = [] if host in _DEAD_HOSTS else [(quoted, 20, 1)]
    candidates.append((f"https://web.archive.org/web/{archive_ts}id_/{quoted}", 600, 3))
    last = None
    for u, timeout, attempts in candidates:
        for attempt in range(attempts):
            try:
                data = _get(u, timeout)
                if data[:15].lower().startswith(b"<!doctype html") or data[:6].lower() == b"<html>":
                    raise IOError("got an HTML page instead of data")
                dest.write_bytes(data)
                if sha256 and _sha(dest) != sha256:
                    dest.unlink()
                    raise IOError("checksum mismatch")
                return u
            except Exception as e:  # noqa: BLE001 - report and try the next source
                last = f"{u}: {e}"
                if u == quoted:
                    _DEAD_HOSTS.add(host)
                    break
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"all sources failed for {dest.name}; last error {last}")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    log = []
    for name, (sub, sha) in MAT_FILES.items():
        used = fetch(EHU + sub + name, RAW / name, sha)
        log.append(f"{name}\t{used}")
        print("ok", name, used)
    for rel in LIB_FILES:
        used = fetch(SPECLAB + rel, LIB / Path(rel).name)
        log.append(f"{Path(rel).name}\t{used}")
        print("ok", Path(rel).name)
    for rec in NATIVE_RECORDS:
        zpath = NATIVE / "zips" / f"splib07a_{rec}.zip"
        used = fetch(SPECLAB_ZIP + f"splib07a_{rec}.zip", zpath)
        with zipfile.ZipFile(zpath) as z:  # keep only the ASCII tables we need
            for member in z.namelist():
                if member.endswith(".txt") and not member.startswith("errorbars") and "/" not in member:
                    (NATIVE / member).write_bytes(z.read(member))
        log.append(f"splib07a_{rec}.zip\t{used}")
        print("ok", rec)
    used = fetch(SPC_URL, RAW / "92AV3C.spc")
    log.append(f"92AV3C.spc\t{used}")
    print("ok 92AV3C.spc")
    try:
        used = fetch(CUPRITE_MAP_IMG, RAW / "usgs_cuprite_map.png")
        log.append(f"usgs_cuprite_map.png	{used}")
        print("ok usgs_cuprite_map.png")
    except RuntimeError as e:  # optional: only used for the visual side-by-side
        print("WARNING could not fetch the USGS Cuprite map image:", e)
    (RAW / "download_log.tsv").write_text("\n".join(log) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
