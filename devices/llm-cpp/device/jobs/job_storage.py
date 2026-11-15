"""Device job: STORAGE.

Manufactures the llm-cpp device's own data volume and measures real storage
throughput. The volume is a dedicated file-backed store owned by the device,
not the workspace tree. Measurements are real timed I/O.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import mb_per_s, now, record, write_json

DEVICE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VOLUME = os.path.join(DEVICE_ROOT, "storage", "volume.img")
SIZE_MB = int(os.environ.get("VOLUME_MB", "64"))
BLOCK = 1024 * 1024
FILL = b"\xa5" * BLOCK


def write_volume():
    t0 = now()
    written = 0
    with open(VOLUME, "wb") as f:
        for _ in range(SIZE_MB):
            f.write(FILL)
            written += BLOCK
        f.flush()
        os.fsync(f.fileno())
    return written, now() - t0


def read_volume():
    t0 = now()
    read = 0
    with open(VOLUME, "rb") as f:
        while True:
            chunk = f.read(BLOCK)
            if not chunk:
                break
            read += len(chunk)
    return read, now() - t0


def fsync_probe():
    t0 = now()
    path = os.path.join(DEVICE_ROOT, "storage", ".fsync-probe")
    with open(path, "wb") as f:
        for _ in range(16):
            f.write(FILL)
            f.flush()
            os.fsync(f.fileno())
    dur = now() - t0
    os.remove(path)
    return 16 * BLOCK, dur


def main():
    os.makedirs(os.path.dirname(VOLUME), exist_ok=True)
    wbytes, wsec = write_volume()
    rbytes, rsec = read_volume()
    fbytes, fsec = fsync_probe()

    result = {
        "volume": VOLUME,
        "volume_mb": SIZE_MB,
        "write_mb_s": mb_per_s(wbytes, wsec),
        "read_mb_s": mb_per_s(rbytes, rsec),
        "fsync_write_mb_s": mb_per_s(fbytes, fsec),
        "write_seconds": round(wsec, 3),
        "read_seconds": round(rsec, 3),
        "fsync_seconds": round(fsec, 3),
        "backing": "file-backed block store on host filesystem",
        "isolation": "dedicated device volume path; not workspace tree",
    }
    write_json(os.path.join(DEVICE_ROOT, "metrics", "storage.json"), result)
    record("storage", "manufactured", result)
    print("[storage] own volume:", result["volume"])
    print(f"[storage] write {result['write_mb_s']} MB/s | read {result['read_mb_s']} MB/s | fsync {result['fsync_write_mb_s']} MB/s")


if __name__ == "__main__":
    main()
