"""Device job: NETWORK.

Gives the llm-cpp device its own network identity and measures real TCP
throughput to it. The device owns a dedicated port and serves a real TCP
endpoint; throughput is measured by moving actual bytes over a real socket.
"""
import hashlib
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import mb_per_s, now, record, write_json

DEVICE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOST = "127.0.0.1"
PORT = int(os.environ.get("DEVICE_NET_PORT", "18090"))
PAYLOAD_MB = int(os.environ.get("NET_MB", "256"))
CHUNK = 1024 * 1024


def server(ready, received):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    ready.set()
    conn, _ = srv.accept()
    digest = hashlib.sha256()
    total = 0
    with conn:
        while True:
            data = conn.recv(CHUNK)
            if not data:
                break
            digest.update(data)
            total += len(data)
        conn.sendall(b"OK")
    received["bytes"] = total
    received["sha256"] = digest.hexdigest()
    srv.close()


def client(payload):
    t0 = now()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    digest = hashlib.sha256()
    sent = 0
    with s:
        for _ in range(PAYLOAD_MB):
            s.sendall(payload)
            digest.update(payload)
            sent += len(payload)
        s.shutdown(socket.SHUT_WR)
        s.recv(16)
    return sent, now() - t0, digest.hexdigest()


def main():
    payload = os.urandom(CHUNK)
    ready = threading.Event()
    received = {}
    t = threading.Thread(target=server, args=(ready, received), daemon=True)
    t.start()
    ready.wait(10)
    sent, seconds, sent_hash = client(payload)
    t.join(30)

    integrity = received.get("sha256") == sent_hash
    result = {
        "device_endpoint": f"{HOST}:{PORT}",
        "bytes_sent": sent,
        "throughput_mb_s": mb_per_s(sent, seconds),
        "throughput_mbit_s": round(mb_per_s(sent, seconds) * 8, 2),
        "seconds": round(seconds, 3),
        "integrity_sha256_match": integrity,
        "interface": "loopback TCP (device-owned port)",
        "note": "measured real byte transfer; host link speed not assumed",
    }
    write_json(os.path.join(DEVICE_ROOT, "metrics", "network.json"), result)
    record("network", "manufactured", result)
    print(f"[network] device endpoint {result['device_endpoint']}")
    print(f"[network] {result['throughput_mb_s']} MB/s ({result['throughput_mbit_s']} Mbit/s), integrity={integrity}")


if __name__ == "__main__":
    main()
