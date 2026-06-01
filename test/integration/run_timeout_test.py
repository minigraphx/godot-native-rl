#!/usr/bin/env python3
"""Verifies NcnnSync's read timeout: after handshake the server goes silent and the
Godot client must quit cleanly (rc 0) within the timeout instead of hanging forever."""
import json
import os
import socket
import subprocess
import sys
import time

HOST, PORT = "127.0.0.1", 11008
SCENE = "res://test/integration/protocol_test_scene.tscn"
GODOT = os.environ.get("GODOT", "godot")
READ_TIMEOUT_S = 2


def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("socket closed early")
        buf += chunk
    return buf


def send(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(len(data).to_bytes(4, "little") + data)


def recv(sock):
    n = int.from_bytes(recvall(sock, 4), "little")
    return json.loads(recvall(sock, n).decode("utf-8"))


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    server.settimeout(30)

    proc = None
    conn = None
    failures = []
    try:
        proc = subprocess.Popen(
            [GODOT, "--headless", "--path", ".", SCENE,
             "action_repeat=1", "speedup=1", "read_timeout=%d" % READ_TIMEOUT_S]
        )
        conn, _ = server.accept()
        conn.settimeout(30)

        # Handshake + env_info, then reset — normal startup.
        send(conn, {"type": "handshake", "major_version": "0", "minor_version": "7"})
        send(conn, {"type": "env_info"})
        recv(conn)  # env_info reply
        send(conn, {"type": "reset"})
        recv(conn)  # reset reply with obs

        # Now go silent: send nothing. The client is waiting for an action and must
        # hit the read timeout and quit cleanly.
        started = time.time()
        rc = proc.wait(timeout=READ_TIMEOUT_S + 15)
        elapsed = time.time() - started
        if rc != 0:
            failures.append("godot exited with code %d (expected clean 0)" % rc)
        if elapsed > READ_TIMEOUT_S + 10:
            failures.append("client took %.1fs to quit (timeout was %ds)" % (elapsed, READ_TIMEOUT_S))
    except subprocess.TimeoutExpired:
        failures.append("client did NOT quit — read timeout failed to fire (hung)")
        if proc is not None:
            proc.kill()
    finally:
        if conn is not None:
            conn.close()
        server.close()

    if failures:
        print("TIMEOUT TEST FAILED:", failures)
        sys.exit(1)
    print("TIMEOUT TEST PASSED")


if __name__ == "__main__":
    main()
