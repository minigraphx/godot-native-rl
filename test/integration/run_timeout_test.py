#!/usr/bin/env python3
"""Verifies NcnnSync's read timeout quits the Godot client cleanly (rc 0) instead of
hanging forever, in two phases where a silent socket can strand it:
  1. post-reset — the trainer goes silent while the client waits for an action;
  2. startup    — the trainer accepts the connection but never sends the handshake
     (regression guard for the null-return path through _handshake/_send_env_info)."""
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


def _launch(server):
    """Start the headless client, accept its connection, return (proc, conn)."""
    proc = subprocess.Popen(
        [GODOT, "--headless", "--path", ".", SCENE,
         "action_repeat=1", "speedup=1", "read_timeout=%d" % READ_TIMEOUT_S]
    )
    conn, _ = server.accept()
    conn.settimeout(30)
    return proc, conn


def _assert_clean_exit(proc, label, failures):
    """Wait for the client to self-terminate; record any failure."""
    started = time.time()
    try:
        rc = proc.wait(timeout=READ_TIMEOUT_S + 15)
    except subprocess.TimeoutExpired:
        failures.append("%s: client did NOT quit — read timeout failed to fire (hung)" % label)
        proc.kill()
        return
    elapsed = time.time() - started
    if rc != 0:
        failures.append("%s: godot exited with code %d (expected clean 0)" % (label, rc))
    if elapsed > READ_TIMEOUT_S + 10:
        failures.append("%s: client took %.1fs to quit (timeout was %ds)" % (label, elapsed, READ_TIMEOUT_S))


def scenario_silent_after_reset(server, failures):
    """Normal startup, then the trainer goes silent waiting for the next action."""
    proc, conn = _launch(server)
    try:
        send(conn, {"type": "handshake", "major_version": "0", "minor_version": "7"})
        send(conn, {"type": "env_info"})
        recv(conn)  # env_info reply
        send(conn, {"type": "reset"})
        recv(conn)  # reset reply with obs
        # Go silent: the client waits for an action and must hit the read timeout.
        _assert_clean_exit(proc, "after-reset", failures)
    finally:
        conn.close()


def scenario_silent_at_startup(server, failures):
    """The trainer accepts but never sends the handshake — the client must time out in
    _handshake() and quit cleanly rather than crashing on a null message."""
    proc, conn = _launch(server)
    try:
        # Send nothing at all.
        _assert_clean_exit(proc, "startup", failures)
    finally:
        conn.close()


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    server.settimeout(30)

    failures = []
    try:
        scenario_silent_after_reset(server, failures)
        scenario_silent_at_startup(server, failures)
    finally:
        server.close()

    if failures:
        print("TIMEOUT TEST FAILED:", failures)
        sys.exit(1)
    print("TIMEOUT TEST PASSED")


if __name__ == "__main__":
    main()
