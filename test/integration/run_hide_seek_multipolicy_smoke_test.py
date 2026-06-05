#!/usr/bin/env python3
"""Multi-policy wire smoke test: launches the multi-policy train scene with --multi-policy and asserts
the env_info handshake carries agent_policy_names == ["seeker", "hider"] (two distinct policies),
n_agents == 2, and the step loop runs. Raw sockets (no trainer); modeled on run_hide_seek_smoke_test.py."""
import json
import os
import socket
import subprocess
import sys

HOST, PORT = "127.0.0.1", 11008
SCENE = "res://examples/hide_and_seek/hide_and_seek_multipolicy_train.tscn"
GODOT = os.environ.get("GODOT", "godot")
OBS_SIZE = 15


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
            [GODOT, "--headless", "--path", ".", SCENE, "--multi-policy", "action_repeat=1", "speedup=1"]
        )
        conn, _ = server.accept()
        conn.settimeout(30)

        send(conn, {"type": "handshake", "major_version": "0", "minor_version": "7"})
        send(conn, {"type": "env_info"})
        info = recv(conn)
        if info.get("n_agents") != 2:
            failures.append("n_agents != 2 (got %r)" % info.get("n_agents"))
        names = info.get("agent_policy_names")
        if names != ["seeker", "hider"]:
            failures.append("agent_policy_names != ['seeker','hider'] (got %r)" % names)

        send(conn, {"type": "reset"})
        msg = recv(conn)
        if len(msg.get("obs") or []) != 2:
            failures.append("reset obs count != 2 (got %r)" % msg.get("obs"))

        for _ in range(3):
            send(conn, {"type": "action", "action": [{"move": 4}, {"move": 3}]})
            step = recv(conn)
            if step.get("type") != "step":
                failures.append("step type (got %r)" % step.get("type"))
                break

        send(conn, {"type": "close"})
    finally:
        if proc is not None:
            try:
                rc = proc.wait(timeout=15)
            except Exception:
                proc.kill()
                rc = -1
            if rc != 0:
                failures.append("godot exited with code %d" % rc)
        if conn is not None:
            conn.close()
        server.close()

    if failures:
        print("MULTI-POLICY SMOKE TEST FAILED:", failures)
        sys.exit(1)
    print("MULTI-POLICY SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
