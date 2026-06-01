#!/usr/bin/env python3
"""Self-play smoke test: drives the hide & seek training scene through the godot_rl protocol and
asserts the parameter-sharing loop — n_agents == 2, one shared obs/action space, both agents step,
clean exit. Raw sockets (no SB3), modeled on run_protocol_test.py."""
import json
import os
import socket
import subprocess
import sys

HOST, PORT = "127.0.0.1", 11008
SCENE = "res://examples/hide_and_seek/hide_and_seek_train.tscn"
GODOT = os.environ.get("GODOT", "godot")
OBS_SIZE = 15  # 2 own pos + 8 wall rays + 4 opponent + 1 role flag


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
            [GODOT, "--headless", "--path", ".", SCENE, "action_repeat=1", "speedup=1"]
        )
        conn, _ = server.accept()
        conn.settimeout(30)

        send(conn, {"type": "handshake", "major_version": "0", "minor_version": "7"})
        send(conn, {"type": "env_info"})
        info = recv(conn)
        if info.get("type") != "env_info":
            failures.append("env_info type")
        if info.get("n_agents") != 2:
            failures.append("n_agents != 2 (got %r)" % info.get("n_agents"))
        action_space = info.get("action_space")
        ag_action = action_space[0] if isinstance(action_space, list) else action_space
        move = (ag_action or {}).get("move")
        if not move or move.get("size") != 5 or move.get("action_type") != "discrete":
            failures.append("action_space move wrong (got %r)" % move)

        # Reset -> two agents' obs, each OBS_SIZE long.
        send(conn, {"type": "reset"})
        msg = recv(conn)
        if msg.get("type") != "reset":
            failures.append("reset reply type (got %r)" % msg.get("type"))
        obs = msg.get("obs") or []
        if len(obs) != 2:
            failures.append("reset obs count != 2 (got %d)" % len(obs))
        elif any(len(o.get("obs", [])) != OBS_SIZE for o in obs):
            failures.append("reset obs size != %d (got %r)" % (OBS_SIZE, [len(o.get("obs", [])) for o in obs]))

        # A few steps with both agents' actions; expect 2-element reward/done/obs each time.
        for _ in range(5):
            send(conn, {"type": "action", "action": [{"move": 4}, {"move": 3}]})
            step = recv(conn)
            if step.get("type") != "step":
                failures.append("step type (got %r)" % step.get("type"))
                break
            if len(step.get("reward", [])) != 2:
                failures.append("reward len != 2 (got %r)" % step.get("reward"))
            if len(step.get("done", [])) != 2:
                failures.append("done len != 2 (got %r)" % step.get("done"))
            if len(step.get("obs", [])) != 2:
                failures.append("step obs count != 2 (got %r)" % step.get("obs"))

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
        print("HIDE&SEEK SMOKE TEST FAILED:", failures)
        sys.exit(1)
    print("HIDE&SEEK SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
