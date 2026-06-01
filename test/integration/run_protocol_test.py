#!/usr/bin/env python3
"""Drives NcnnSync through the godot_rl protocol and asserts message shapes."""
import json
import os
import socket
import subprocess
import sys

HOST, PORT = "127.0.0.1", 11008
SCENE = "res://test/integration/protocol_test_scene.tscn"
GODOT = os.environ.get("GODOT", "godot")


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

        # Handshake + env_info exchange.
        send(conn, {"type": "handshake", "major_version": "0", "minor_version": "7"})
        send(conn, {"type": "env_info"})
        info = recv(conn)
        if info.get("type") != "env_info":
            failures.append("env_info type")
        if "observation_space" not in info:
            failures.append("missing observation_space")
        if "action_space" not in info:
            failures.append("missing action_space")
        if info.get("n_agents") != 1:
            failures.append("n_agents != 1 (got %r)" % info.get("n_agents"))
        obs_space = info.get("observation_space")
        # observation_space may be a dict (single agent) or list of dicts.
        agent_space = obs_space[0] if isinstance(obs_space, list) else obs_space
        cam = (agent_space or {}).get("camera_2d")
        if cam != {"space": "box", "size": [2, 2, 3]}:
            failures.append("camera_2d obs_space wrong (got %r)" % cam)

        # Reset -> expect reset reply with obs.
        send(conn, {"type": "reset"})
        msg = recv(conn)
        if msg.get("type") != "reset":
            failures.append("reset reply type (got %r)" % msg.get("type"))
        obs = msg.get("obs") or []
        if not obs:
            failures.append("reset missing obs")
        elif len(obs[0]["obs"]) != 5:
            failures.append("obs size != 5 (got %d)" % len(obs[0]["obs"]))
        # reset obs carries the camera_2d image too (same _get_obs_from_agents path as step).
        if obs:
            reset_cam = obs[0].get("camera_2d")
            if not isinstance(reset_cam, str) or len(bytes.fromhex(reset_cam)) != 2 * 2 * 3:
                failures.append("reset missing valid camera_2d hex (got %r)" % reset_cam)

        # Action -> expect step reply.
        send(conn, {"type": "action", "action": [{"move": 2}]})
        step = recv(conn)
        if step.get("type") != "step":
            failures.append("step type (got %r)" % step.get("type"))
        for k in ("obs", "reward", "done"):
            if k not in step:
                failures.append("step missing %s" % k)
        if len(step.get("reward", [])) != 1:
            failures.append("reward len != 1")
        if len(step.get("done", [])) != 1:
            failures.append("done len != 1")
        info = step.get("info")
        if not isinstance(info, list) or len(info) != 1:
            failures.append("info not a 1-element list (got %r)" % info)
        elif info[0].get("is_success") is not True:
            failures.append("info[0] missing is_success=true (got %r)" % info[0])
        if len(step.get("obs") or []) != 1:
            failures.append("step obs count != 1")
        elif len(step["obs"][0].get("obs") or []) != 5:
            failures.append("step obs size != 5")
        cam_hex = (step["obs"][0] if step.get("obs") else {}).get("camera_2d")
        if not isinstance(cam_hex, str):
            failures.append("step missing camera_2d hex (got %r)" % cam_hex)
        else:
            cam_bytes = bytes.fromhex(cam_hex)
            if len(cam_bytes) != 2 * 2 * 3:
                failures.append("camera_2d byte count != 12 (got %d)" % len(cam_bytes))
            elif any(cam_bytes[i] != (255 if i % 3 == 0 else 0) for i in range(len(cam_bytes))):
                failures.append("camera_2d bytes not all-red")

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
        print("PROTOCOL TEST FAILED:", failures)
        sys.exit(1)
    print("PROTOCOL TEST PASSED")


if __name__ == "__main__":
    main()
