import getpass
import os
import sys

import paramiko

# IMPORTANT: Do not hardcode credentials in the repository.
# Configure via env vars (recommended) or you'll be prompted for the password.
HOST = os.environ.get("VPS_HOST", "").strip()
USER = os.environ.get("VPS_USER", "").strip()
PASS = os.environ.get("VPS_PASS")
REMOTE_DIR = os.environ.get("VPS_REMOTE_DIR", "~/bot-cripto").strip()


def safe_print(text: str, end: str = "\n") -> None:
    """Print safely on Windows consoles with cp1252 encoding."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        sys.stdout.write(text + end)
    except UnicodeEncodeError:
        sanitized = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        sys.stdout.write(sanitized + end)


def run_interactive(client: paramiko.SSHClient) -> None:
    safe_print(f"--- Conectado a {HOST} (Sesion Interactiva) ---")
    safe_print("Escribe 'exit' para salir.\n")

    while True:
        try:
            cmd = input(f"{USER}@{HOST}:~$ ")
            if cmd.lower() in ["exit", "quit"]:
                break
            if not cmd.strip():
                continue

            stdin, stdout, stderr = client.exec_command(f"cd {REMOTE_DIR} && {cmd}")
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")

            if out:
                safe_print(out, end="")
            if err:
                safe_print(err, end="")

        except (EOFError, KeyboardInterrupt):
            safe_print("\nSaliendo...")
            break


if __name__ == "__main__":
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        if not HOST or not USER:
            safe_print("Missing VPS_HOST/VPS_USER. Example:")
            safe_print("  $env:VPS_HOST='100.64.x.x'; $env:VPS_USER='ubuntu'; python vps.py")
            raise SystemExit(2)
        if PASS is None:
            PASS = getpass.getpass(f"Password for {USER}@{HOST}: ")

        client.connect(HOST, username=USER, password=PASS)

        if len(sys.argv) >= 4 and sys.argv[1] == "--put":
            local_path = sys.argv[2]
            remote_path = sys.argv[3]
            sftp = client.open_sftp()
            sftp.put(local_path, remote_path)
            sftp.close()
            safe_print(f"PUT_OK {local_path} -> {remote_path}")
            client.close()
            raise SystemExit(0)

        if len(sys.argv) > 1:
            command = " ".join(sys.argv[1:])
            stdin, stdout, stderr = client.exec_command(f"cd {REMOTE_DIR} && {command}")
            safe_print(stdout.read().decode("utf-8", errors="replace"), end="")
            safe_print(stderr.read().decode("utf-8", errors="replace"), end="")
        else:
            run_interactive(client)

        client.close()
    except Exception as exc:
        safe_print(f"Error: {exc}")
