import getpass
import os
import tarfile
import paramiko
import sys
from io import BytesIO
from shlex import quote

# Configuración (no hardcode secrets in repo)
HOST = os.environ.get("VPS_HOST", "").strip()
USER = os.environ.get("VPS_USER", "").strip()
PASS = os.environ.get("VPS_PASS")
REMOTE_DIR = os.environ.get("VPS_REMOTE_DIR", "").strip()
GIT_REPO = os.environ.get("VPS_GIT_REPO", "https://git.kl3d.uy/maikelrm95/Crypto.git").strip()
GIT_BRANCH = os.environ.get("VPS_GIT_BRANCH", "main").strip()
STATE_DIR = os.environ.get("VPS_STATE_DIR", "").strip()
LOCAL_EXCLUDES = {
    '.venv', '__pycache__', '.git', '.mypy_cache', 
    '.pytest_cache', '.ruff_cache', 'data', 'models', 'logs',
    'bot_cripto_deploy.tar.gz'
}

def create_tarball():
    print("📦 Empaquetando proyecto...")
    bio = BytesIO()
    with tarfile.open(fileobj=bio, mode='w:gz') as tar:
        for root, dirs, files in os.walk('.'):
            # Excluir carpetas de la raíz solamente
            if root == '.':
                dirs[:] = [d for d in dirs if d not in LOCAL_EXCLUDES]
            
            # Excluir carpetas ocultas o caches en cualquier nivel
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file not in LOCAL_EXCLUDES:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, '.')
                    if rel_path.startswith(".."): continue 
                    tar.add(full_path, arcname=rel_path)
    bio.seek(0)
    return bio

def run_command(ssh, command, description):
    print(f"🚀 Ejecutando: {description}...")
    stdin, stdout, stderr = ssh.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    
    if exit_status != 0:
        print(f"❌ Error en {description}:")
        print(err)
        return False, out + "\n" + err
    
    if out: print(f"📄 Output:\n{out}")
    return True, out

def deploy_git_swap(ssh, remote_dir: str, state_dir: str, password: str):
    # This implements the "release + swap" flow:
    # - stop user timers/services
    # - move existing release to bot-cripto.old.<ts>
    # - keep data/models/logs in a separate state dir and symlink into the new release
    # - git clone new code
    # - copy .env from old release
    # - create venv + pip install
    # - install user systemd units + restart
    old_dir = f"{remote_dir}.old.$(date -u +%Y%m%dT%H%M%SZ)"
    remote_dir_q = quote(remote_dir)
    state_dir_q = quote(state_dir)
    repo_q = quote(GIT_REPO)
    branch_q = quote(GIT_BRANCH)

    cmds = [
        "set -euo pipefail",
        "cd ~",
        "systemctl --user stop bot-cripto-inference.timer bot-cripto-retrain.timer bot-cripto-telegram-control.service || true",
        "systemctl --user stop bot-cripto-inference.service bot-cripto-retrain.service || true",
        f"if [ -d {remote_dir_q} ]; then mv {remote_dir_q} {quote(old_dir)}; fi",
        f"mkdir -p {state_dir_q}",
        f"if [ -d {quote(old_dir)}/data ]; then mv {quote(old_dir)}/data {state_dir_q}/ 2>/dev/null || true; fi",
        f"if [ -d {quote(old_dir)}/models ]; then mv {quote(old_dir)}/models {state_dir_q}/ 2>/dev/null || true; fi",
        f"if [ -d {quote(old_dir)}/logs ]; then mv {quote(old_dir)}/logs {state_dir_q}/ 2>/dev/null || true; fi",
        f"git clone --branch {branch_q} {repo_q} {remote_dir_q}",
        f"if [ -f {quote(old_dir)}/.env ]; then cp {quote(old_dir)}/.env {remote_dir_q}/.env; fi",
        f"ln -sfn {state_dir_q}/data {remote_dir_q}/data",
        f"ln -sfn {state_dir_q}/models {remote_dir_q}/models",
        f"ln -sfn {state_dir_q}/logs {remote_dir_q}/logs",
        f"cd {remote_dir_q}",
        "python3.11 -m venv .venv",
        ".venv/bin/pip install -U pip setuptools wheel",
        ".venv/bin/pip install -e '.[dev]'",
        "bash systemd/install_user_systemd.sh",
        "systemctl --user daemon-reload",
        "systemctl --user start bot-cripto-telegram-control.service",
        "systemctl --user start bot-cripto-inference.timer bot-cripto-retrain.timer",
        "systemctl --user list-timers | grep bot-cripto || true",
        "systemctl --user status bot-cripto-inference.timer --no-pager | head -n 25 || true",
        "systemctl --user status bot-cripto-telegram-control.service --no-pager | head -n 40 || true",
    ]

    full_cmd = " && ".join(cmds)
    return run_command(ssh, full_cmd, "Deploy git-swap (release + swap)")

def main():
    try:
        if not HOST or not USER:
            print("❌ Missing VPS_HOST/VPS_USER env vars.")
            print("   PowerShell example:")
            print("   $env:VPS_HOST='100.64.x.x'; $env:VPS_USER='ubuntu'; python scripts/deploy_auto.py")
            raise SystemExit(2)
        password = PASS if PASS is not None else getpass.getpass(f"Password for {USER}@{HOST}: ")
        remote_dir = REMOTE_DIR or f"/home/{USER}/bot-cripto"
        state_dir = STATE_DIR or f"/home/{USER}/bot-cripto-state"

        # 2. Conectar SSH
        print(f"🔌 Conectando a {HOST}...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(HOST, username=USER, password=password)

        if "--git-swap" in sys.argv:
            success, msg = deploy_git_swap(ssh, remote_dir=remote_dir, state_dir=state_dir, password=password)
            if success:
                print("\n✅ Deploy git-swap completado.")
                print(f"📂 Release: {remote_dir}")
                print(f"🗄️  Estado:  {state_dir} (data/models/logs)")
            else:
                print("\n⚠️ Deploy git-swap terminó con errores.")
            ssh.close()
            return

        # 1. Empaquetar (modo tarball legacy)
        tar_data = create_tarball()
        print(f"📦 Tamaño del paquete: {len(tar_data.getvalue()) / 1024 / 1024:.2f} MB")

        # 3. Preparar directorio remoto
        run_command(ssh, f"mkdir -p {remote_dir}", "Crear directorio remoto")
        
        # 4. Subir archivo (SFTP)
        print("📤 Subiendo archivos...")
        sftp = ssh.open_sftp()
        sftp.chdir(remote_dir)
        with sftp.file("deploy.tar.gz", "wb") as f:
            f.write(tar_data.getvalue())
        sftp.close()
        
        # 5. Descomprimir y Setup
        print("🔧 Configurando en servidor (Python 3.11 + venv)...")
        
        setup_cmds = [
            f"cd {remote_dir}",
            "rm -rf .venv",
            "tar -xzf deploy.tar.gz",
            "rm deploy.tar.gz",
            "find . -name '*.sh' -exec sed -i 's/\\r$//' {} +",
            "chmod +x scripts/*.sh",
            # Crear venv con 3.11
            "python3.11 -m venv .venv",
            # Instalar dependencias
            ".venv/bin/pip install --upgrade pip setuptools wheel",
            ".venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
            ".venv/bin/pip install -e '.[dev]'"
        ]
        
        full_cmd = " && ".join(setup_cmds)
        success, msg = run_command(ssh, full_cmd, "Instalación remota con Python 3.11")
        
        if success:
            print("\n✅ ¡Despliegue completado con éxito con Python 3.11!")
            print(f"📂 Ubicación: {remote_dir}")
            print(f"🚀 Para ejecutar el bot: {remote_dir}/.venv/bin/bot-cripto --help")
        else:
            print("\n⚠️ El despliegue terminó con errores.")

        ssh.close()

    except Exception as e:
        print(f"\n❌ Error crítico: {str(e)}")

if __name__ == "__main__":
    main()
