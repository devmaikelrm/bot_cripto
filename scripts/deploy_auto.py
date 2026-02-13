import getpass
import os
import tarfile
import paramiko
import sys
from io import BytesIO

# Configuración (no hardcode secrets in repo)
HOST = os.environ.get("VPS_HOST", "").strip()
USER = os.environ.get("VPS_USER", "").strip()
PASS = os.environ.get("VPS_PASS")
REMOTE_DIR = os.environ.get("VPS_REMOTE_DIR", "").strip()
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

def main():
    try:
        # 1. Empaquetar
        tar_data = create_tarball()
        print(f"📦 Tamaño del paquete: {len(tar_data.getvalue()) / 1024 / 1024:.2f} MB")

        if not HOST or not USER:
            print("❌ Missing VPS_HOST/VPS_USER env vars.")
            print("   PowerShell example:")
            print("   $env:VPS_HOST='100.64.x.x'; $env:VPS_USER='ubuntu'; python scripts/deploy_auto.py")
            raise SystemExit(2)
        password = PASS if PASS is not None else getpass.getpass(f"Password for {USER}@{HOST}: ")
        remote_dir = REMOTE_DIR or f"/home/{USER}/bot-cripto"

        # 2. Conectar SSH
        print(f"🔌 Conectando a {HOST}...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(HOST, username=USER, password=password)
        
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
