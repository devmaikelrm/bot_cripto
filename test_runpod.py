import paramiko
import os
import sys

HOST = "103.196.86.56"
PORT = 10801
USER = "root"
PASS = "Maikel@2026&."

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    print(f"Intentando conectar a {HOST}:{PORT}...")
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
    print("CONECTADO")
    
    stdin, stdout, stderr = client.exec_command("nvidia-smi")
    print("--- GPU ---")
    print(stdout.read().decode())
    
    stdin, stdout, stderr = client.exec_command("ls -la /workspace")
    print("--- /workspace ---")
    print(stdout.read().decode())
    
    client.close()
except Exception as e:
    print(f"ERROR: {e}")
