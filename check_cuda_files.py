import os

cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"

print("=" * 50)
print("Checking CUDA bin directory")
print(f"Path: {cuda_bin}")
print("=" * 50)

# 检查目录是否存在
if os.path.exists(cuda_bin):
    print("✓ Directory exists")
else:
    print("✗ Directory NOT exists")
    exit()

# 列出关键 DLL 文件
dlls_to_check = [
    "nvrtc.dll",
    "nvrtc64_132_0.dll",
    "nvrtc-builtins64_132.dll",
    "cudart64_132.dll",
    "cublas64_132.dll",
    "cudnn64_8.dll",  # 如果有
]

print("\nChecking required DLLs:")
for dll in dlls_to_check:
    path = os.path.join(cuda_bin, dll)
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"  ✓ {dll} ({size:.0f} KB)")
    else:
        print(f"  ✗ {dll} NOT found")

# 检查 PATH
import subprocess
print("\nChecking PATH:")
result = subprocess.run(['where', 'nvrtc'], capture_output=True, text=True)
if result.returncode == 0:
    print(f"  nvrtc found at: {result.stdout.strip()}")
else:
    print("  nvrtc NOT found in PATH")

print("=" * 50)