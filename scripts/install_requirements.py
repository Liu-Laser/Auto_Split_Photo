#!/usr/bin/env python3
"""安装 auto_split 所需的 Python 依赖。"""
import subprocess
import sys


def install(package):
    print(f"正在安装 {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])


if __name__ == "__main__":
    deps = ["opencv-python", "Pillow", "numpy"]
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
            print(f"  ✓ {dep} 已安装")
        except ImportError:
            install(dep)
    print("\n所有依赖安装完成。")
