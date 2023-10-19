import subprocess 
import sys


def build_gelu():
    compile_command = [
        "c++",
        "-O3", "-Wall", "-shared", "-std=c++11", "-fPIC",
        f"`{sys.executable} -m pybind11 --includes`",
        "gelu_binding.cpp",
        "-o", "gelu_module`python3-config --extension-suffix`",
        "-L/usr/local/cuda/lib64",
        "-lcudart"
    ]
    subprocess.run(compile_command)


if __name__ == "__main__":
    build_gelu()

    