# fakelosslesschecker
This project provides a Python-based tool for analyzing audio files to determine their authenticity based on their most significant frequencies. It leverages signal processing techniques to evaluate whether the audio files are likely to be authentic or fake by analyzing their frequency content.

There are 3 versions of this application:
1. Base CLI - fake_lossless_checker.py
2. Multithreaded CLI - fake_lossless_checker-mt.py
3. Multithreaded GUI (WIP)

If you're going to run this tool on Android, please use the base CLI as Android's Python doesn't support threading.

## Installation
### Running Executable
#### On Windows
Download the executable in [releases](https://github.com/kangwijen/fakelosslesschecker/releases) then run.

### Running Python File
#### On Computer
To use this project on a computer, you need to install the required Python libraries. You can do this using pip:

```
pip install librosa colorama prettytable tqdm pyloudnorm
```

#### On Android
To use this project on a computer, you need to install [Termux](https://github.com/termux/termux-app). Then you need to run these commands to install Python and the required libraries:

```
pkg update && pkg upgrade -y 

pkg install git python python-pip build-essential tur-repo ninja patchelf python-scipy python-numpy libsndfile -y

pip install colorama tqdm prettytable wheel setuptools cython==3.0.10 meson-python build scikit-build-core setuptools-scm

export LDFLAGS="-lpython3.11" 

pip install scikit-learn --no-build-isolation

pip install pyloudnorm

pkg install libsoxr

pip install soxr

pkg install llvm-14 -y

LLVM_CONFIG=/data/data/com.termux/files/usr/opt/libllvm-14/bin/llvm-config pip install llvmlite

pip install librosa --no-build-isolation
```

## Usage
1. Place the audio files you want to analyze in a folder.
2. Run the script and input the path to the folder when prompted.
3. The script will process each audio file, analyze its frequency content and dynamic range, and print a table with the results.
