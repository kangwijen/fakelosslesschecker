# fakelosslesschecker
This project provides a Python-based tool for analyzing audio files to determine their authenticity based on their most significant frequencies. It leverages signal processing techniques to evaluate whether the audio files are likely to be authentic or fake by analyzing their frequency content.

## Installation
To use this project, you need to install the required Python libraries. You can do this using pip:

```
pip install librosa colorama prettytable tqdm
```

## Usage
1. Place the audio files you want to analyze in a folder.
2. Run the script and input the path to the folder when prompted.
3. The script will process each audio file, analyze its frequency content, and print a table with the results.
