import numpy as np
import librosa
import scipy.signal as signal
import colorama
import os
import prettytable as pt
from tqdm import tqdm

def detect(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    
    D = np.abs(librosa.stft(y))
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr)
    smoothed_db = signal.savgol_filter(D_db, window_length=11, polyorder=2, axis=0)

    mean_db = np.mean(smoothed_db)
    std_db = np.std(smoothed_db)
    base_threshold = mean_db + 1.5 * std_db  

    hf_threshold = 20
    db_thresholds = base_threshold - (hf_threshold * (freqs / np.max(freqs)))

    significant_freqs = []
    for i, freq in enumerate(freqs):
        max_freq_index = np.where(smoothed_db[i, :] > db_thresholds[i])
        if max_freq_index[0].size > 0:
            significant_freqs.append(freq)

    if len(significant_freqs) > 0:
        max_significant_freq = np.ceil(significant_freqs[-1])
    else:
        max_significant_freq = None

    return max_significant_freq

table = pt.PrettyTable()
table.field_names = ["Audio File", "Max Significant Frequency (Hz)", "Sample Rate (Hz)", "Verdict"]

folder_path = str(input("Enter the path to the folder containing the audio files: "))
audio_files = [os.path.join(root, file) for root, dirs, files in os.walk(folder_path) for file in files if file.endswith(('.flac', '.wav'))]

for audio_file_path in tqdm(audio_files, desc="Processing Audio Files", unit="file"):
    audio_file = os.path.basename(audio_file_path)
    max_significant_freq = detect(audio_file_path)

    file_sample_rate = librosa.get_samplerate(audio_file_path)
    nyquist_freq = file_sample_rate / 2

    if max_significant_freq is not None:
        if file_sample_rate > 44100:
            if max_significant_freq < 22050:
                verdict = colorama.Fore.RED + "Fake" + colorama.Style.RESET_ALL
            elif max_significant_freq >= 22050 and max_significant_freq < (nyquist_freq * 0.80):
                verdict = colorama.Fore.RED + "Most likely Fake" + colorama.Style.RESET_ALL
            elif max_significant_freq >= (nyquist_freq * 0.80) and max_significant_freq < (nyquist_freq * 0.95):
                verdict = colorama.Fore.YELLOW + "Might be Authentic" + colorama.Style.RESET_ALL
            elif max_significant_freq >= (nyquist_freq * 0.95) and max_significant_freq < (nyquist_freq * 0.99):
                verdict = colorama.Fore.GREEN + "Most likely Authentic" + colorama.Style.RESET_ALL
            elif max_significant_freq >= (nyquist_freq * 0.99):
                verdict = colorama.Fore.GREEN + "Authentic" + colorama.Style.RESET_ALL
            else:
                verdict = colorama.Fore.BLUE + "Can't Determine" + colorama.Style.RESET_ALL
        else:
            if max_significant_freq < 19000:
                verdict = colorama.Fore.RED + "Fake" + colorama.Style.RESET_ALL
            elif max_significant_freq >= 19000 and max_significant_freq < 19500:
                verdict = colorama.Fore.RED + "Most likely Fake" + colorama.Style.RESET_ALL
            elif max_significant_freq >= 19500 and max_significant_freq < 21000:
                verdict = colorama.Fore.YELLOW + "Might be Authentic" + colorama.Style.RESET_ALL
            elif max_significant_freq >= 21000 and max_significant_freq < 22000:
                verdict = colorama.Fore.GREEN + "Most likely Authentic" + colorama.Style.RESET_ALL
            elif max_significant_freq >= 22000:
                verdict = colorama.Fore.GREEN + "Authentic" + colorama.Style.RESET_ALL
            else:
                verdict = colorama.Fore.BLUE + "Can't Determine" + colorama.Style.RESET_ALL
    else:
        verdict = colorama.Fore.YELLOW + "Can't determine" + colorama.Style.RESET_ALL
        max_significant_freq = colorama.Fore.YELLOW + "N/A" + colorama.Style.RESET_ALL

    max_significant_freq_str = colorama.Fore.CYAN + str(max_significant_freq) + colorama.Style.RESET_ALL if max_significant_freq != "N/A" else max_significant_freq
    file_sample_rate_str = colorama.Fore.CYAN + str(file_sample_rate) + colorama.Style.RESET_ALL

    table.add_row([audio_file, max_significant_freq_str, file_sample_rate_str, verdict])

print(table)
