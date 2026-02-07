import numpy as np
import librosa
import scipy.signal as signal
import colorama
import os
import prettytable as pt
from tqdm import tqdm
import pyloudnorm as pyln

colorama.init()

class TooShortError(Exception):
    pass

class SilentTrackError(Exception):
    pass

def _c(name, text):
    colors = {"red": colorama.Fore.RED, "green": colorama.Fore.GREEN, "yellow": colorama.Fore.YELLOW, "blue": colorama.Fore.BLUE, "cyan": colorama.Fore.CYAN}
    return (colors.get(name, "") + str(text) + colorama.Fore.RESET)

def to_db(x):
    return round(20 * np.log10(x), 2)

def get_dr(filename=None, floats=False, audio=None, sr=None):
    if audio is None or sr is None:
        audio, sr = librosa.load(filename, sr=None)
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)

    channels = audio.shape[0]
    if channels not in (1, 2):
        raise NotImplementedError("We only handle mono or stereo at the moment")

    total_frames = audio.shape[1]
    block_size = sr * 3
    num_blocks = total_frames // block_size

    peaks = [[] for _ in range(channels)]
    rmss = [[] for _ in range(channels)]

    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        block = audio[:, start:end]

        for c in range(channels):
            peak = np.max(np.abs(block[c]))
            rms = np.sqrt(np.mean(np.square(block[c])))
            peaks[c].append(peak)
            rmss[c].append(rms)

    drs = []
    avg_peaks = []
    avg_rmss = []

    for c in range(channels):
        peaks[c].sort()
        rmss[c].sort()
        p2 = peaks[c][-2]
        if p2 == 0:
            raise SilentTrackError
        N = int(0.2 * len(peaks[c]))
        if N == 0:
            raise TooShortError
        r = np.sqrt(np.mean(np.square(rmss[c][-N:])))
        dr = -to_db(r / p2)
        drs.append(dr)
        avg_peaks.append(np.mean(peaks[c]))
        avg_rmss.append(np.mean(rmss[c]))

    if not floats:
        fdr = round(np.mean(drs))
    else:
        fdr = np.mean(drs)

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio.T)

    avg_peak = to_db(np.mean(avg_peaks))
    avg_rms = to_db(np.mean(avg_rmss))
    lufs = round(loudness, 2)

    return fdr, avg_peak, avg_rms, lufs

def detect(audio_path=None, y=None, sr=None):
    if y is None or sr is None:
        y, sr = librosa.load(audio_path, sr=None)
    D = np.abs(librosa.stft(y))
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr)
    smoothed_db = signal.savgol_filter(D_db, window_length=11, polyorder=2, axis=0)

    mean_db = np.mean(smoothed_db)
    std_db = np.std(smoothed_db)
    base_threshold = mean_db + 1.5 * std_db  

    hf_threshold = 18
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

def _verdict_from_freq(file_sample_rate, nyquist_freq, max_significant_freq):
    if max_significant_freq is None:
        return _c("yellow", "Can't determine")
    if file_sample_rate == 48000:
        if max_significant_freq < 20000:
            return _c("red", "Fake")
        if max_significant_freq < nyquist_freq * 0.50:
            return _c("red", "Most likely Fake")
        if max_significant_freq < nyquist_freq * 0.80:
            return _c("yellow", "Might be Fake")
        if max_significant_freq < nyquist_freq * 0.90:
            return _c("yellow", "Might be Authentic")
        if max_significant_freq < nyquist_freq * 0.99:
            return _c("green", "Most likely Authentic")
        return _c("green", "Authentic")
    if file_sample_rate > 48000:
        if max_significant_freq < 22050:
            return _c("red", "Fake")
        if max_significant_freq < nyquist_freq * 0.50:
            return _c("red", "Most likely Fake")
        if max_significant_freq < nyquist_freq * 0.70:
            return _c("yellow", "Might be Fake")
        if max_significant_freq < nyquist_freq * 0.90:
            return _c("yellow", "Might be Authentic")
        if max_significant_freq < nyquist_freq * 0.99:
            return _c("green", "Most likely Authentic")
        return _c("green", "Authentic")
    limit = 22050
    if max_significant_freq < limit * 0.80:
        return _c("red", "Fake")
    if max_significant_freq < limit * 0.85:
        return _c("red", "Most likely Fake")
    if max_significant_freq < limit * 0.90:
        return _c("yellow", "Might be Fake")
    if max_significant_freq < limit * 0.95:
        return _c("yellow", "Might be Authentic")
    if max_significant_freq < limit * 0.99:
        return _c("green", "Most likely Authentic")
    return _c("green", "Authentic")

def _format_metric_dr(val):
    if not isinstance(val, (int, float)):
        return val
    if val < 8:
        return _c("red", val)
    if val < 12:
        return _c("yellow", val)
    return _c("green", val)

def _format_metric_peak(val):
    if not isinstance(val, (int, float)):
        return val
    if val > -2:
        return _c("red", f"{val:.2f} dB")
    if val > -4:
        return _c("yellow", f"{val:.2f} dB")
    return _c("green", f"{val:.2f} dB")

def _format_metric_rms(val):
    if not isinstance(val, (int, float)):
        return val
    if val > -6:
        return _c("red", f"{val:.2f} dB")
    if val > -9:
        return _c("yellow", f"{val:.2f} dB")
    return _c("green", f"{val:.2f} dB")

def _format_metric_lufs(val):
    if not isinstance(val, (int, float)):
        return val
    if val > -6:
        return _c("red", f"{val:.2f} LUFS")
    if val > -9:
        return _c("yellow", f"{val:.2f} LUFS")
    return _c("green", f"{val:.2f} LUFS")

def process_file(audio_file_path):
    audio_file = os.path.basename(audio_file_path)
    try:
        audio, file_sample_rate = librosa.load(audio_file_path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
        y = np.mean(audio, axis=0)
        max_significant_freq = detect(y=y, sr=file_sample_rate)
        nyquist_freq = file_sample_rate / 2
        verdict = _verdict_from_freq(file_sample_rate, nyquist_freq, max_significant_freq)
        if max_significant_freq is None:
            max_significant_freq = _c("yellow", "N/A")
        else:
            max_significant_freq = _c("cyan", str(max_significant_freq))

        try:
            dynamic_range, avg_peak, avg_rms, lufs = get_dr(audio=audio, sr=file_sample_rate)
        except TooShortError:
            dynamic_range = _c("red", "Too Short")
            avg_peak = avg_rms = lufs = _c("yellow", "N/A")
        except SilentTrackError:
            dynamic_range = _c("red", "Silent Track")
            avg_peak = avg_rms = lufs = _c("yellow", "N/A")

        dynamic_range = _format_metric_dr(dynamic_range)
        avg_peak = _format_metric_peak(avg_peak)
        avg_rms = _format_metric_rms(avg_rms)
        lufs = _format_metric_lufs(lufs)
        file_sample_rate_str = _c("cyan", str(file_sample_rate))
        return [audio_file, file_sample_rate_str, max_significant_freq, avg_peak, avg_rms, lufs, dynamic_range, verdict]
    except Exception as e:
        na = _c("yellow", "N/A")
        err = _c("red", str(e)[:40])
        return [audio_file, na, na, na, na, na, na, err]

def main():
    logo = """

$$$$$$$$\ $$\          $$$$$$\  $$\                           $$\                           
$$  _____|$$ |        $$  __$$\ $$ |                          $$ |                          
$$ |      $$ |        $$ /  \__|$$$$$$$\   $$$$$$\   $$$$$$$\ $$ |  $$\  $$$$$$\   $$$$$$\  
$$$$$\    $$ |$$$$$$\ $$ |      $$  __$$\ $$  __$$\ $$  _____|$$ | $$  |$$  __$$\ $$  __$$\ 
$$  __|   $$ |\______|$$ |      $$ |  $$ |$$$$$$$$ |$$ /      $$$$$$  / $$$$$$$$ |$$ |  \__|
$$ |      $$ |        $$ |  $$\ $$ |  $$ |$$   ____|$$ |      $$  _$$<  $$   ____|$$ |      
$$ |      $$$$$$$$\   \$$$$$$  |$$ |  $$ |\$$$$$$$\ \$$$$$$$\ $$ | \$$\ \$$$$$$$\ $$ |      
\__|      \________|   \______/ \__|  \__| \_______| \_______|\__|  \__| \_______|\__|      
                                                                                            

Fake Lossless Checker v1.0 Made by @kangwijen
https://github.com/kangwijen/fakelosslesschecker
    """

    os.system("cls" if os.name == "nt" else "clear")

    print(logo)

    folder_path = str(input("Enter the path to the folder containing the audio files: ")).strip()
    if not folder_path or not os.path.isdir(folder_path):
        print("Error: path does not exist or is not a directory.")
        return
    audio_files = [os.path.join(root, file) for root, dirs, files in os.walk(folder_path) for file in files if file.endswith(('.flac', '.wav'))]
    if not audio_files:
        print("No .flac or .wav files found in that folder.")
        return

    table = pt.PrettyTable()
    table.field_names = ["Audio File", "Sample Rate", "Max Freq", "Avg Peak", "Avg RMS", "LUFS", "DR", "Verdict"]

    for audio_file_path in tqdm(audio_files, desc="Processing Audio Files", unit="file"):
        table.add_row(process_file(audio_file_path))

    table.sortby = "Audio File"
    print(table)

if __name__ == "__main__":
    main()
