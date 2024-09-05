import numpy as np
import librosa
import scipy.signal as signal
import colorama
import os
import prettytable as pt
from tqdm import tqdm
import soundfile as sf
import pyloudnorm as pyln
from concurrent.futures import ThreadPoolExecutor, as_completed

class TooShortError(Exception):
    pass

class SilentTrackError(Exception):
    pass

def to_db(x):
    return round(20 * np.log10(x), 2)

def get_dr(filename, floats=False):
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

def detect(audio_path):
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

def process_file(audio_file_path):
    audio_file = os.path.basename(audio_file_path)
    
    max_significant_freq = detect(audio_file_path)
    file_sample_rate = librosa.get_samplerate(audio_file_path)
    nyquist_freq = file_sample_rate / 2

    if max_significant_freq is not None:
        if file_sample_rate == 48000:
            if max_significant_freq < 20000:
                verdict = colorama.Fore.RED + "Fake" + colorama.Fore.RESET
            elif max_significant_freq >= 20000 and max_significant_freq < (nyquist_freq * 0.50):
                verdict = colorama.Fore.RED + "Most likely Fake" + colorama.Fore.RESET
            elif max_significant_freq >= (nyquist_freq * 0.50) and max_significant_freq < (nyquist_freq * 0.80):
                verdict = colorama.Fore.YELLOW + "Might be Fake" + colorama.Fore.RESET
            elif max_significant_freq >= (nyquist_freq * 0.80) and max_significant_freq < (nyquist_freq * 0.90):
                verdict = colorama.Fore.YELLOW + "Might be Authentic" + colorama.Fore.RESET
            elif max_significant_freq >= (nyquist_freq * 0.90) and max_significant_freq < (nyquist_freq * 0.99):
                verdict = colorama.Fore.GREEN + "Most likely Authentic" + colorama.Fore.RESET
            elif max_significant_freq >= (nyquist_freq * 0.99):
                verdict = colorama.Fore.GREEN + "Authentic" + colorama.Fore.RESET
            else:
                verdict = colorama.Fore.BLUE + "Can't Determine" + colorama.Fore.RESET
        elif file_sample_rate > 48000:
            if max_significant_freq < 22050:
                verdict = colorama.Fore.RED + "Fake" + colorama.Fore.RESET
            elif max_significant_freq >= 22050 and max_significant_freq < (nyquist_freq * 0.50):
                verdict = colorama.Fore.RED + "Most likely Fake" + colorama.Fore.RESET
            elif max_significant_freq >= (nyquist_freq * 0.50) and max_significant_freq < (nyquist_freq * 0.70):
                verdict = colorama.Fore.YELLOW + "Might be Fake" + colorama.Fore.RESET
            elif max_significant_freq >= (nyquist_freq * 0.70) and max_significant_freq < (nyquist_freq * 0.90):
                verdict = colorama.Fore.YELLOW + "Might be Authentic" + colorama.Fore.RESET
            elif max_significant_freq >= (nyquist_freq * 0.90) and max_significant_freq < (nyquist_freq * 0.99):
                verdict = colorama.Fore.GREEN + "Most likely Authentic" + colorama.Fore.RESET
            elif max_significant_freq >= (nyquist_freq * 0.99):
                verdict = colorama.Fore.GREEN + "Authentic" + colorama.Fore.RESET
            else:
                verdict = colorama.Fore.BLUE + "Can't Determine" + colorama.Fore.RESET
        else:
            if max_significant_freq < (22050 * 0.80):
                verdict = colorama.Fore.RED + "Fake" + colorama.Fore.RESET
            elif max_significant_freq >= (22050 * 0.80) and max_significant_freq < (22050 * 0.85):
                verdict = colorama.Fore.RED + "Most likely Fake" + colorama.Fore.RESET
            elif max_significant_freq >= (22050 * 0.85) and max_significant_freq < (22050 * 0.90):
                verdict = colorama.Fore.YELLOW + "Might be Fake" + colorama.Fore.RESET
            elif max_significant_freq >= (22050 * 0.90) and max_significant_freq < (22050 * 0.95):
                verdict = colorama.Fore.YELLOW + "Might be Authentic" + colorama.Fore.RESET
            elif max_significant_freq >= (22050 * 0.95) and max_significant_freq < (22050 * 0.99):
                verdict = colorama.Fore.GREEN + "Most likely Authentic" + colorama.Fore.RESET
            elif max_significant_freq >= (22050 * 0.99):
                verdict = colorama.Fore.GREEN + "Authentic" + colorama.Fore.RESET
            else:
                verdict = colorama.Fore.BLUE + "Can't Determine" + colorama.Fore.RESET
    else:
        verdict = colorama.Fore.YELLOW + "Can't determine" + colorama.Fore.RESET
        max_significant_freq = colorama.Fore.YELLOW + "N/A" + colorama.Fore.RESET

    try:
        dynamic_range, avg_peak, avg_rms, lufs = get_dr(audio_file_path)
    except TooShortError:
        dynamic_range = colorama.Fore.RED + "Too Short" + colorama.Fore.RESET
        avg_peak = colorama.Fore.YELLOW + "N/A" + colorama.Fore.RESET
        avg_rms = colorama.Fore.YELLOW + "N/A" + colorama.Fore.RESET
        lufs = colorama.Fore.YELLOW + "N/A" + colorama.Fore.RESET
    except SilentTrackError:
        dynamic_range = colorama.Fore.RED + "Silent Track" + colorama.Fore.RESET
        avg_peak = colorama.Fore.YELLOW + "N/A" + colorama.Fore.RESET
        avg_rms = colorama.Fore.YELLOW + "N/A" + colorama.Fore.RESET
        lufs = colorama.Fore.YELLOW + "N/A" + colorama.Fore.RESET

    if isinstance(dynamic_range, (int, float)):
        if dynamic_range < 8:
            dynamic_range = colorama.Fore.RED + str(dynamic_range) + colorama.Fore.RESET
        elif dynamic_range >= 8 and dynamic_range < 12:
            dynamic_range = colorama.Fore.YELLOW + str(dynamic_range) + colorama.Fore.RESET
        elif dynamic_range >= 12:
            dynamic_range = colorama.Fore.GREEN + str(dynamic_range) + colorama.Fore.RESET

    if isinstance(avg_peak, (int, float)):
        if avg_peak > -2:
            avg_peak = colorama.Fore.RED + f"{avg_peak:.2f} dB" + colorama.Fore.RESET
        elif -4 <= avg_peak <= -2:
            avg_peak = colorama.Fore.YELLOW + f"{avg_peak:.2f} dB" + colorama.Fore.RESET
        else:
            avg_peak = colorama.Fore.GREEN + f"{avg_peak:.2f} dB" + colorama.Fore.RESET

    if isinstance(avg_rms, (int, float)):
        if avg_rms > -6:
            avg_rms = colorama.Fore.RED + f"{avg_rms:.2f} dB" + colorama.Fore.RESET
        elif -9 <= avg_rms <= -6:
            avg_rms = colorama.Fore.YELLOW + f"{avg_rms:.2f} dB" + colorama.Fore.RESET
        else:
            avg_rms = colorama.Fore.GREEN + f"{avg_rms:.2f} dB" + colorama.Fore.RESET

    if isinstance(lufs, (int, float)):
        if lufs > -6:
            lufs = colorama.Fore.RED + f"{lufs:.2f} LUFS" + colorama.Fore.RESET
        elif -9 <= lufs <= -6:
            lufs = colorama.Fore.YELLOW + f"{lufs:.2f} LUFS" + colorama.Fore.RESET
        else:
            lufs = colorama.Fore.GREEN + f"{lufs:.2f} LUFS" + colorama.Fore.RESET

    max_significant_freq_str = colorama.Fore.CYAN + str(max_significant_freq) + colorama.Fore.RESET if max_significant_freq != "N/A" else max_significant_freq
    file_sample_rate_str = colorama.Fore.CYAN + str(file_sample_rate) + colorama.Fore.RESET

    return [audio_file, file_sample_rate_str, max_significant_freq_str, avg_peak, avg_rms, lufs, dynamic_range, verdict]

def main():
    folder_path = str(input("Enter the path to the folder containing the audio files: "))
    audio_files = [os.path.join(root, file) for root, dirs, files in os.walk(folder_path) for file in files if file.endswith(('.flac', '.wav'))]

    table = pt.PrettyTable()
    table.field_names = ["File", "Sample Rate", "Max Freq", "Avg Peak", "Avg RMS", "LUFS", "Dynamic Range", "Verdict"]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, audio_file_path) for audio_file_path in audio_files]

        for future in tqdm(as_completed(futures), total=len(audio_files), desc="Processing Audio Files", unit="file"):
            result = future.result()
            table.add_row(result)

    table.sortby = "File"
    print(table)

if __name__ == "__main__":
    main()
