import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import threading
import time
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa.display

class PitchAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pitch Analyzer")
        
        self.temp_audio = os.path.join(os.path.dirname(__file__), "temp_audio.wav")
        self.temp_output = os.path.join(os.path.dirname(__file__), "temp_output.mkv")

        self.label = tk.Label(root, text="Select an audio or video file to play at 432 Hz.")
        self.label.pack(pady=10)

        self.browse_button = tk.Button(root, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.result_label.pack(pady=5)

        self.play_button = tk.Button(root, text="Play with VLC", command=self.start_playback_thread, state=tk.DISABLED)
        self.play_button.pack(pady=0)

        self.stop_button = tk.Button(root, text="Stop VLC", command=self.stop_vlc, state=tk.DISABLED)
        self.stop_button.pack(pady=0)

        self.file_path = None
        self.vlc_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
        self.vlc_cache_gen_path = r"C:\Program Files\VideoLAN\VLC\vlc-cache-gen.exe"
        self.ffmpeg_path = os.path.join(os.path.dirname(__file__), "bin", "ffmpeg.exe")
        self.ffmpeg_process = None
        self.vlc_process = None
        self.running = False

        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 4))
        self.fig.tight_layout(pad=4.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=20)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.stop_vlc()
        self.cleanup_temp_files()
        self.root.destroy()

    def browse_file(self):
        self.cleanup_temp_files()
        self.file_path = filedialog.askopenfilename(
            filetypes=(("Audio/Video Files", "*.mp3;*.wav;*.flac;*.mp4"), ("All Files", "*.*"))
        )
        if self.file_path:
            self.label.config(text=f"Selected file: {self.file_path}")
            self.play_button.config(state=tk.NORMAL)
            self.analyze_pitch()

    def analyze_pitch(self):
        if self.file_path.lower().endswith(('.mp4', '.mov', '.avi')):
            self.extract_audio()
            audio_path = self.temp_audio
        else:
            audio_path = self.file_path

        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            y, sr = librosa.load(audio_path, sr=None)
            self.plot_spectrum(y, sr, "Original Audio Spectrum", 0)
        except Exception as e:
            print(f"Error analyzing pitch: {str(e)}")
            messagebox.showerror("Error", f"Error analyzing pitch: {str(e)}")

    def extract_audio(self):
        ffmpeg_command = [self.ffmpeg_path, "-y", "-i", self.file_path, "-q:a", "0", "-map", "a", self.temp_audio]
        result = subprocess.run(ffmpeg_command)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            messagebox.showerror("Error", "Failed to extract audio using FFmpeg.")

    def start_playback_thread(self):
        playback_thread = threading.Thread(target=self.play_with_vlc)
        playback_thread.start()

    def play_with_vlc(self):
        if self.file_path:
            try:
                if not os.path.exists(self.vlc_path):
                    raise FileNotFoundError(f"VLC Player not found at {self.vlc_path}")

                if not os.path.exists(self.ffmpeg_path):
                    raise FileNotFoundError(f"FFmpeg not found at {self.ffmpeg_path}")

                file_path = os.path.abspath(self.file_path)

                ffmpeg_command = [
                    self.ffmpeg_path, "-i", file_path, "-filter_complex",
                    "[0:a]asetrate=44100*432/440,aresample=44100,atempo=1.0185185185185186",
                    "-c:v", "copy", "-f", "matroska", self.temp_output
                ]

                self.ffmpeg_process = subprocess.Popen(ffmpeg_command, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)

                time.sleep(2)

                if not os.path.exists(self.temp_output):
                    raise FileNotFoundError(f"Temporary output file not found: {self.temp_output}")

                vlc_command = [
                    self.vlc_path, self.temp_output
                ]
                self.vlc_process = subprocess.Popen(vlc_command, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)

                self.play_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.running = True

                self.update_spectrum_thread = threading.Thread(target=self.update_spectrum)
                self.update_spectrum_thread.start()

                self.monitor_vlc()

            except FileNotFoundError as e:
                error_message = f"An error occurred: {str(e)}"
                print(error_message)
                messagebox.showerror("Error", error_message)
            except Exception as e:
                error_message = f"An error occurred while trying to play the file: {str(e)}"
                print(error_message)
                messagebox.showerror("Error", error_message)

    def monitor_vlc(self):
        while self.running:
            if self.ffmpeg_process:
                ffmpeg_line = self.ffmpeg_process.stderr.readline()
                if ffmpeg_line:
                    print("FFmpeg error:", ffmpeg_line.decode())
            if self.vlc_process:
                vlc_line = self.vlc_process.stderr.readline()
                if vlc_line:
                    print("VLC error:", vlc_line.decode())
            time.sleep(1)

        y_adjusted, sr_adjusted = librosa.load(self.temp_audio, sr=None)
        self.plot_spectrum(y_adjusted, sr_adjusted, "Adjusted Audio Spectrum", 1)

    def update_spectrum(self):
        while self.running:
            try:
                y, sr = librosa.load(self.temp_audio, sr=None)
                self.root.after(0, self.plot_spectrum, y, sr, "Live Adjusted Audio Spectrum", 1)
                time.sleep(1)
            except Exception as e:
                print(f"Error updating spectrum: {str(e)}")
                break

    def plot_spectrum(self, y, sr, title, subplot_index):
        self.ax[subplot_index].clear()
        D = np.abs(librosa.stft(y))**2
        S = librosa.feature.melspectrogram(S=D, sr=sr, n_fft=2048, n_mels=128, fmax=1024)
        S_dB = librosa.power_to_db(S, ref=np.max)

        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=self.ax[subplot_index], fmax=1024)
        self.ax[subplot_index].set(title=title, ylabel='Hz', ylim=(0, 1024))
        self.canvas.draw_idle()

    def stop_vlc(self):
        self.running = False
        if self.vlc_process:
            self.vlc_process.terminate()
            self.vlc_process.wait()
            self.vlc_process = None
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
        self.play_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def cleanup_temp_files(self):
        try:
            if os.path.exists(self.temp_audio):
                os.remove(self.temp_audio)
            if os.path.exists(self.temp_output):
                os.remove(self.temp_output)
            print("Temporary files cleaned up successfully.")
        except Exception as e:
            print(f"Error cleaning up temporary files: {str(e)}")

def run_vlc_cache_gen(vlc_cache_gen_path):
    command = [vlc_cache_gen_path, r"C:\Program Files\VideoLAN\VLC\plugins"]
    subprocess.run(command, creationflags=subprocess.CREATE_NO_WINDOW)
    print("VLC plugin cache cleaned.")

if __name__ == "__main__":
    run_vlc_cache_gen(r"C:\Program Files\VideoLAN\VLC\vlc-cache-gen.exe")
    root = tk.Tk()
    app = PitchAnalyzerApp(root)
    app.cleanup_temp_files()
    root.mainloop()
