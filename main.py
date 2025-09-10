# ===================================================================
# ALL IMPORTS
# ===================================================================
import requests
import pygame
import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import threading
import sounddevice as sd
from scipy.fft import fft
import time

# ===================================================================
# CONFIGURATION
# ===================================================================
BOT_TOKEN = "8282067282:AAHl9xUZvVyLnU8n_cIOs9D-TBB1uu9-QJo"
CHAT_ID = "7570730250"
PROCESS_EVERY_NTH_FRAME = 5 
RESIZE_WIDTH = 640          

# --- Audio Settings ---
SIREN_FREQUENCY_RANGE = (700, 1500) 
SIREN_LOUDNESS_THRESHOLD = 10      
SIREN_TIMEOUT = 2.0  # seconds until traffic light resets to red

# --- Shared State ---
last_siren_time = 0  # timestamp of last detected siren

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================
def send_alert(message="Siren detected! Clearing route."):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
        print("Alert sent successfully!")
    except Exception as e:
        print(f"Failed to send alert: {e}")

def choose_video_file():
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title="Select a Video File")
    return filepath

def audio_listener():
    global last_siren_time
    SAMPLE_RATE = 44100
    CHUNK_SIZE = 1024

    def audio_callback(indata, frames, time_info, status):
        global last_siren_time
        if status:
            print(status, flush=True)

        yf = fft(indata[:, 0])
        xf = np.fft.fftfreq(CHUNK_SIZE, 1 / SAMPLE_RATE)
        peak_index = np.argmax(np.abs(yf))
        peak_frequency = abs(xf[peak_index])
        peak_magnitude = np.abs(yf[peak_index])

        if (SIREN_FREQUENCY_RANGE[0] < peak_frequency < SIREN_FREQUENCY_RANGE[1] and
            peak_magnitude > SIREN_LOUDNESS_THRESHOLD):
            last_siren_time = time.time()  # update last detection time
            print(f"SIREN DETECTED! (Freq: {peak_frequency:.0f} Hz, Loudness: {peak_magnitude:.0f})")

    print("🎤 Starting audio listener...")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE):
        while True:
            sd.sleep(1000)

# ===================================================================
# MAIN PROGRAM LOGIC
# ===================================================================
def main():
    listener_thread = threading.Thread(target=audio_listener, daemon=True)
    listener_thread.start()

    # --- Choose video via dialog ---
    video_path = choose_video_file()
    if not video_path:
        print("No video selected, exiting program.")
        return

    pygame.init()
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_height = int(RESIZE_WIDTH * (frame_height / frame_width))
    screen = pygame.display.set_mode((RESIZE_WIDTH + 150, new_height))
    pygame.display.set_caption("Live Ambulance Detection")
    
    alert_sent, running, frame_count = False, True, 0
    last_results = None 

    RED, GREEN, BLACK = (255,0,0), (0,255,0), (0,0,0)
    BOX_COLOR_DEFAULT = (255, 0, 0) # Blue

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Siren detection logic ---
        current_time = time.time()
        siren_heard = (current_time - last_siren_time) < SIREN_TIMEOUT

        if siren_heard and not alert_sent:
            print("\n✅ SIREN CONFIRMED!")
            send_alert("🚨 Siren detected near Junction A! Clearing route now.")
            alert_sent = True

        success, frame = cap.read()
        if not success:
            # 🔄 Restart video when it ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        resized_frame = cv2.resize(frame, (RESIZE_WIDTH, new_height))
        
        # Run visual detection periodically
        if frame_count % PROCESS_EVERY_NTH_FRAME == 0:
            last_results = model(resized_frame, verbose=False) 
        
        # Draw detection boxes
        if last_results:
            for r in last_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = model.names[int(box.cls[0])]
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), BOX_COLOR_DEFAULT, 2)
                    cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_COLOR_DEFAULT, 2)

        screen.fill(BLACK)
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frame_pygame = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
        screen.blit(frame_pygame, (0, 0))

        # Traffic light based on siren presence
        light_color = GREEN if siren_heard else RED
        traffic_light_x = RESIZE_WIDTH + 75
        pygame.draw.circle(screen, light_color, (traffic_light_x, 100), 40)
            
        pygame.display.flip()
        frame_count += 1

    cap.release()
    pygame.quit()
    print("\nProgram finished.")

if __name__ == "__main__":
    main()
