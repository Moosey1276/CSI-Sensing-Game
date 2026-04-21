import csv
from CSI_collection import read_serial_to_csv
import time

POSES = ["standing", "crouching", "ski_jump", "x_pose"]
LOOPS = 14
SECONDS = 8
REST = 5

def collect_data(ser, progress_queue):
    writers = {}
    files = {}
    for pose in POSES:
        f = open(f"{pose}2.txt", mode="w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow(["type","role","mac","rssi","rate","sig_mode","mcs","bandwidth","smoothing","not_sounding","aggregation","stbc","fec_coding","sgi","noise_floor","ampdu_cnt","channel","secondary_channel","local_timestamp","ant","sig_len","rx_state","real_time_set","real_timestamp","len","CSI_DATA"])
        files[pose] = f
        writers[pose] = writer

    print("Get ready to stand on the correct spot!")
    time.sleep(5)
    try:
        for loop in range (LOOPS):
            # print(f"\n=== Loop {loop + 1}/{LOOPS} ===")
            for i, pose in enumerate(POSES):
                # print(f"Do {pose} for {SECONDS} seconds...")
                progress_queue.put([pose, str(loop+1)])
                read_serial_to_csv(ser, writers[pose], duration=SECONDS)
                # print(f"Added {SECONDS}s of data to {pose}.csv")
                if not (loop+1 == LOOPS and pose == "x_pose"):
                    if i < len(POSES) - 1:
                        next_pose = POSES[i+1]
                    else:
                        next_pose = POSES[0]
                    progress_queue.put([pose, next_pose, str(loop+1)])
                    # print(f"Rest for {REST} seconds and get ready for pose {next_pose}\n")
                    time.sleep(REST)

    finally:
        for f in files.values():
            f.close()

    progress_queue.put(["quit"])