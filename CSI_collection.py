import time

def read_serial_to_csv(ser, writer, duration=10):
    print_line = False
    start_time = time.time()
    while time.time() - start_time < duration:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                if not print_line:
                    print(f"{line}")
                    print_line = True
                if line.count('[') == 1 and line.count(']') == 1:
                    line = line.split(',')
                    CSI_data = line[-1].replace('[', '').replace(']', '').split(' ')
                    if line[0] == "CSI_DATA" and line[1] == "AP" and len(line) == 26 and len(CSI_data) == 129:
                        writer.writerow([field.strip() for field in line])