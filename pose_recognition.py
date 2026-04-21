import math
import numpy as np
import pandas as pd
import pycuda.driver as cuda
import tensorrt as trt
from collections import deque, Counter
from shared_state import shared_pose, stop_event
import time

POSE_MAP = {
    2: "standing",
    0: "crouching",
    1: "ski_pose",
    3: "x_pose"
}

def load_engine(engine_file_path):
    with open(engine_file_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
        engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
    return engine


def allocate_buffers(engine):
    binding_name_input = engine.get_tensor_name(0)
    binding_name_output = engine.get_tensor_name(1)

    h_input = cuda.pagelocked_empty(trt.volume(engine.get_tensor_shape(binding_name_input)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_tensor_shape(binding_name_output)), dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    return h_input, d_input, h_output, d_output

def inference(engine, h_input, d_input, h_output, d_output, input_data):
    stream = cuda.Stream()
    input_data = np.array(input_data)
    input_data /= 255
    np.random.seed(123)
    # input_data = np.random.rand(*h_input.shape).astype(np.float32)
    np.copyto(h_input, input_data.ravel())
    binding_name_input = engine.get_tensor_name(0)

    cuda.memcpy_htod_async(d_input, h_input, stream)

    context = engine.create_execution_context()
    context.set_input_shape(binding_name_input, input_data.shape)  # If dynamic shapes
    context.set_tensor_address(binding_name_input, int(d_input))
    binding_name_output = engine.get_tensor_name(1)
    context.set_tensor_address(binding_name_output, int(d_output))
    context.execute_async_v3(stream.handle)

    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    return h_output

def process(res):
    # Parser
    all_data = res.split(',')
    csi_data = all_data[25].split(" ")
    csi_data[0] = csi_data[0].replace("[", "")
    csi_data[-1] = csi_data[-1].replace("]", "")

    csi_data.pop()
    csi_data = [int(c) for c in csi_data if c]
    imaginary = []
    real = []
    for i, val in enumerate(csi_data):
        if i % 2 == 0:
            imaginary.append(val)
        else:
            real.append(val)

    csi_size = len(csi_data)
    amplitudes = []
    if len(imaginary) > 0 and len(real) > 0:
        for j in range(int(csi_size / 2)):
            amplitude_calc = math.sqrt(imaginary[j] ** 2 + real[j] ** 2)
            amplitudes.append(amplitude_calc)
    df = pd.DataFrame(amplitudes)
    # print("Dataframe: " + df)
    return df


def predict(df, engine, h_input, d_input, h_output, d_output, prediction_buffer):
    columns_to_drop = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]  # note for later, in training changed to also delete 0, 1
    df.drop(df.columns[columns_to_drop], axis=1, inplace=True)
    # df = df.transpose().values.reshape((200, 55, 1))
    # print("Dataframe shape: " + df.shape)
    input_for_inference = df.values.astype(np.float32).reshape(1, 50, 52, 1)
    output_result = inference(engine, h_input, d_input, h_output, d_output, input_for_inference)

    predicted_class = int(np.argmax(output_result))

    prediction_buffer.append(predicted_class)
    print(prediction_buffer)

    most_common = Counter(prediction_buffer).most_common(1)[0][0]

    pose = POSE_MAP.get(most_common, "unknown")
    shared_pose.set_pose(pose)

    print("Raw prediction:", predicted_class, " | Smoothed:", most_common)
    if most_common == 0:
        print(r""" ####### 
#     # 
#     # 
#     # 
####### """)
    elif most_common == 1:
        print(r"""    #    
    #    
    #    
    #    
    #    """)
    elif most_common == 2:
        print(r""" ####### 
      # 
####### 
#       
####### """)
    else:
        print(r"""####### 
      # 
####### 
      # 
####### """)


def main(ser):
    cuda.init()
    device = cuda.Device(0)
    context = device.make_context()
    try:
        count = 0
        dfs = []
        prediction_buffer = deque(maxlen=5)

        engine_file_path = 'poses.engine'

        trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(trt_logger, '')
        engine = load_engine(engine_file_path)
        h_input, d_input, h_output, d_output = allocate_buffers(engine)
        print(h_input.shape)
        try:
            start = time.time()
            while not stop_event.is_set():
                try:
                    data = ser.readline().decode("utf-8").strip()
                    if "CSI_DATA" in data:
                        df = process(data)
                        df_transposed = df.transpose()
                        # print(df_transposed.shape)
                        if df_transposed.shape[1] == 64:
                            # Append the DataFrame to the list
                            dfs.append(df_transposed)
                            count += 1
                        if (count == 50):
                            # print("Chunk", len(perm_amp))
                            result_df = pd.concat(dfs, axis=0)
                            # result_df = result_df.reset_index(drop=True)
                            print(result_df.shape)
                            dfs = []
                            count = 0
                            predict(result_df, engine, h_input, d_input, h_output, d_output, prediction_buffer)
                            end = time.time()
                            print(end - start)

                except Exception as e:
                    print("Error:", {e})
                    pass

        except KeyboardInterrupt:
            print("Exiting gracefully.")
        finally:
            ser.close()
    finally:
        context.pop()

# if __name__ == "__main__":
#     main()