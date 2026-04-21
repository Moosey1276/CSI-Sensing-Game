import tensorflow as tf
import tf2onnx
import subprocess
import os
import shutil

def keras2engine():
    model = tf.keras.models.load_model("engine.keras")

    # Check if it's Sequential
    if isinstance(model, tf.keras.Sequential):
        inputs = tf.keras.Input(shape=model.input_shape[1:])
        outputs = model(inputs)
        model = tf.keras.Model(inputs, outputs)

    spec = (tf.TensorSpec((None,) + model.input_shape[1:], tf.float32, name="input"),)
    output_path = ("engine.onnx")

    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

    print(f"ONNX model saved at: {output_path}")

    trtexec_path = shutil.which("trtexec")

    if trtexec_path:
        cmd = [trtexec_path, "--onnx=engine.onnx", "--saveEngine=poses.engine"]
        subprocess.run(cmd, check=True)
    else:
        print("Tensorrt bin folder not added to path!")

# keras2engine()