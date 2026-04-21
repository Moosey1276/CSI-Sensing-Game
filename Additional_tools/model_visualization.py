import tensorflow as tf
import visualkeras
from PIL import ImageFont

model_path = "model_3.keras"
model = tf.keras.models.load_model(model_path)

dummy_input = tf.zeros((1, 50, 52, 1))
_ = model(dummy_input)

for layer in model.layers:
    if not hasattr(layer, 'output_shape'):
        try:
            layer.output_shape = layer.output.shape
        except AttributeError:
            pass

print(f"{'Layer':25} {'Type':20} {'Output Shape'}")
print("-" * 100)

for layer in model.layers:
    print(
        f"{layer.name:25} "
        f"{layer.__class__.__name__:20} "
        f"{str(layer.output_shape)}"
    )

try:
    font = ImageFont.truetype("arial.ttf", 16)
except IOError:
    font = None

try:
    image = visualkeras.layered_view(
        model,
        legend=True,
        font=font,
        spacing=30,
        scale_xy=2,
        scale_z=1,
    )
    image.save("model_visualization_3_2.pdf")
    print("Success! Visualization saved as model_visualization.pdf")
except Exception as e:
    print(f"Visualization failed: {e}")
