import sys
sys.path.append(".")
from gradcam import GradCam

def test_everything(snapshot):
    filename = "examples/cat_dog.png"
    layer_name = "block5_conv3"
    model_filename = "model.h5"

    gc = GradCam.from_hdf(model_filename)
    cam, heatmap = gc.compute_cam(filename, layer_name)
    snapshot.assert_match(cam)
    snapshot.assert_match(heatmap)
