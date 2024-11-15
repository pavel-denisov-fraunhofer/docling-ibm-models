from  torchvision.datasets import ImageFolder
import os
import json
from torchvision.transforms.functional import resize, to_tensor
import torch 
import nncf
import openvino as ov

doclaynet = "/path/to/doclaynet"

dataset = ImageFolder(doclaynet, allow_empty=True) # finds only .png files

train_coco = json.load(open(f"{doclaynet}/COCO/train.json"))
train_images = set([image["file_name"] for image in train_coco["images"]])
dataset.samples = [sample for sample in dataset.samples if os.path.basename(sample[0]) in train_images]

def transform_fn(data_item):
    img = to_tensor(data_item[0][0])
    return resize(img, (640, 640)).unsqueeze(0), torch.tensor([[img.shape[1], img.shape[2]]])

calibration_loader = torch.utils.data.DataLoader(dataset, collate_fn=transform_fn)
calibration_dataset = nncf.Dataset(calibration_loader)

core = ov.Core()
ov_model = core.read_model("openvino_conversion/models/layout_predictor/model.xml")
quantized_model = nncf.quantize(ov_model, calibration_dataset, fast_bias_correction=False, target_device=nncf.TargetDevice.CPU, subset_size=1000)
ov.save_model(quantized_model, "openvino_conversion/models/layout_predictor_quantized/model.xml")
