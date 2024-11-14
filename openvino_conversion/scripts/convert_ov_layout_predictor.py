import openvino as ov
import torch

img = torch.randn(1, 3, 640, 640, dtype=torch.float32)
orig_size = torch.tensor([[1237, 1612]]).to(dtype=torch.int64)

ov_model = ov.convert_model(self.model, example_input=(img, orig_size,))
ov.save_model(ov_model, 'openvino_conversion/models/layout_predictor/model.xml')
