import openvino as ov
import torch
from huggingface_hub import snapshot_download
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

class ModelWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        download_path = snapshot_download(repo_id="ds4sd/docling-models", revision="v2.1.0")
        layout_predictor = LayoutPredictor(artifact_path=download_path + "/model_artifacts/layout/")
        self._model = layout_predictor._model

    def forward(self, x):
        outputs = self._model(pixel_values=x)
        return outputs["logits"], outputs["pred_boxes"]

model = ModelWrapper()
img = torch.randn(1, 3, 640, 640, dtype=torch.float32)

ov_model = ov.convert_model(model, example_input=img)
ov.save_model(ov_model, 'openvino_conversion/models/layout_predictor/model.xml')
