import numpy as np
import torch

from AircraftDataset import AircraftDataset
from model.fcos import FCOSDetector


class predict(object):
    def __init__(self, **kwargs):
        return

    def detect_image(self, image_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eval_dataset = AircraftDataset(image_path, mode='predict', resize_size=[512, 800])
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)

        model = FCOSDetector(mode="predict")
        model.load_state_dict(
            torch.load("./weights/aircraft_512x800_epoch23_loss0.5937.pth", map_location=torch.device('cpu')))

        model = model.to(device).eval()
        for img, boxes, classes in eval_loader:
            with torch.no_grad():
                out = model(img.to(device))
                pred_box = out[2][0].cpu().numpy()
                pred_class = out[1][0].cpu().numpy()

        # 创建一个空的二维列表
        combined_results = []

        # 合并数据
        for i in range(len(pred_box)):
            ymin, xmin, ymax, xmax = pred_box[i]
            class_pred = pred_class[i]

            combined_results.append([ymin, xmin, ymax, xmax, 0.8, class_pred])

        return combined_results
