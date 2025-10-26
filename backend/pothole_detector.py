from ultralytics import YOLO

class PotholeDetector:
    def _init_(self, model_path: str, imgsz: int):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.names = None

    def predict_image(self, img_bgr, conf: float = 0.4):
        """
        input:  image
        output: list of detections as tuples:
                (x1, y1, x2, y2, conf, name, cls_id)
        """
        res_gen = self.model.predict(
            img_bgr, imgsz=self.imgsz, conf=conf, stream=True, verbose=False
        )
        try:
            # >> take first result 
            res = next(iter(res_gen))
        except StopIteration:
            return []

        if self.names is None:
            self.names = res.names

        out = []
        # >> loop over detections and collect data
        for b in res.boxes:
            cls_id = int(b.cls)
            conf_f = float(b.conf)
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            name = self.names[cls_id]
            out.append((x1, y1, x2, y2, conf_f, name, cls_id))
        return out