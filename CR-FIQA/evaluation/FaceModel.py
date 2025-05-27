import cv2
import numpy as np
from sklearn.preprocessing import normalize


class FaceModel():
    def __init__(self, model_prefix, model_epoch, ctx_id=7, backbone="iresnet50"):
        self.gpu_id = ctx_id
        self.image_size = (112, 112)
        self.model_prefix = model_prefix
        self.model_epoch = model_epoch
        self.model = self._get_model(ctx=ctx_id, image_size=self.image_size, prefix=self.model_prefix,
                                     epoch=self.model_epoch, layer='fc1', backbone=backbone)

    def _get_model(self, ctx, image_size, prefix, epoch, layer):
        pass

    def _getFeatureBlob(self, input_blob):
        pass

    def get_feature(self, image_path, color="BGR"):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Image at {image_path} could not be loaded.")
            return None

        try:
            image = cv2.resize(image, (112, 112))
        except cv2.error as e:
            print(f"Warning: Failed to resize image at {image_path}. Error: {e}")
            return None

        if color == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        a = np.transpose(image, (2, 0, 1))
        input_blob = np.expand_dims(a, axis=0)
        emb = self._getFeatureBlob(input_blob)
        if emb is not None:
            emb = normalize(emb.reshape(1, -1))
        return emb

    def get_batch_feature(self, image_path_list, batch_size=16, color="BGR"):
        count = 0
        num_batch = int(len(image_path_list) / batch_size)
        features = []
        quality_score = []

        for i in range(0, len(image_path_list), batch_size):
            if count < num_batch:
                tmp_list = image_path_list[i: i + batch_size]
            else:
                tmp_list = image_path_list[i:]
            count += 1

            images = []
            for image_path in tmp_list:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Image at {image_path} could not be loaded.")
                    continue

                try:
                    image = cv2.resize(image, (112, 112))
                except cv2.error as e:
                    print(f"Warning: Failed to resize image at {image_path}. Error: {e}")
                    continue

                if color == "RGB":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                a = np.transpose(image, (2, 0, 1))
                images.append(a)

            if not images:
                continue

            input_blob = np.array(images)
            emb, qs = self._getFeatureBlob(input_blob)
            if emb is not None and qs is not None:
                quality_score.append(qs)
                features.append(emb)

        if features:
            features = np.vstack(features)
            features = normalize(features)
        else:
            features = np.array([])

        if quality_score:
            quality_score = np.vstack(quality_score)
        else:
            quality_score = np.array([])

        return features, quality_score
