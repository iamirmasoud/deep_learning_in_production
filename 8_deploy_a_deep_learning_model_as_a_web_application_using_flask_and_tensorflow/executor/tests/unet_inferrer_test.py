import unittest

from PIL import Image
import numpy as np

from executor.unet_inferrer import UnetInferrer


class MyTestCase(unittest.TestCase):
    def test_infer(self):
        image = np.asarray(
            Image.open(
                "/media/masoud/F60C689F0C685C9D/GIT_REPOS/ML_OPS/deep_learning_in_production/8_deploy_a_deep_learning_model_as_a_web_application_using_flask_and_tensorflow/app/resources/yorkshire_terrier.jpg"
            )
        ).astype(np.float32)
        inferrer = UnetInferrer()
        inferrer.infer(image)


if __name__ == "__main__":
    unittest.main()
