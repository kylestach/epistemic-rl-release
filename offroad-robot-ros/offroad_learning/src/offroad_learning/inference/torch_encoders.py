import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import collections
import numpy as np
import os

torch.set_float32_matmul_precision("high")


def transform_images(img: Image.Image, img_size):  # takes in PIL Image
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * (4 / 3))))  # crop to the right ratio
    else:
        img = TF.center_crop(img, (int(w / (4 / 3)), w))
    img = img.resize(img_size)
    transf_img = transform(img)
    return transf_img


class DinoEncoder:
    def __init__(self, num_stack):
        self.dinov2_vits = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14"
        ).cuda()
        self.embedding_history = collections.deque(maxlen=num_stack)
        for _ in range(num_stack):
            self.embedding_history.append(np.zeros((384,)))
        self.img_size = (126, 126)

    def forward(self, img: Image.Image, goal_image: Image.Image):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                pixels_torch = transform_images(img, self.img_size).unsqueeze(0).cuda()
                embedding = self.dinov2_vits(pixels_torch).squeeze(0)
        self.embedding_history.append(embedding.cpu().numpy())
        return np.concatenate(self.embedding_history, axis=-1)


class FullGnmEncoder:
    def __init__(self, encoder_file: str, encoder_dir: str):
        self.img_size = (128, 96)
        model = torch.load(
            os.path.join(encoder_dir, encoder_file), map_location="cuda:0"
        )["model"]
        self.gnm = model

        self.gnm.eval()

        self.latest_pixels = collections.deque(maxlen=model.context_size + 1)
        self.latest_embeddings = collections.deque(maxlen=model.context_size + 1)
        for _ in range(model.context_size + 1):
            self.latest_pixels.append(torch.zeros((3, 128, 96)).cuda())
        for _ in range(model.context_size + 1):
            self.latest_embeddings.append(
                torch.zeros((1, model.obs_encoding_size)).cuda()
            )

    def forward(self, img: Image.Image, goal_image: Image.Image):
        pixels_torch = transform_images(img, self.img_size).cuda()
        self.latest_pixels.append(pixels_torch)

        if goal_image is None:
            print("No goal image")
        else:
            with torch.no_grad():
                # Run the encoders on the latest image only
                goal_torch = (
                    transform_images(goal_image, self.img_size).cuda().unsqueeze(0)
                )
                obs_embedding, obsgoal_embedding = self.gnm.forward_encoder(
                    pixels_torch.unsqueeze(0), goal_torch
                )

                # Stack the latest images and embeddings
                self.latest_embeddings.append(obs_embedding)
                obs_embedding_stacked = torch.stack(list(self.latest_embeddings), dim=1)

                assert obs_embedding_stacked.shape == (
                    1,
                    self.gnm.context_size + 1,
                    self.gnm.obs_encoding_size,
                ), f"obs_embedding.shape: {obs_embedding.shape}"
                assert obsgoal_embedding.shape == (
                    1,
                    self.gnm.goal_encoding_size,
                ), f"obsgoal_embedding.shape: {obsgoal_embedding.shape}"

                embedding = self.gnm.forward_transformer(
                    obs_embedding_stacked, obsgoal_embedding
                ).squeeze()

            return embedding.cpu().numpy()
