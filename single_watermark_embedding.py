# Define a 32-bit message to be embedded into the images
import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image
from watermark_anything.models import wam, embedder, extractor
from watermark_anything.data.metrics import msg_predict_inference
from notebooks.inference_utils import (
    load_model_from_checkpoint,
    default_transform,
    create_random_mask,
    unnormalize_img,
    plot_outputs,
    msg2str
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wm_msg = torch.randint(0, 2, (32,)).float().to(device).unsqueeze(0)
img_dir = "./assets/images"  # Directory containing input images


def msg2str(msg):
    # Flatten the tensor and convert each element to an integer (0 or 1)
    msg = msg.view(-1).int().tolist()
    return "".join(['1' if el else '0' for el in msg])


def load_img(path):
    img = Image.open(path).convert("RGB")
    img = default_transform(img).unsqueeze(0).to(device)
    return img


def create_output_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Load the model from the specified checkpoint
exp_dir = "checkpoints"
json_path = os.path.join(exp_dir, "params.json")
ckpt_path = os.path.join(exp_dir, 'wam_mit.pth')
wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()

output_dir = "./assets/output_single_watermark"

create_output_dir(output_dir)

# Proportion of the image to be watermarked (0.5 means 50% of the image).
# This is used here to show the watermark localization property.
# In practice, you may want to use a predefined mask or the entire image.
proportion_masked = 0.5

# Iterate over each image in the directory
for img_ in os.listdir(img_dir):
    # Load and preprocess the image
    img_path = os.path.join(img_dir, img_)
    print("Processing image:", img_)
    img = Image.open(img_path).convert("RGB")
    img_pt = default_transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

    # Embed the watermark message into the image
    outputs = wam.embed(img_pt, wm_msg)

    # Create a random mask to watermark only a part of the image
    mask = create_random_mask(img_pt, num_masks=1, mask_percentage=proportion_masked)  # [1, 1, H, W]
    img_w = outputs['imgs_w'] * mask + img_pt * (1 - mask)  # [1, 3, H, W]

    # Detect the watermark in the watermarked image
    preds = wam.detect(img_w)["preds"]  # [1, 33, 256, 256]
    mask_preds = F.sigmoid(preds[:, 0, :, :])  # [1, 256, 256], predicted mask
    bit_preds = preds[:, 1:, :, :]  # [1, 32, 256, 256], predicted bits

    # Predict the embedded message and calculate the bit accuracy
    pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()  # [1, 32]
    # Predict message for each image and save to folder
    print("Original message string:  ", msg2str(wm_msg))
    print("Predicted message string: ", msg2str(pred_message))
    # save message strings to a text file
    with open(f"{output_dir}/messages.txt", "a") as text_file:
        text_file.write(f"Image: {img_}\n")
        text_file.write(f"Original message string: {msg2str(wm_msg)}\n")
        text_file.write(f"Predicted message string: {msg2str(pred_message)}\n")
        text_file.write("\n")

    bit_acc = (pred_message == wm_msg).float().mean().item()

    # Save the watermarked image and the detection mask
    mask_preds_res = F.interpolate(mask_preds.unsqueeze(1), size=(img_pt.shape[-2], img_pt.shape[-1]), mode="bilinear",
                                   align_corners=False)  # [1, 1, H, W]
    save_image(unnormalize_img(img_w), f"{output_dir}/{img_}_wm.png")
    save_image(mask_preds_res, f"{output_dir}/{img_}_pred.png")
    save_image(mask, f"{output_dir}/{img_}_target.png")

    # Print the predicted message and bit accuracy for each image
    print(f"Predicted message for image {img_}: ", pred_message[0].numpy())
    print(f"Bit accuracy for image {img_}: ", bit_acc)
