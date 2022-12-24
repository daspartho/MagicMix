from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, logging
import torch
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from PIL import Image

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Loading components we'll use

tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-large-patch14",
)

text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14",
).to(device)

vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder = "vae",
).to(device)

unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder = "unet",
).to(device)

beta_start,beta_end = 0.00085,0.012
scheduler = DDIMScheduler(
    beta_start=beta_start,
    beta_end=beta_end,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    clip_sample=False, 
    set_alpha_to_one=False,
)


# convert PIL image to latents
def encode(img):
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(img).unsqueeze(0).to(device)*2-1)
        latent = 0.18215 * latent.latent_dist.sample()
    return latent


# convert latents to PIL image
def decode(latent):
    latent = (1 / 0.18215) * latent
    with torch.no_grad():
        img = vae.decode(latent).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
    img = (img * 255).round().astype("uint8")
    return Image.fromarray(img[0])


# convert prompt into text embeddings, also unconditional embeddings
def prep_text(prompt):

    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_embedding = text_encoder(
        text_input.input_ids.to(device)
    )[0]

    uncond_input = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    uncond_embedding = text_encoder(
        uncond_input.input_ids.to(device)
    )[0]

    return torch.cat([uncond_embedding, text_embedding])


def magic_mix(
    img, # specifies the layout semantics
    prompt, # specifies the content semantics
    kmin=0.3,
    kmax=0.6,
    v=0.5, # interpolation constant
    seed=42,
    steps=50,
    guidance_scale=7.5,
):

    tmin = steps- int(kmin*steps)
    tmax = steps- int(kmax*steps)

    text_embeddings = prep_text(prompt)

    scheduler.set_timesteps(steps)

    width, height = img.size
    encoded = encode(img)

    torch.manual_seed(seed)
    noise = torch.randn(
        (1,unet.in_channels,height // 8,width // 8),
    ).to(device)

    latents = scheduler.add_noise(
        encoded, 
        noise, 
        timesteps=scheduler.timesteps[tmax]
    )

    input = torch.cat([latents]*2)
                
    input = scheduler.scale_model_input(input, scheduler.timesteps[tmax])

    with torch.no_grad():
        pred = unet(
            input, 
            scheduler.timesteps[tmax],
            encoder_hidden_states=text_embeddings,
        ).sample

    pred_uncond, pred_text = pred.chunk(2)
    pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

    latents = scheduler.step(pred, scheduler.timesteps[tmax], latents).prev_sample

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        if i > tmax:
            if i < tmin: # layout generation phase
                orig_latents = scheduler.add_noise(
                    encoded, 
                    noise, 
                    timesteps=t
                )
                
                input = (v*latents) + (1-v)*orig_latents # interpolating between layout noise and conditionally generated noise to preserve layout sematics
                input = torch.cat([input]*2)

            else: # content generation phase
                input = torch.cat([latents]*2)
                
            input = scheduler.scale_model_input(input, t)

            with torch.no_grad():
                pred = unet(
                    input, 
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample

            pred_uncond, pred_text = pred.chunk(2)
            pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

            latents = scheduler.step(pred, t, latents).prev_sample

    return decode(latents)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("img_file", type=str, help="image file to provide the layout semantics for the mixing process")
    parser.add_argument("prompt", type=str, help="prompt to provide the content semantics for the mixing process")
    parser.add_argument("out_file", type=str, help="filename to save the generation to")
    parser.add_argument("--kmin", type=float, default=0.3)
    parser.add_argument("--kmax", type=float, default=0.6)
    parser.add_argument("--v", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)

    args = parser.parse_args()

    img = Image.open(args.img_file)
    out_img = magic_mix(
        img, 
        args.prompt,
        args.kmin,
        args.kmax,
        args.v,
        args.seed,
        args.steps,
        args.guidance_scale
        )
    out_img.save(args.out_file)