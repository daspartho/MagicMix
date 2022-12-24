# MagicMix
Implementation of [MagicMix: Semantic Mixing with Diffusion Models](https://arxiv.org/pdf/2210.16056.pdf) paper.

![magicmix](https://user-images.githubusercontent.com/59410571/206903603-6c8da6ef-69c4-4400-b4a3-aef9206ff396.png)

The aim of the method is to mix two different concepts in a semantic manner to synthesize a new concept while preserving the spatial layout and geometry.

The method takes an image that provides the layout semantics and a prompt that provides the content semantics for the mixing process.

There are 3 parameters for the method-
- `v`: It is the interpolation constant used in the layout generation phase. The greater the value of v, the greater the influence of the prompt on the layout generation process.
- `kmax` and `kmin`: These determine the range for the layout and content generation process. A higher value of kmax results in loss of more information about the layout of the original image and a higher value of kmin results in more steps for content generation process.

### Usage

```python
from PIL import Image
from magic_mix import magic_mix

img = Image.open('phone.jpg')
out_img = magic_mix(img, 'bed', kmax=0.5)
out_img.save("mix.jpg")
```
```
python3 magic_mix.py \
    "phone.jpg" \
    "bed" \
    "mix.jpg" \
    --kmin 0.3 \
    --kmax 0.6 \
    --v 0.5 \
    --steps 50 \
    --seed 42 \
    --guidance_scale 7.5
```
Also, check out the [demo notebook](https://github.com/daspartho/MagicMix/blob/main/demo.ipynb) for example usage of the implementation to reproduce examples from the paper.

### Some examples reproduced from the paper:

##### Input Image:

![telephone](https://user-images.githubusercontent.com/59410571/206903102-34e79b9f-9ed2-4fac-bb38-82871343c655.jpg)

##### Prompt: "Bed"

##### Output Image:

![telephone-bed](https://user-images.githubusercontent.com/59410571/206903104-913a671d-ef53-4ae4-919d-64c3059c8f67.jpg)

##### Input Image:

![sign](https://user-images.githubusercontent.com/59410571/206903307-b066dddd-8aaf-4104-9d5c-8427a51f37a7.jpg)

##### Prompt: "Family"

##### Output Image:

![sign-family](https://user-images.githubusercontent.com/59410571/206903320-7530a8ac-6594-4449-8328-bbc31befd9e8.jpg)

##### Input Image:

![sushi](https://user-images.githubusercontent.com/59410571/206903325-a06268ef-903e-434b-8365-68fb8b003d1e.jpg)

##### Prompt: "ice-cream"

##### Output Image:

![sushi-ice-cream](https://user-images.githubusercontent.com/59410571/206903341-e66d5c27-1543-489f-833b-dc8afc6c68e6.jpg)

##### Input Image:

![pineapple](https://user-images.githubusercontent.com/59410571/206903362-7c0464a7-ace4-4810-8fe3-37cab3d929a6.jpg)

##### Prompt: "Cake"

##### Output Image:

![pineapple-cake](https://user-images.githubusercontent.com/59410571/206903377-3b0fb63c-061e-4070-a8d1-eaca5738ae36.jpg)

### Note
**I'm not the author of the paper, and this is not an official implementation**
