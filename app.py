import gradio as gr
from magic_mix import magic_mix

iface = gr.Interface(
    description="Implementation of MagicMix: Semantic Mixing with Diffusion Models paper",
    article="<p style='text-align: center'><a href='https://github.com/daspartho/MagicMix' target='_blank'>Github</a></p>",
    fn=magic_mix,
    inputs=[
        gr.Image(shape=(512, 512), type="pil"),
        gr.Text(),
        gr.Slider(value=0.3, minimum=0, maximum=1, step=0.1),
        gr.Slider(value=0.6, minimum=0, maximum=1, step=0.1),
        gr.Slider(value=0.5, minimum=0, maximum=1, step=0.1),
        gr.Number(value=42, maximum=2**64 - 1),
        gr.Slider(value=50),
        gr.Slider(value=7.5, minimum=1, maximum=15, step=0.1),
    ],
    outputs=gr.Image(),
    title="MagicMix",
)

iface.launch()
