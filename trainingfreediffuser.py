import math
from pathlib import Path
import typer
from diffusers import (
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.models.attention import Attention
import torch

from aux import list_layers, Tref
from utils import parse_wh, seed_everything


class TrainingFreeAttnProcessor:
    def __init__(self, name: str = None):
        self.name = name
        self.is_mid = None
        if name is not None:
            self.is_mid = "mid_block" in name

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states
        is_selfatten = encoder_hidden_states is None
        is_selfatten2 = "attn1" in self.name
        assert is_selfatten == is_selfatten2

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if is_selfatten:
            T = list(Tref[self.name])
            assert len(T) == 1
            T = T[0]
            dim_head = attn.inner_dim / attn.heads

            qk_scale = ((math.log(sequence_length, T)) / dim_head) ** 0.5
            attn.scale = qk_scale

        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


app = typer.Typer()


@app.command()
def generate_single(
    prompt: str = typer.Argument(..., help="Prompt"),
    enable: bool = typer.Option(False, "--enable", "-e", help="Enable reweighting"),
    wh: str = typer.Option("512", help="Width,Height; in case no ',' -> square"),
    output_path: Path = typer.Option(None, help="output basepath"),
    n_samples: int = typer.Option(1, "--n_samples", "-n", help="nsample"),
    seed: int = typer.Option(0, "--seed", "-s", help="seed"),
):
    seed_everything(seed)
    model_id = "stabilityai/stable-diffusion-2-1-base"
    w, h = parse_wh(wh)

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=torch.float16
    )
    unet = pipe.unet
    unet: UNet2DConditionModel

    if enable:
        unet.set_attn_processor(
            {name: TrainingFreeAttnProcessor(name) for name in list_layers}
        )
    pipe = pipe.to("cuda")

    if output_path is None:
        output_basepath = Path("outputs") / prompt / "image"
    for i in range(n_samples):
        image = pipe(prompt, width=w, height=h).images[0]
        output_real_name = (
            output_basepath.stem
            + f"_seed{seed}_wh_{w}x{h}_{i}_{'reweighted' if enable else 'original'}.png"
        )
        output_path = output_basepath.parent / output_real_name
        output_path.parent.mkdir(exist_ok=True, parents=True)

        image.save(output_path.as_posix())


if __name__ == "__main__":
    app()
