from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools.core import util
from tqdm import tqdm

from kvae_audio import KVAEAudio


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(weights_path: str, device: torch.device) -> KVAEAudio:
    weights_path = Path(weights_path)
    print(f"Loading weights from {weights_path.resolve()}")
    generator = KVAEAudio.load(str(weights_path), map_location="cpu")
    generator = generator.to(device)
    generator.eval()
    return generator


@torch.no_grad()
def infer_signal(
    signal: AudioSignal,
    device: torch.device,
    generator: KVAEAudio,
) -> AudioSignal:
    signal = signal.to(device)
    output = generator(signal.audio_data)["audio"]
    recons = AudioSignal(output, signal.sample_rate)
    recons = recons.normalize(signal.loudness())
    return recons.cpu()


@argbind.bind(without_prefix=True)
@torch.no_grad()
def main(
    weights_path: str = "kvae-audio.pt",
    input_path: str = "input_samples",
    output_path: str = "output_samples",
):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    generator = load_model(weights_path, device)

    audio_files = util.find_audio(input_path)
    for audio_file in tqdm(audio_files, desc="Infer"):
        signal = AudioSignal(audio_file)
        recons = infer_signal(signal, device, generator)
        recons.write(output_dir / audio_file.name)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
