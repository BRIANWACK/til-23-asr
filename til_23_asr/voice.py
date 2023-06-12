"""Code to extract clean voice from any noisy audio."""

from typing import Optional

import torch
import torch.nn as nn
from demucs.apply import apply_model
from demucs.pretrained import get_model
from noisereduce.torchgate import TorchGate as TG
from torchaudio.functional import resample

__all__ = ["load_demucs_model", "VoiceExtractor"]

DEMUCS_MODEL = "htdemucs_ft"
DEMUCS_MODEL_REPO = None

# Competition audio files are 22050Hz.
DEFAULT_SR = 22050
BEST_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_demucs_model(name=DEMUCS_MODEL, repo=DEMUCS_MODEL_REPO):
    """Load demucs model."""
    return get_model(name=name, repo=repo)


class VoiceExtractor(nn.Module):
    """Class to extract voice from audio tensor."""

    # NOTE: Below was already tuned using the validation split.
    # TODO: Expose below kwargs for tuning in the config file.
    def __init__(
        self,
        sr: int = DEFAULT_SR,
        spectral_gate_std_thres: float = 1.0,
        spectral_noise_remove: float = 1.0,
        spectral_n_fft: int = 512,
        spectral_freq_mask_hz: Optional[int] = None,
        spectral_time_mask_ms: Optional[int] = None,
        demucs_shifts: int = 4,
        skip_demucs: bool = False,
        skip_spectral: bool = False,
        use_ori: bool = False,
        return_noise: bool = False,
        model=None,
    ):
        """Initialize VoiceExtractor.

        All `spectral` keyword arguments configure the Stationary Spectral Gating
        from https://github.com/timsainb/noisereduce.

        Spectral gating without a good noise sample typically causes artifacting
        in the audio. `spectral_freq_mask_hz` and `spectral_time_mask_ms` smoothes
        the noise mask over frequency and time respectively to reduce this artifacting
        when not set to None. Conversely, it has been observed a good noise sample
        removes the need for any mask smoothing. The noise sample is obtained by
        subtracting the extracted voice from the original audio. This approach has
        been observed to work well.

        `demucs_shifts` improves the voice extraction quality at the cost of inference
        time.

        Parameters
        ----------
        sr : int, optional
            Input sampling rate, by default DEFAULT_SR
        spectral_gate_std_thres : float, optional
            Standard deviation threshold for noise, by default 1.0
        spectral_noise_remove : float, optional
            Proportion of noise to remove, by default 1.0
        spectral_n_fft : int, optional
            FFT sample size, by default 512
        spectral_freq_mask_hz : Optional[int], optional
            Frequency smoothing of mask to remove artifacts, by default None
        spectral_time_mask_ms : Optional[int], optional
            Time smoothing of mask to remove artifacts, by default None
        demucs_shifts : int, optional
            Number of random shifts to apply in `demucs`, by default 4
        skip_demucs : bool, optional
            Skip usage of demucs for tuning purposes, by default False
        skip_spectral : bool, optional
            Skip usage of spectral gating for tuning purposes, by default False
        use_ori : bool, optional
            Apply spectral gating on original audio instead of `demucs`-ed audio, by default False
        return_noise : bool, optional
            Return the extracted noise sample for debug purposes, by default False
        model : Any, optional
            Alternative `demucs` model to use, by default None
        """
        super(VoiceExtractor, self).__init__()
        # See: https://github.com/timsainb/noisereduce.
        self.spectral = TG(
            sr=sr,
            nonstationary=False,
            n_std_thresh_stationary=spectral_gate_std_thres,
            prop_decrease=spectral_noise_remove,
            n_fft=spectral_n_fft,
            # Below smoothes out the noise mask to prevent artifacting.
            # NOTE: Given a good noise sample, there is no artifacting even with
            # the masks disabled.
            freq_mask_smooth_hz=spectral_freq_mask_hz,
            time_mask_smooth_ms=spectral_time_mask_ms,
        )
        self.demucs = load_demucs_model() if model is None else model
        self.skip_demucs = skip_demucs
        self.skip_spectral = skip_spectral
        self.use_ori = use_ori
        self.return_noise = return_noise
        self.demucs_shifts = demucs_shifts

    def _spectral(
        self, wav: torch.Tensor, sr: int, noise: Optional[torch.Tensor] = None
    ):
        wav = resample(wav, orig_freq=sr, new_freq=self.spectral.sr)
        noise = noise[None] if noise is not None else None
        wav = self.spectral(wav[None], noise)[0]
        return wav, self.spectral.sr

    def _demucs(self, wav: torch.Tensor, sr: int):
        wav = resample(wav, orig_freq=sr, new_freq=self.demucs.samplerate)
        wav = wav[None].expand(self.demucs.audio_channels, -1)
        # Copied from `demucs.separate.main` (v4.0.0) to ensure correctness.
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = apply_model(
            self.demucs,
            wav[None],  # BCT
            shifts=self.demucs_shifts,
            split=True,
            overlap=0.25,  # Applies only if split=True
            progress=False,
            num_workers=0,  # Ignored if GPU is used
        )[0]
        sources = sources * ref.std() + ref.mean()
        wav = sources[self.demucs.sources.index("vocals")]
        wav = wav.mean(0)
        return wav, self.demucs.samplerate

    @torch.inference_mode()
    def forward(self, wav: torch.Tensor, sr: int):
        """Extract voice from audio."""
        assert len(wav.shape) == 1, "Input must be T."

        ori_wav, ori_sr = wav, sr
        noise = None

        if not self.skip_demucs:
            wav, sr = self._demucs(wav, sr)

            # Use extracted voice sample to find noise sample.
            if self.use_ori:
                noisy = ori_wav
                noise_sr = ori_sr
                voice = resample(wav, orig_freq=sr, new_freq=ori_sr)
            else:
                noisy = resample(ori_wav, orig_freq=ori_sr, new_freq=sr)
                noise_sr = sr
                voice = wav
            noise = noisy - voice

        if not self.skip_spectral:
            if self.use_ori:
                wav, sr = self._spectral(ori_wav, ori_sr, noise)
            else:
                wav, sr = self._spectral(wav, sr, noise)

        if self.return_noise:
            noise = resample(noise, orig_freq=noise_sr, new_freq=sr)
            return wav, sr, noise
        return wav, sr
