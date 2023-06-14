"""Code to extract clean voice from any noisy audio."""

from typing import Optional

import torch
import torch.nn as nn
from df.enhance import enhance, init_df
from noisereduce.torchgate import TorchGate as TG
from torchaudio.functional import resample

__all__ = [
    "VoiceExtractor",
    "normalize_volume",
    "normalize_distribution",
]


# Competition audio files are 22050Hz.
DEFAULT_SR = 22050

# TODO: Implement Automatic Gain Control.


def normalize_volume(wav: torch.Tensor):
    """Normalize volume."""
    return wav / wav.abs().max()


def normalize_distribution(wav: torch.Tensor):
    """Normalize distribution."""
    return (wav - wav.mean()) / wav.std()


class VoiceExtractor(nn.Module):
    """Class to extract voice from audio tensor."""

    # NOTE: Below was already tuned using the validation split.
    # TODO: Expose below kwargs for tuning in the config file.
    def __init__(
        self,
        model_dir: str,
        sr: int = DEFAULT_SR,
        spectral_gate_std_thres: float = 1.5,
        spectral_noise_remove: float = 1.0,
        spectral_n_fft: int = 512,
        spectral_freq_mask_hz: Optional[int] = None,
        spectral_time_mask_ms: Optional[int] = None,
        df_post_filter: bool = False,
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

        Parameters
        ----------
        model_dir : str
            Path to DeepFilterNet model directory.
        sr : int, optional
            Input sampling rate, by default DEFAULT_SR
        spectral_gate_std_thres : float, optional
            Standard deviation threshold for noise, by default 1.5
        spectral_noise_remove : float, optional
            Proportion of noise to remove, by default 1.0
        spectral_n_fft : int, optional
            FFT sample size, by default 512
        spectral_freq_mask_hz : Optional[int], optional
            Frequency smoothing of mask to remove artifacts, by default None
        spectral_time_mask_ms : Optional[int], optional
            Time smoothing of mask to remove artifacts, by default None
        df_post_filter : bool, optional
            Apply DeepFilterNet post-filtering, by default False
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
        self.sr = sr
        self.model, self.df_state, _ = init_df(
            model_base_dir=model_dir, post_filter=df_post_filter, log_level="WARNING"
        )

    def _deepfilter(self, wav: torch.Tensor, sr: int, remove_limit_db: float = 0):
        # Glitch with DeepFilterNet library means wav has to be on CPU first.
        ori_device = wav.device
        wav = resample(wav, orig_freq=sr, new_freq=self.df_state.sr())
        wav = wav.to("cpu")[None]
        # If `atten_lim_db` is set to 0, it will fully suppress the noise.
        wav = enhance(
            self.model, self.df_state, wav, atten_lim_db=remove_limit_db, pad=True
        )
        return wav[0].to(ori_device), self.df_state.sr()

    def _spectral(
        self, wav: torch.Tensor, sr: int, noise: Optional[torch.Tensor] = None
    ):
        wav = resample(wav, orig_freq=sr, new_freq=self.spectral.sr)
        noise = noise[None] if noise is not None else None
        wav = self.spectral(wav[None], noise)[0]
        return wav, self.spectral.sr

    @torch.inference_mode()
    def forward(
        self,
        wav: torch.Tensor,
        sr: int,
        skip_vol_norm: bool = False,
        skip_df: bool = False,
        skip_spectral: bool = True,
        use_ori: bool = False,
        noise_removal_limit_db: float = 0,
        return_noise: bool = False,
    ):
        """Extract voice from audio.

        Parameters
        ----------
        wav : torch.Tensor
            Input audio tensor of shape (T,).
        sr : int
            Input sampling rate.
        skip_vol_norm : bool, optional
            Skip volume normalization for tuning purposes, by default False
        skip_df : bool, optional
            Skip usage of DeepFilterNet for tuning purposes, by default False
        skip_spectral : bool, optional
            Skip usage of spectral gating for tuning purposes, by default True
        use_ori : bool, optional
            Apply spectral gating on original audio, by default False
        noise_removal_limit_db : float, optional
            Limit of noise to remove, by default 0
        return_noise : bool, optional
            Return the extracted noise sample for debug purposes, by default False

        Returns
        -------
        wav, sr : torch.Tensor, int
            Extracted voice audio tensor of shape (T,) and sampling rate.
        """
        assert len(wav.shape) == 1, "Input must be T."

        if not skip_vol_norm:
            wav = normalize_volume(wav)

        ori_wav, ori_sr = wav, sr
        noise = None

        if not skip_df:
            wav, sr = self._deepfilter(wav, sr, noise_removal_limit_db)

            # Use extracted voice sample to find noise sample.
            if use_ori:
                noisy = ori_wav
                noise_sr = ori_sr
                voice = resample(wav, orig_freq=sr, new_freq=ori_sr)
            else:
                noisy = resample(ori_wav, orig_freq=ori_sr, new_freq=sr)
                noise_sr = sr
                voice = wav
            noise = noisy - voice

        if not skip_spectral:
            if use_ori:
                wav, sr = self._spectral(ori_wav, ori_sr, noise)
            else:
                wav, sr = self._spectral(wav, sr, noise)

        wav = resample(wav, orig_freq=sr, new_freq=self.sr)
        if return_noise:
            if noise is None:
                return wav, self.sr, wav
            noise = resample(noise, orig_freq=noise_sr, new_freq=self.sr)
            return wav, self.sr, noise
        return wav, self.sr
