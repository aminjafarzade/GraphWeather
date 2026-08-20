"""band_spectral_power_loss: the properties the old error-power loss gets wrong.

The two pinned behaviors are the exact failure modes of low_frequency_spectral_loss
for anti-blur use: (1) a blurred forecast must be PENALIZED (the old loss scores it
exactly 0), and (2) a forecast with the right band power but unpredictable phase
must NOT be penalized (any error-power form scores it ~2x the blurred one, i.e.
teaches blur).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import (  # noqa: E402
    band_spectral_power_loss,
    low_frequency_spectral_loss,
    low_frequency_spectral_mask,
    radial_band_ids,
)

H, W = 72, 144


def _lowpass(field: torch.Tensor, lat_cutoff: int = 8, lon_cutoff: int = 16) -> torch.Tensor:
    mask = low_frequency_spectral_mask(H, W, lat_cutoff, lon_cutoff)
    fft = torch.fft.rfft2(field, dim=(-2, -1), norm="ortho")
    return torch.fft.irfft2(fft * mask, s=(H, W), dim=(-2, -1), norm="ortho")


class TestBandSpectralPowerLoss(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.target = torch.randn(4, 3, H, W)

    def test_identical_prediction_scores_zero(self) -> None:
        loss = band_spectral_power_loss(
            self.target, self.target, [0, 1, 2], band_weights=[0.0, 1.0, 1.0]
        )
        self.assertLess(float(loss), 1.0e-10)

    def test_blurred_prediction_is_penalized_where_old_loss_is_blind(self) -> None:
        # weight_latitude=False isolates the pure spectral behavior: the
        # sqrt(cos) taper otherwise leaks a sliver of high-k error into the
        # masked band. Unweighted, the old loss is EXACTLY blind to blur.
        blurred = _lowpass(self.target)
        old = low_frequency_spectral_loss(
            blurred, self.target, [0, 1, 2], 8, 16, weight_latitude=False
        )
        new = band_spectral_power_loss(
            blurred, self.target, [0, 1, 2], band_weights=[0.0, 1.0, 1.0],
            weight_latitude=False,
        )
        self.assertLess(float(old), 1.0e-10)  # the documented blind spot
        self.assertGreater(float(new), 1.0)   # log-power gap on emptied bands is large

    def test_right_power_wrong_phase_is_not_penalized(self) -> None:
        # High-k content with per-sample random phase: correct amplitude,
        # unknowable placement. Power matching must score ~0; for reference,
        # the blurred alternative must score much worse.
        def with_random_phase(seed: int) -> torch.Tensor:
            generator = torch.Generator().manual_seed(seed)
            fft = torch.zeros(4, 1, H, W // 2 + 1, dtype=torch.cfloat)
            fft[..., 2, 3] = 6.0  # shared predictable low-k mode
            phase = torch.rand(4, 1, generator=generator) * (2.0 * torch.pi)
            fft[..., 20, 40] = 2.0 * torch.exp(1j * phase)
            return torch.fft.irfft2(fft, s=(H, W), dim=(-2, -1), norm="ortho")

        target = with_random_phase(1)
        sharp = with_random_phase(2)     # same band power, different phases
        blurred = _lowpass(target)
        weights = [0.0, 1.0, 1.0]
        loss_sharp = band_spectral_power_loss(sharp, target, [0], band_weights=weights)
        loss_blurred = band_spectral_power_loss(blurred, target, [0], band_weights=weights)
        self.assertLess(float(loss_sharp), 1.0e-6)
        self.assertGreater(float(loss_blurred), 100.0 * float(loss_sharp) + 1.0e-4)

    def test_gradient_flows_toward_restoring_power(self) -> None:
        blurred = _lowpass(self.target).clone().requires_grad_(True)
        loss = band_spectral_power_loss(
            blurred, self.target, [0, 1, 2], band_weights=[0.0, 1.0, 1.0]
        )
        loss.backward()
        self.assertIsNotNone(blurred.grad)
        self.assertGreater(float(blurred.grad.abs().max()), 0.0)

    def test_band_ids_match_diagnostics_binning(self) -> None:
        from src.diagnostics.spectral import radial_wavenumber_bins
        import numpy as np

        bins = radial_wavenumber_bins(H, W)
        occupied = torch.unique(bins).numpy()
        groups = np.array_split(occupied, 3)
        expected = torch.empty_like(bins)
        for band, group in enumerate(groups):
            for bin_idx in group:
                expected[bins == int(bin_idx)] = band
        torch.testing.assert_close(radial_band_ids(H, W, 3), expected)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            band_spectral_power_loss(self.target, self.target, [0], band_weights=[0.0, 0.0])
        with self.assertRaises(ValueError):
            band_spectral_power_loss(self.target, self.target, [], band_weights=[1.0])
        with self.assertRaises(ValueError):
            band_spectral_power_loss(self.target[..., :10], self.target, [0], band_weights=[1.0])

    def test_latitude_weighting_and_bf16_inputs(self) -> None:
        blurred = _lowpass(self.target)
        loss = band_spectral_power_loss(
            blurred.to(torch.bfloat16),
            self.target.to(torch.bfloat16),
            [0],
            band_weights=[0.2, 1.0, 1.0],
            weight_latitude=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
