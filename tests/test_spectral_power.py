import unittest
import numpy as np
from agu_james.analysis.spectral import modal_weights, mode_counts, physical_band_power
from agu_james.plotting.energy import compute_energy_ratios, total_energy_mask


class SpectralPower(unittest.TestCase):
    def test_parseval_and_retained_annuli(self):
        x = np.random.default_rng(3407).normal(size=(174, 174))
        power = np.abs(np.fft.fftshift(np.fft.fft2(x)))**2
        yy, xx = np.indices(x.shape)
        bins = np.floor(np.hypot(yy-87, xx-87)).astype(int)
        counts = mode_counts()
        sums = np.bincount(bins.ravel(), weights=power.ravel())
        means = sums / np.maximum(counts, 1)
        self.assertEqual(counts.sum(), x.size)
        self.assertAlmostEqual(sums.sum()/x.size**2, np.mean(x*x))
        k = np.arange(88)
        self.assertAlmostEqual(np.sum(means[:88]*modal_weights(k)), power[bins<=87].sum()/x.size**2)

    def test_weighted_ratio_and_nominal_cutoff(self):
        k = np.arange(88)
        forecast = np.ones((1,88));target = np.ones((1,88))
        forecast[0,5] = 10
        result = compute_energy_ratios(k,forecast,target)
        weights = mode_counts()[5:88]
        expected = (10*weights[0]+weights[1:].sum()) / weights.sum()
        self.assertAlmostEqual(result[-1][0], expected)
        self.assertFalse(np.isclose(expected,forecast[0,5:].mean()))
        np.testing.assert_array_equal(np.flatnonzero(total_energy_mask(k)),np.arange(5,88))

    def test_physical_cartesian_spacing(self):
        xx=np.arange(174)[None,:]
        wave=np.broadcast_to(np.cos(2*np.pi*7*xx/174),(174,174))
        self.assertAlmostEqual(physical_band_power(wave,28,28),.5)
        self.assertLess(physical_band_power(wave,56,28),1e-25)
        self.assertAlmostEqual(physical_band_power(wave.T,56,28),.5)

    def test_zero_power(self):
        k=np.arange(88)
        a=np.ones((1,88));b=np.ones((1,88));b[0,8]=0
        self.assertTrue(np.isnan(compute_energy_ratios(k,a,b)[3][0,7]))
        with self.assertRaises(ValueError):compute_energy_ratios(k,a,np.zeros_like(b))


if __name__=='__main__':unittest.main()
