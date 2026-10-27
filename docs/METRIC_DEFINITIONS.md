# Metric definitions

These are the conventions implemented by the companion workflows. Scientific
tests are under `tests/`; full execution instructions are in [WORKFLOW.md](WORKFLOW.md).

## ACC and Activity

For case j and grid point i, let a_ji = forecast_ji - climatology_ji and
b_ji = ERA5_ji - climatology_ji. Spatial weights w_i retain cosine-latitude
weighting and the nine-cell boundary exclusion.

`ACC_j = sum_i(w_i*a_ji*b_ji) / sqrt(sum_i(w_i*a_ji^2)*sum_i(w_i*b_ji^2))`.

ACC is the arithmetic mean of finite case correlations at each channel and
lead. No additional spatial demeaning is applied to these anomalies.

`v_j = sum_i[w_i*(a_ji-weighted_mean_i(a_ji))^2] / sum_i(w_i)`.

Activity is `sqrt(mean_j(v_j))`. ERA5 activity uses b_ji, and Activity/ERA5 is
the ratio of the two aggregated activities, not the mean of case ratios.
Overall channel activity aggregates variances across cases and leads before
taking the square root. Zero-weight cells are excluded before arithmetic;
nonfinite positive-weight values invalidate that diagnostic. Zero ACC
denominators give NaN; zero activity is valid. Finite-case counts are metric-,
channel- and lead-specific. RMSE, bias and absolute activity use variable units;
ACC and Activity/ERA5 are dimensionless.

## Exact-radius GWB

GWB uses the height integral

`Delta Z_GWB(R) = (1/g) * integral_0^R (Vtheta(r)^2/r + f*Vtheta(r)) dr`.

The GNN and Pangu implementations use SI units, regular-origin Vtheta=O(r),
zero origin integrand, linear interpolation of the integrand at exactly R and
trapezoidal integration on [0,R]. Ring means, centres, tangential projection
and Coriolis conventions follow the diagnostic modules. The upper supporting
ring is used only to interpolate the endpoint. Missing first, interior or
endpoint-support rings invalidate the radius; gaps are not bridged.
The regular-origin condition does not resolve the unobserved inner core.
This is distinct from integrating only between ring centres. GWB and SIM
height differences are expressed in metres.

## Spectral power

For a band B of integer radial bins, the band mean square is

`V_B = sum_{k in B}(N_k * mean(|FFT(field)|^2)_k) / (H*W)^2`.

N_k counts actual modes of the 174×174 FFT grid using
`floor(hypot(integer_fft_indices))`. It is not a sample count or a 2*pi*k
approximation. NumPy's unnormalized forward transform gives the Parseval
factor `(H*W)^2`. Band ratios are ratios of modal-weighted sums, rather than
sums of annular means or a mean of individual ratios. Nonpositive denominators
are missing. Annular ratios retain their original source means.

Retained non-DC annuli are k=1–87; square-corner modes are excluded, so this
is not the complete FFT total. DC-inclusive cumulative products describe mean
square rather than zero-mean variance. Humidity units are `(kg/kg)^2`, wind
component units `(m/s)^2`. These are scalar mean-square diagnostics, not
domain-integrated kinetic energy. Horizontal kinetic energy would additionally
require the matching u/v components and a factor of one half.

Nominal wavelength is `4872/k km = 174*28/k km`. The 28-km isotropic
approximation does not resolve latitude-dependent zonal spacing. Applying
the nominal <1000-km criterion to whole annuli selects **k=5–87**. Modes within
the boundary annulus cannot be separated from annular means; exact geographic
scale separation requires reprojection/resampling and recomputation from
individual fields. Each model retains its existing aggregate population and
ERA5 denominator. The monthly maps use one initialization while their spectra
use aggregate populations. Figure 14 pairs wind-speed maps with **850v** spectra.

Coherence uses the actual radial cross-spectrum calculation; an energy ratio
or SNR conversion is not a substitute. Training loss uses reconstructed
standardized fields, whereas diagnostic spectra use their documented physical
field conventions.

## Signed thermal wind

Stored `forecast_vtl`, `target_vtl`, `forecast_vtu` and `target_vtu` represent
signed **-V_T^L** and **-V_T^U**, in metres. Positive values indicate warm cores;
negative values indicate cold cores. No absolute value or extra sign flip is
applied to the slope.

At each native pressure level:

`Delta Z_core(p) = max_{d<=500 km} Z(p) - min_{d<=500 km} Z(p)`.

Use equal-weight OLS of this range against `ln(p / 1 hPa)`. Lower levels are
exactly `(925,850,700,600)` hPa and upper levels `(600,500,400,300)` hPa.
At least two finite native values are required in each layer. Do not insert
interpolated levels or fill missing layers. Convert geopotential to height
using g=9.80665 m/s² before taking extrema. Forecast and target centres follow
their respective diagnostic workflows. This height-range regression differs
from the radial SIM quantity `Z(R)-Z(0)`. The lower layer here is 925–600 hPa;
Hart's conventional lower layer is 900–600 hPa.

Phase analysis also writes counts and method metadata beside its output CSV.
These describe the calculation and missing-level eligibility.

## TC structural distributions and tails

B remains signed: motion-relative right-minus-left mean 925–600 hPa thickness
within 500 km. Use Q95 for B and the thermal-wind indicators; use Q05 for
minimum sea-level pressure and Rmax. Q95 of signed B describes strong positive
directional asymmetry; it is not the quantile of absolute B.

Composite comparisons use the finite common cases required by the plotting
group and a shared ERA5 reference. Case-paired errors and marginal distribution
differences measure different properties. KS D describes marginal differences;
the reported nominal two-sample p-values do not account for forecast pairing
or repeated forecasts within storms. Large p-values do not establish
distributional equivalence. Pressure-group vertical composites additionally
require compatible profiles from all three GNNs.
