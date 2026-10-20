# Revised structural-distribution text

Paste-ready revision of the supplied paragraph (Figure 9 numbering retained).
No manuscript DOCX or TeX source is present in this workspace; this file is
replacement text, not an edit to the manuscript itself.

We further evaluate structural parameters through their full distributions and tails (Figure 9). We quantify case-paired errors using RMSE and describe marginal distributional discrepancies using the Kolmogorov–Smirnov (KS) statistic D (Massey Jr., 1951). Because the forecast and reanalysis samples are paired and include repeated forecasts of individual TCs, the reported nominal KS p-values do not account for these dependencies; large p-values do not establish distributional equivalence. Tail behavior is summarized using the 5th percentile for MSL and Rmax, and the 95th percentile for B and the other parameters. The lower-tail diagnostics represent low central pressure and compact inner cores, respectively, while the upper tail of signed B represents strong positive motion-relative thickness asymmetry associated with typical extratropical transition. As shown in Section 4, spectral energy decays at different rates for specific humidity and wind, indicating variable-dependent smoothing behavior. Similar differences are evident among the TC structural diagnostics.

## Structure-figure caption

Use the figure number in the current manuscript; the earlier local artifact
folder calls this structure figure Figure 10.

Tropical cyclone structure diagnostics at 72 h. (a) Rmax–B and (b) −V_T^U–−V_T^L joint distributions. Colored shading represents forecast distributions, while gray-to-black ERA5 contours enclose 50%, 75%, and 90% of the probability mass. Marginal densities are shown above and to the right of each panel. (c) Case-paired RMSE, marginal two-sample Kolmogorov–Smirnov statistics D with corresponding nominal p-values, and forecast tail quantiles (Q05 for Rmax; Q95 for B, −V_T^U, and −V_T^L). KS D describes marginal distributional discrepancies; nominal p-values do not account for pairing or repeated forecasts within TCs, and large p-values do not establish distributional equivalence. B retains its sign and uses the motion-relative right-minus-left mean 925–600 hPa thickness within 500 km. The thermal-wind parameters use native levels spanning 925–600 and 600–300 hPa; positive values indicate warm cores and negative values cold cores. All models use the same common cases and shared ERA5 reference.

## Implementation scope

B is summarized by Q95 without taking its absolute value. MSL and Rmax retain
Q05. This update does not add uncertainty intervals for KS D or tail quantiles;
paired, whole-storm resampling remains a possible additional analysis.
