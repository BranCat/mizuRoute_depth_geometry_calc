# v2 (April 2026): script not preserved

v2 extended v1 by calibrating both the width scale and the width exponent (`w = wscale·Q^b`) using Nelder-Mead **without bounds**.

It was used for the Athabasca River (07DD001, gauge reach 222). The optimiser drove the exponent to about 0 (6×10⁻¹⁶), so width became a constant of about 438 m at every reach and every flow, including small tributaries. In another run the exponent went negative (−0.046), meaning channels narrowed as flow rose.

The resulting file, `channel_geometry_all-athabasca.nc`, was used in the SedHydro Paper 2 Athabasca runs. Its metadata records the v2 parameters.

These problems led to v3: bounded optimisation (L-BFGS-B) and excluding extreme discharge from calibration.
