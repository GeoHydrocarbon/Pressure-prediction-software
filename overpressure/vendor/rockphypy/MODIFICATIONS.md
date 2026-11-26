# RockPhyPy source integration

Source: GeoHydrocarbon/shale_oil_rp, rockphypy-main/rockphypy, commit 6ad244cb95e01e1c2491b38ad871605b9db5fb8d.
Original library: RockPhyPy 0.0.2, Jiaxin Yu and contributors; LGPL version 3 (see LICENSE).
Citation: Jiaxin Yu, Tapan Mukerji, Per Avseth, SoftwareX 24 (2023), 101567, doi:10.1016/j.softx.2023.101567.

Local changes (2026-11-25): absolute rockphypy imports are relative to this namespaced library; eager package imports are replaced by lazy adapter imports; the optional kde_diffusion import in QI is deferred until KDE plotting is requested. Original numerical models and documentation are retained. NumPy product calls, where present, use prod for NumPy 2 compatibility. Source and modified file hashes are recorded in SOURCE.json.

This library remains in a replaceable, separate source directory. The application rock_physics adapter performs parameter validation and tabular result conversion outside these modules. It does not use unrelated upstream data files, notebooks, workflows or fixed local paths.

## Numerical and compatibility corrections

- EM.Berryman_DEM integration grid ends at exactly the requested inclusion fraction (including a zero-fraction host-only result), rather than overshooting by up to 0.01.
- Fluid.Gassmann_vels converts density in g/cc and velocities in m/s to moduli in GPa, and converts the resulting velocities back to m/s. Independent same-fluid substitution preserves Vp/Vs; upstream mixed incompatible modulus units.
- AVO.zoeppritz computes the ray parameter from the radian angle theta1 (rather than sin of a degree value). The H term uses the reflected shear cosine/velocity, matching an independent four-equation displacement/traction boundary solve.
- Anisotropy.Bond_trans includes the factor 2 in the normal/shear Bond block for engineering Voigt stiffness. Independent isotropic rotation and inverse-rotation tests verify the convention; supported axes remain 1 and 3.
- QI's duplicate 0.05 colormap knot is separated by 1e-9 for modern Matplotlib validation. The optional kde-diffusion import remains deferred.
- The utils.V source density docstring says g/m3, but its equation requires g/cc for GPa and m/s. The adapter explicitly labels g/cc; the source docstring is retained for traceability.

Numerical adapter examples have input/domain validation and finite-output checks; advanced catalog methods without presets are not represented as independently scientifically verified. Known regression invariants are in tests/test_rock_physics.py. A numerical result alone does not validate a model's geological applicability.
