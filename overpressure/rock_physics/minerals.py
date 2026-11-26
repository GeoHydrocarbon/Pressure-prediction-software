"""Editable starting values, not universal mineral constants.

Quartz/calcite/dolomite/illite/pyrite: reference workflow config.yaml at
GeoHydrocarbon/shale_oil_rp commit 6ad244cb95e01e1c2491b38ad871605b9db5fb8d.
Feldspar, mixed clay and muscovite: Table 3, JGE gxab062 (Mavko et al. 2009):
https://academic.oup.com/view-large/325015832
Units: density g/cm3, bulk and shear moduli GPa.
"""
MINERAL_PRESETS = {
    '石英': dict(density=2.65,K=37.,G=44.),
    '方解石': dict(density=2.71,K=76.8,G=32.),
    '白云石': dict(density=2.87,K=94.9,G=45.),
    '长石': dict(density=2.62,K=37.5,G=15.),
    '伊利石': dict(density=2.60,K=21.,G=7.),
    '黏土': dict(density=2.60,K=21.,G=7.),
    '黄铁矿': dict(density=4.93,K=147.4,G=132.5),
    '白云母': dict(density=2.79,K=61.5,G=41.1),
}
