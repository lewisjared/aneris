from scmdata.units import _unit_registry as unit_registry

if not hasattr(unit_registry, "hydrogen"):
    unit_registry.define("H = [hydrogen] = H2")
    unit_registry.define("hydrogen = H")
    unit_registry.define("t{symbol} = t * {symbol}".format(symbol="H"))

ur = unit_registry
