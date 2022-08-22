from scmdata.units import _unit_registry as unit_registry

unit_registry.define("H = [hydrogen]")
unit_registry.define("hydrogen = H")
unit_registry.define("t{symbol} = t * {symbol}".format(symbol="H"))
