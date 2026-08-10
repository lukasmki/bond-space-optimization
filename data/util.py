"""Shared configuration for the HCombustion pipeline.

`configure_threads` / `available_cpus` now live in `bondspace.threads` so that
`experiments/` can use them without importing from this directory; they are
re-exported here so the stage scripts are unchanged.
"""

from bondspace.threads import available_cpus, configure_threads  # noqa: F401


rxn_data = {
    "rxn_01": {"charge": 0, "spin": 3},  # 3
    "rxn_02": {"charge": 0, "spin": 2},  # 2
    "rxn_03": {"charge": 0, "spin": 1},  # 1
    "rxn_04": {"charge": 0, "spin": 2},  # 2
    "rxn_05": {"charge": 0, "spin": 0},  # 0 -> 2
    "rxn_06": {"charge": 0, "spin": 2},  # 2 -> 4
    "rxn_07": {"charge": 0, "spin": 1},  # 2 -> 4
    "rxn_08": {"charge": 0, "spin": 0},  # 0 -> 2
    "rxn_09": {"charge": 0, "spin": 1},  # 1 -> 3
    "rxn_10": {"charge": 0, "spin": 2},  # 2
    "rxn_11": {"charge": 0, "spin": 2},  # 2
    "rxn_12": {"charge": 0, "spin": 3},  # 3
    "rxn_13": {"charge": 0, "spin": 2},  # 2
    "rxn_14": {"charge": 0, "spin": 2},  # 2
    "rxn_15": {"charge": 0, "spin": 0},  # 0 -> 2
    "rxn_16": {"charge": 0, "spin": 1},  # 1
    "rxn_17": {"charge": 0, "spin": 1},  # 1
    "rxn_18": {"charge": 0, "spin": 2},  # 2
    "rxn_19": {"charge": 0, "spin": 1},  # 1
}

# bond_data = {
#     "rxn_01": {"bonds": [(0, 1, 2.0)]},
#     "rxn_02": {"bonds": [()]},
#     "rxn_03": {"bonds": [()]},
#     "rxn_04": {"bonds": [()]},
#     "rxn_05": {"bonds": [()]},
#     "rxn_06": {"bonds": [()]},
#     "rxn_07": {"bonds": [()]},
#     "rxn_08": {"bonds": [()]},
#     "rxn_09": {"bonds": [()]},
#     "rxn_10": {"bonds": [()]},
#     "rxn_11": {"bonds": [()]},
#     "rxn_12": {"bonds": [()]},
#     "rxn_13": {"bonds": [()]},
#     "rxn_14": {"bonds": [()]},
#     "rxn_15": {"bonds": [()]},
#     "rxn_16": {"bonds": [()]},
#     "rxn_17": {"bonds": [()]},
#     "rxn_18": {"bonds": [()]},
#     "rxn_19": {"bonds": [()]},
# }
