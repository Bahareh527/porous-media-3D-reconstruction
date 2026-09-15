"""Unit-safe Darcy permeability and invasion-sequence saturation helpers."""

import numpy as np

MILLIDARCY_M2 = 9.869233e-16


def darcy_permeability(
    flow_rate_m3_s: float,
    length_m: float,
    viscosity_pa_s: float,
    area_m2: float,
    pressure_drop_pa: float,
) -> dict:
    values = np.asarray(
        [flow_rate_m3_s, length_m, viscosity_pa_s, area_m2, pressure_drop_pa], dtype=float
    )
    if not np.isfinite(values).all() or flow_rate_m3_s < 0 or np.any(values[1:] <= 0):
        raise ValueError("Darcy inputs must be finite, SI-valued, and physically valid")
    permeability = float(flow_rate_m3_s * length_m * viscosity_pa_s / (area_m2 * pressure_drop_pa))
    return {"permeability_m2": permeability, "permeability_md": permeability / MILLIDARCY_M2}


def invasion_limits(pore_sequence, throat_sequence, points: int = 20):
    if points < 2:
        raise ValueError("at least two saturation points are required")
    sequences = np.concatenate(
        [np.asarray(pore_sequence).ravel(), np.asarray(throat_sequence).ravel()]
    )
    invaded = sequences[np.isfinite(sequences) & (sequences >= 0)]
    maximum = int(invaded.max()) + 1 if invaded.size else 1
    return np.unique(np.linspace(0, maximum, points, dtype=int))


def occupancy_and_saturation(
    pore_sequence, throat_sequence, pore_volume, throat_volume, limit: int
):
    pore_sequence, throat_sequence = np.asarray(pore_sequence), np.asarray(throat_sequence)
    pore_volume, throat_volume = (
        np.asarray(pore_volume, dtype=float),
        np.asarray(throat_volume, dtype=float),
    )
    if pore_sequence.shape != pore_volume.shape or throat_sequence.shape != throat_volume.shape:
        raise ValueError("sequence and volume shapes must match")
    for volume in (pore_volume, throat_volume):
        if not np.isfinite(volume).all() or np.any(volume < 0):
            raise ValueError("network volumes must be finite and nonnegative")
    total = float(pore_volume.sum() + throat_volume.sum())
    if total <= 0:
        raise ValueError("network void volume must be positive")
    pores = np.isfinite(pore_sequence) & (pore_sequence >= 0) & (pore_sequence < limit)
    throats = np.isfinite(throat_sequence) & (throat_sequence >= 0) & (throat_sequence < limit)
    saturation = float((pore_volume[pores].sum() + throat_volume[throats].sum()) / total)
    return pores, throats, saturation


def relative_permeability(
    network, nonwetting_phase, wetting_phase, invasion, inlets, outlets, *, points: int = 20
) -> dict:
    """Evaluate caller-configured OpenPNM phases without changing extracted geometry.

    Caller must provide validated, connected geometry, hydraulic size factors,
    fluid properties and invasion boundary conditions. This optional integration
    is separate from the dependency-light numerical validation suite.
    """
    try:
        import openpnm as op
    except ImportError as error:
        raise ImportError("flow simulations require pip install '.[flow]'") from error

    def rate(phase, conductance):
        phase.regenerate_models()
        algorithm = op.algorithms.StokesFlow(network=network, phase=phase)
        algorithm.settings["conductance"] = conductance
        algorithm.set_value_BC(pores=inlets, values=1.0)
        algorithm.set_value_BC(pores=outlets, values=0.0)
        algorithm.run()
        return float(np.abs(algorithm.rate(pores=inlets, mode="group")).sum())

    phases = (nonwetting_phase, wetting_phase)
    absolute = [rate(phase, "throat.hydraulic_conductance") for phase in phases]
    if min(absolute) <= 0:
        raise ValueError("single-phase reference flow must be positive")
    for phase in phases:
        phase.add_model(
            propname="throat.conduit_hydraulic_conductance",
            model=op.models.physics.multiphase.conduit_conductance,
            throat_conductance="throat.hydraulic_conductance",
            mode="medium",
            regen_mode="deferred",
        )
    saturation, nonwetting, wetting = [], [], []
    for limit in invasion_limits(
        invasion["pore.invasion_sequence"], invasion["throat.invasion_sequence"], points
    ):
        pores, throats, value = occupancy_and_saturation(
            invasion["pore.invasion_sequence"],
            invasion["throat.invasion_sequence"],
            network["pore.volume"],
            network["throat.volume"],
            int(limit),
        )
        nonwetting_phase["pore.occupancy"], nonwetting_phase["throat.occupancy"] = pores, throats
        wetting_phase["pore.occupancy"], wetting_phase["throat.occupancy"] = ~pores, ~throats
        saturation.append(value)
        nonwetting.append(
            rate(nonwetting_phase, "throat.conduit_hydraulic_conductance") / absolute[0]
        )
        wetting.append(rate(wetting_phase, "throat.conduit_hydraulic_conductance") / absolute[1])
    return {"nonwetting_saturation": saturation, "kr_nonwetting": nonwetting, "kr_wetting": wetting}
