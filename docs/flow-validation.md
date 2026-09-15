# Physical and pore-network validation

The dependency-light flow notebook validates Darcy arithmetic and invasion bookkeeping only. It does not simulate a rock network or reproduce thesis permeability values.

## Required real-data decisions

1. Register reference and prediction, preserve voxel spacing, and define a justified boolean pore mask. Mineral class IDs and grayscale thresholds are not interchangeable.
2. Obtain voxel size from provider metadata, expressed in **meters**. `configs/bentheimer.json` deliberately leaves it unset. Never interpret an unlabelled number such as 2.15 as a unit-safe length.
3. Install the optional backend in a separate environment with `pip install -e ".[flow]"`. Record the exact environment and solver versions.
4. Use PoreSpy SNOW2 with that meter-valued `voxel_size`, then import the extracted network through OpenPNM's public `network_from_porespy` API. Do not replace extracted radii/volumes with random geometry to make a solver run.
5. Validate network connectivity, boundary pores, geometry, hydraulic size factors, throat lengths, pore/throat volumes and any trimming. Confirm inlet/outlet choices align with the measured sample length and cross-sectional area.
6. Configure wetting/nonwetting phases, viscosity in Pa·s, hydraulic conductance, entry-pressure/contact-angle models and invasion boundary conditions. Run the invasion algorithm before evaluating its sequences.

PoreSpy describes SNOW2's parameters and unit convention in its [official API](https://porespy.org/autoapi/porespy/networks/snow2.html) and [worked network extraction example](https://porespy.org/examples/networks/tutorials/snow_basic.html).

## Absolute permeability

`darcy_permeability(Q, L, mu, A, delta_p)` accepts m³/s, m, Pa·s, m² and Pa respectively. It returns both m² and mD using `1 mD = 9.869233e-16 m²`. Length and area must be physical sample dimensions, not counts of voxels. Calculate area once in SI—do not multiply already-scaled area by voxel spacing again.

## Relative permeability helper

`relative_permeability(network, nonwetting_phase, wetting_phase, invasion, inlets, outlets)` is an optional caller-configured OpenPNM integration. It changes phase occupancies/models, not network geometry. Use dedicated phases if preserving a previous simulation state matters.

The helper uses single-phase hydraulic-conductance rates as reference flows and conduit conductance (`mode="medium"`) for occupied two-phase flow. It uses the same 1 Pa pressure drop for both, so the rate ratio defines the reference relative permeability. Finite sequences >=0 and below the chosen limit are invaded; negative/nonfinite sequences remain uninvaded. Saturation uses pore plus throat void volume. Other saturation conventions and conduit closure modes require an explicit scientific decision.

Validate the optional integration on a small known network before a real study. Check monotonic saturation, endpoint behavior, phase properties, positive reference rates, mass conservation, solver health and sensitivity to closure mode. Disconnected/trapped regions can prevent saturation reaching one and may require explicit handling. A small nonzero endpoint flow can reflect the conduit closure model rather than true residual permeability.

The automated dependency-light tests do not validate extraction or fluid models. Do not use the historical manually entered curves as outputs of the maintained helper.
