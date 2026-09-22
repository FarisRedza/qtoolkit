import csv
import dataclasses
import pathlib
import typing
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import qtoolkit


@dataclasses.dataclass(frozen=True)
class ModelParameters:
    """Parameters that are independent of the variable channel loss."""

    heralding_1550_db: float
    heralding_780_db: float
    qber: float
    qx: float
    brightness_per_mw: float
    coincidence_window: float


@dataclasses.dataclass
class Datarun:
    corrections: typing.Optional[str]
    loss_dB: float
    fibre_length_km: float
    power_mW: float
    acquisition_time_s: float
    coincidence_window_ps: float
    brightness_cps: float
    heralding: float
    qber: float
    qx: float
    skr_bps: float
    tstamp: float
    exit_flag: typing.Optional[str]
    h1550: float
    v1550: float
    d1550: float
    a1550: float
    h780: float
    v780: float
    d780: float
    a780: float
    h1550h780: float
    h1550v780: float
    h1550d780: float
    h1550a780: float
    v1550h780: float
    v1550v780: float
    v1550d780: float
    v1550a780: float
    d1550h780: float
    d1550v780: float
    d1550d780: float
    d1550a780: float
    a1550h780: float
    a1550v780: float
    a1550d780: float
    a1550a780: float


@dataclasses.dataclass
class CoincidenceCounts:
    h1550h780: float
    h1550v780: float
    h1550d780: float
    h1550a780: float
    v1550h780: float
    v1550v780: float
    v1550d780: float
    v1550a780: float
    d1550h780: float
    d1550v780: float
    d1550d780: float
    d1550a780: float
    a1550h780: float
    a1550v780: float
    a1550d780: float
    a1550a780: float

    @classmethod
    def from_dataruns(cls, dataruns: list[Datarun]) -> 'CoincidenceCounts':
        """Get the average values from a list of Datarun objects."""
        acquisition_time = sum(d.acquisition_time_s for d in dataruns)
        return cls(
            **{
                field.name: sum(getattr(d, field.name) for d in dataruns)
                / acquisition_time
                for field in dataclasses.fields(cls)
            }
        )

    @property
    def total(self) -> float:
        return sum(self)

    def __iter__(self):
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)


def parse_data_from_file(
    file_path: typing.Union[pathlib.Path, str],
) -> list[Datarun]:
    """Read one processed experimental CSV file."""
    file_path = pathlib.Path(file_path)

    datapoints: list[Datarun] = []
    with file_path.open(newline='') as file:
        for row in csv.DictReader(file):
            values: list[float | str | None] = []
            for value in row.values():
                if value is None:
                    values.append(None)
                    continue
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    values.append(str(value))

            # Compatibility with the older processed-data format.
            if len(values) == 35:
                values.insert(5, 0.0)
                values.insert(6, None)

            datapoints.append(Datarun(*typing.cast(typing.Any, values)))

    return datapoints

def dead_time_corrected_efficiency(
    efficiency: float,
    brightness: float,
    dead_time: float,
    detector_count: int,
) -> float:
    """Apply the detector dead-time correction of Neumann et al., Eq. (B1).

    ``efficiency`` is the total efficiency of the communication arm before
    it is distributed over ``detector_count`` identical detectors.
    """
    dead_time_efficiency = 1.0 / (
        1.0 + brightness * efficiency * dead_time / detector_count
    )
    return efficiency * dead_time_efficiency


def detector_resolved_rates(
    *,
    brightness: float,
    efficiency_a: float,
    efficiency_b: float,
    coincidence_window: float,
    timing_imprecision: float,
    dark_count_a: float,
    dark_count_b: float,
    bit_error: float,
    phase_error: float,
    detector_count: int = 4,
    dead_time_a: float = 0.0,
    dead_time_b: float = 0.0,
) -> tuple[float, float, float]:
    """Return measured coincidence rate, bit error and phase error.

    The calculation assumes identical detectors and a correlated Bell state.
    """
    cw = qtoolkit.qkd.cw

    corrected_a = dead_time_corrected_efficiency(
        efficiency_a, brightness, dead_time_a, detector_count
    )
    corrected_b = dead_time_corrected_efficiency(
        efficiency_b, brightness, dead_time_b, detector_count
    )

    # Identical detectors split the arm efficiency equally.
    detector_efficiency_a = corrected_a / detector_count
    detector_efficiency_b = corrected_b / detector_count

    window_efficiency = cw.coincidence_window_efficiency(
        coincidence_window=coincidence_window,
        timing_imprecision=timing_imprecision,
    )

    true_cc = cw.true_coincidence_rate(
        brightness=brightness,
        efficiency_a=detector_efficiency_a,
        efficiency_b=detector_efficiency_b,
    )

    singles_a = cw.measured_singles_rate(
        true_singles_rate=cw.true_singles_rate(
            brightness=brightness,
            efficiency=detector_efficiency_a,
        ),
        dark_count_rate=dark_count_a,
    )
    singles_b = cw.measured_singles_rate(
        true_singles_rate=cw.true_singles_rate(
            brightness=brightness,
            efficiency=detector_efficiency_b,
        ),
        dark_count_rate=dark_count_b,
    )
    accidental_cc = cw.accidental_coincidence_rate_approx(
        measured_singles_a=singles_a,
        measured_singles_b=singles_b,
        coincidence_window=coincidence_window,
    )

    measured_per_pair = cw.measured_coincidence_rate(
        coincidence_window_efficiency=window_efficiency,
        true_coincidence_rate=true_cc,
        accidental_coincidence_rate=accidental_cc,
    )
    measured_cc = detector_count**2 * measured_per_pair

    erroneous_pair_count = detector_count * (detector_count - 1)

    erroneous_bit_cc = erroneous_pair_count * (
        window_efficiency * true_cc * bit_error + accidental_cc
    )
    erroneous_phase_cc = erroneous_pair_count * (
        window_efficiency * true_cc * phase_error + accidental_cc
    )

    return (
        measured_cc,
        cw.qber_from_rate(erroneous_bit_cc, measured_cc),
        cw.qber_from_rate(erroneous_phase_cc, measured_cc),
    )


def model_secure_key_rate(
    *,
    brightness: float,
    efficiency_a: float,
    efficiency_b: float,
    coincidence_window: float,
    timing_imprecision: float,
    dark_count_a: float,
    dark_count_b: float,
    qber: float,
    qx: float,
    detector_count: int = 4,
    dead_time_a: float = 0.0,
    dead_time_b: float = 0.0,
    error_correction_efficiency: float = 1.1,
) -> float:
    """Calculate the asymptotic BBM92 secure key rate."""
    measured_cc, bit_error, phase_error = detector_resolved_rates(
        brightness=brightness,
        efficiency_a=efficiency_a,
        efficiency_b=efficiency_b,
        coincidence_window=coincidence_window,
        timing_imprecision=timing_imprecision,
        dark_count_a=dark_count_a,
        dark_count_b=dark_count_b,
        bit_error=qber,
        phase_error=qx,
        detector_count=detector_count,
        dead_time_a=dead_time_a,
        dead_time_b=dead_time_b,
    )

    return qtoolkit.qkd.cw.secure_key_rate(
        measured_coincidence_rate=measured_cc,
        bit_error_rate=bit_error,
        phase_error_rate=phase_error,
        sifting_probability=0.5,
        error_correction_efficiency=error_correction_efficiency,
    )


def plot_model(
    ax: Axes,
    params: ModelParameters,
    loss_range: tuple[float, float],
    *,
    power_mW: float = 2.5,
    timing_imprecision: float = 0.4e-9,
    dark_count_a: float = 200,
    dark_count_b: float = 70,
    dead_time_a: float = 25e-9,
    dead_time_b: float = 45e-9,
) -> None:
    """Plot the detector-resolved CW-QKD model."""
    losses = np.linspace(*loss_range, 200)

    efficiency_a = float(qtoolkit.dB_to_fraction(params.heralding_1550_db))
    brightness = params.brightness_per_mw * power_mW

    rates = []
    for loss_db in losses:
        efficiency_b = float(
            qtoolkit.dB_to_fraction(params.heralding_780_db + loss_db)
        )
        rates.append(
            model_secure_key_rate(
                brightness=brightness,
                efficiency_a=efficiency_a,
                efficiency_b=efficiency_b,
                coincidence_window=params.coincidence_window,
                timing_imprecision=timing_imprecision,
                dark_count_a=dark_count_a,
                dark_count_b=dark_count_b,
                qber=params.qber,
                qx=params.qx,
                detector_count=4,
                dead_time_a=dead_time_a,
                dead_time_b=dead_time_b,
                error_correction_efficiency=1.1,
            )
        )

    linestyle = '--' if dark_count_b > 250 else '-'
    ax.plot(losses, rates, linestyle=linestyle)


def plot_data(
    ax: Axes,
    directory: typing.Union[pathlib.Path, str],
    dark_count_label: int,
    *,
    power_mw: float = 5.0,
) -> tuple[float, float]:
    """Plot measured SKR points and return their loss range."""
    directory = pathlib.Path(directory)

    rates: defaultdict[float, list[float]] = defaultdict(list)
    losses: defaultdict[float, list[float]] = defaultdict(list)

    for file_path in directory.glob('*.csv'):
        dataruns = parse_data_from_file(file_path)
        counts = CoincidenceCounts.from_dataruns(dataruns)

        power = dataruns[0].power_mW
        losses[power].append(dataruns[0].loss_dB)

        qber = qtoolkit.qkd.qz(
            c_hh=counts.h1550h780,
            c_hv=counts.h1550v780,
            c_vh=counts.v1550h780,
            c_vv=counts.v1550v780,
        )
        qx = qtoolkit.qkd.qx(
            c_dd=counts.d1550d780,
            c_da=counts.d1550a780,
            c_ad=counts.a1550d780,
            c_aa=counts.a1550a780,
        )

        rates[power].append(
            qtoolkit.qkd.cw.secure_key_rate(
                measured_coincidence_rate=counts.total,
                bit_error_rate=qber,
                phase_error_rate=qx,
                sifting_probability=0.5,
                error_correction_efficiency=1.1,
            )
        )

    rate = np.asarray(rates[power_mw])
    loss = np.asarray(losses[power_mw])
    order = np.argsort(loss)
    rate = rate[order]
    loss = loss[order]

    marker = '^' if dark_count_label > 250 else 'o'
    ax.scatter(
        loss,
        rate,
        s=10,
        edgecolors='black',
        linewidths=0.5,
        label=f'~{dark_count_label} DC/s',
        marker=marker,
        zorder=1000,
    )

    return (float(loss.min()), float(loss.max()))


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(3.5, 1.8))

    normal_params = ModelParameters(
        heralding_1550_db=-10 * np.log10(0.2063),
        heralding_780_db=-10 * np.log10(0.158),
        qber=0.039,
        qx=0.017,
        brightness_per_mw=3_826_895.017621633,
        coincidence_window=0.4e-9,
    )
    loss_range = plot_data(
        ax=ax,
        directory='examples/data/normal_dc',
        dark_count_label=250,
    )
    plot_model(
        ax=ax,
        params=normal_params,
        loss_range=loss_range,
        timing_imprecision=0.4e-9,
        dark_count_a=200,
        dark_count_b=250,
    )

    high_dc_params = dataclasses.replace(
        normal_params,
        heralding_1550_db=normal_params.heralding_1550_db + 1,
        heralding_780_db=normal_params.heralding_780_db + 1,
    )
    loss_range = plot_data(
        ax=ax,
        directory='examples/data/high_dc',
        dark_count_label=1400,
    )
    plot_model(
        ax=ax,
        params=high_dc_params,
        loss_range=loss_range,
        timing_imprecision=0.4e-9,
        dark_count_a=200,
        dark_count_b=1400,
    )

    ax.set_yscale('log')
    ax.legend(fontsize=10, loc=(0.015, 0.03), frameon=False)
    ax.set_xlabel('Loss (dB)', fontsize=10)
    ax.set_ylabel('SKR (bits/s)', fontsize=10)

    fig.savefig(
        pathlib.Path(__file__).with_suffix('.png'),
        dpi='figure',
        bbox_inches='tight',
    )
    plt.show()
