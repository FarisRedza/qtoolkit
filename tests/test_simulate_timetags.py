import qtoolkit
import numpy as np


def test_generate_timetags(tmp_path):
    data = qtoolkit.generate_timetags(
        channel_rates={
            0: 50_000,
            1: 45_000,
            4: 55_000,
            5: 48_000,
        },
        coincidence_pairs=[
            qtoolkit.CoincidencePair(
                channel_a=0,
                channel_b=4,
                rate_hz=5_000,
                delay_ps=300,
                jitter_ps=50,
            ),
        ],
        duration_s=1.0,
        rng=np.random.default_rng(42),
    )

    assert len(data) > 0
    assert np.all(np.diff(data.timetags) >= 0)

    file_path = tmp_path.joinpath('timetags.txt')
    data.to_file(file_path)

    loaded = qtoolkit.TimetagData.from_file(file_path)

    np.testing.assert_array_equal(
        loaded.timetags,
        data.timetags,
    )
    np.testing.assert_array_equal(
        loaded.channels,
        data.channels,
    )