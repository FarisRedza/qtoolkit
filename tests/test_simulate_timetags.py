import numpy as np
import pytest

import qtoolkit


def test_generate_timetags(tmp_path) -> None:
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

def test_generate_timetags_zero_duration() -> None:
    with pytest.raises(
        ValueError,
        match='duration_s must be positive.',
    ):
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
            duration_s=0.0,
            rng=np.random.default_rng(42),
        )

def test_generate_timetags_negative_duration() -> None:
    with pytest.raises(
        ValueError,
        match='duration_s must be positive.',
    ):
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
            duration_s=-1.0,
            rng=np.random.default_rng(42),
        )

def test_generate_timetags_default_rng() -> None:
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
    )

def test_generate_timetags_negative_channel_rate() -> None:
    with pytest.raises(
        ValueError,
        match='must be non-negative.',
    ):
        data = qtoolkit.generate_timetags(
            channel_rates={
                0: -50_000,
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

def test_generate_timetags_negative_coincidence_rate() -> None:
    with pytest.raises(
        ValueError,
        match='Coincidence rates must be non-negative.',
    ):
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
                    rate_hz=-5_000,
                    delay_ps=300,
                    jitter_ps=50,
                ),
            ],
            duration_s=1.0,
            rng=np.random.default_rng(42),
        )

def test_generate_timetags_negative_jitter() -> None:
    with pytest.raises(
        ValueError,
        match='jitter_ps must be non-negative.',
    ):
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
                    jitter_ps=-50,
                ),
            ],
            duration_s=1.0,
            rng=np.random.default_rng(42),
        )

@pytest.mark.parametrize(
    'channel_rates',
    [
        {
            1: 45_000,
            4: 55_000,
            5: 48_000,
        },
        {
            0: 50_000,
            1: 45_000,
            5: 48_000,
        },
    ],
)
def test_generate_timetags_no_singles_rate(
        channel_rates: dict
) -> None:
    with pytest.raises(
        ValueError,
        match='has no singles rate.',
    ):
        data = qtoolkit.generate_timetags(
            channel_rates=channel_rates,
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

def test_generate_timetags_cc_rate_greater_than_singles(
        tmp_path
    ) -> None:
   with pytest.raises(
        ValueError,
        match='exceeds its singles rate',
    ):
        data = qtoolkit.generate_timetags(
            channel_rates={
                0: 4_000,
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

def test_generate_timetags_no_timetags() -> None:
    data = qtoolkit.generate_timetags(
        channel_rates={},
        coincidence_pairs=[],
        duration_s=1.0,
        rng=np.random.default_rng(42),
    )

    assert len(data) == 0