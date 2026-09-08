import qtoolkit

# PolarisationChannelMap

def test_polarisation_channel_map_channels() -> None:
    channel_map = qtoolkit.polarisation.PolarisationChannelMap(
        h=0,
        v=1,
        r=4,
        l=5,
    )

    assert channel_map.channels == (
        0,
        1,
        4,
        5,
    )


def test_polarisation_channel_map_ignores_none() -> None:
    channel_map = qtoolkit.polarisation.PolarisationChannelMap(
        h=0,
        v=1,
    )

    assert channel_map.channels == (
        0,
        1,
    )


def test_polarisation_channel_map_as_channel_map() -> None:
    channel_map = qtoolkit.polarisation.PolarisationChannelMap(
        h=0,
        v=1,
        d=2,
        a=3,
    )

    result = channel_map.as_channel_map()

    assert result.channels == {
        'H': 0,
        'V': 1,
        'D': 2,
        'A': 3,
    }