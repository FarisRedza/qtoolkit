import qtoolkit

# ChannelPair

def test_channel_pair_as_tuple() -> None:
    pair = qtoolkit.timetags.ChannelPair(
        first=1,
        second=4,
        name='test',
    )

    assert pair.as_tuple() == (1, 4)

# ChannelMap

def test_channel_map_getitem() -> None:
    channel_map = qtoolkit.timetags.ChannelMap({
        'signal': 1,
        'idler': 5,
    })

    assert channel_map['signal'] == 1

def test_channel_map_get() -> None:
    channel_map = qtoolkit.timetags.ChannelMap({
        'signal': 1,
        'idler': 5,
    })

    assert channel_map.get('idler') == 5


def test_channel_map_get_missing() -> None:
    channel_map = qtoolkit.timetags.ChannelMap({
        'signal': 1,
    })

    assert channel_map.get('missing') is None


def test_channel_map_names() -> None:
    channel_map = qtoolkit.timetags.ChannelMap({
        'signal': 1,
        'idler': 5,
    })

    assert channel_map.names == (
        'signal',
        'idler',
    )


def test_channel_map_numbers() -> None:
    channel_map = qtoolkit.timetags.ChannelMap({
        'signal': 1,
        'idler': 5,
    })

    assert channel_map.numbers == (
        1,
        5,
    )