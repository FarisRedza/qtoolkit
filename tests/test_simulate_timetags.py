import qtoolkit
import numpy as np

test_data = qtoolkit.generate_timetags(
    channel_rates={
        0: 50_000,
        1: 45_000,
        2: 40_000,
        3: 42_000,
        4: 55_000,
        5: 48_000,
        6: 43_000,
        7: 41_000,
    },
    coincidence_pairs=[
        qtoolkit.simulate_timetags.CoincidencePair(
            channel_a=0,
            channel_b=4,
            rate_hz=5_000,
            delay_ps=300,
            jitter_ps=50,
        ),
        qtoolkit.simulate_timetags.CoincidencePair(
            channel_a=1,
            channel_b=5,
            rate_hz=5_000,
            delay_ps=300,
            jitter_ps=50,
        ),
    ],
    duration_s=1.0,
    rng=np.random.default_rng(42)
)

test_data.to_file(file_path='test.txt')

proc_data = qtoolkit.ProcessedTimetagData.from_file(
    file_path='test.txt',
    pairs=[
        qtoolkit.ChannelPair(first=0,second=1),
        qtoolkit.ChannelPair(first=0,second=2),
        qtoolkit.ChannelPair(first=0,second=3),
        qtoolkit.ChannelPair(first=0,second=4),
        qtoolkit.ChannelPair(first=0,second=5),
        qtoolkit.ChannelPair(first=1,second=5)
    ],
    coincidence_window=500
)
print(proc_data)