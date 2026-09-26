"""Master edit: scene order and durations (seconds). Bar length at 96 BPM = 2.5 s."""
from scenes_a import ColdOpen, FlirTeaser, Hook, MilkyWay, Plutarch, YangzhouPearl, Nuremberg
from scenes_b import Airship, LosAngeles1942, Arnold1947, Roswell, BlueBook, Radar1952, Seti, WowSignal
from scenes_c import (Montage, Rendlesham, BelgianTriangle, PhoenixLights, Hangzhou, NavyFlir, NewsMontage, Hearing,
                      Hellfire)
from scenes_d import Interstellar, Mars, Recap, Title, EndCard

SCENES = [
    ColdOpen(),        # 0.0
    FlirTeaser(),      # 5.0
    Hook(),            # 12.5
    MilkyWay(),        # 20.0
    Plutarch(),        # 30.0
    YangzhouPearl(),   # 40.0
    Nuremberg(),       # 52.5
    Airship(),         # 62.5
    LosAngeles1942(),  # 72.5
    Arnold1947(),      # 82.5
    Roswell(),         # 92.5
    BlueBook(),        # 102.5
    Radar1952(),       # 115.0
    Seti(),            # 125.0
    WowSignal(),       # 137.5
    Montage(),         # 147.5
    Rendlesham(),      # 165.0
    BelgianTriangle(), # 175.0
    PhoenixLights(),   # 182.5
    Hangzhou(),        # 190.0
    NavyFlir(),        # 200.0
    NewsMontage(),     # 212.5
    Hearing(),         # 222.5
    Hellfire(),        # 235.0
    Interstellar(),    # 242.5
    Mars(),            # 252.5
    Recap(),           # 260.0
    Title(),           # 267.5
    EndCard(),         # 277.5
]

TIMELINE = []
_t = 0.0
for _s in SCENES:
    TIMELINE.append((_t, _s))
    _t += _s.dur
TOTAL = _t
