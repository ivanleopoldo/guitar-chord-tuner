import numpy as np

frequency_array = np.array(
    [
        32.7,
        34.65,
        36.71,
        38.89,
        41.2,
        43.65,
        46.25,
        49.0,
        51.91,
        55.0,
        58.27,
        61.74,
        65.41,
        69.3,
        73.42,
        77.78,
        82.41,
        87.31,
        92.5,
        98.0,
        103.83,
        110.0,
        116.54,
        123.47,
        130.81,
        138.59,
        146.83,
        155.56,
        164.81,
        174.61,
        185.0,
        196.0,
        207.65,
        220.0,
        233.08,
        246.94,
        261.63,
        277.18,
        293.66,
        311.13,
        329.63,
        349.23,
        369.99,
        392.0,
        415.3,
        440.0,
        466.16,
        493.88,
        523.25,
        554.37,
        587.33,
        622.25,
        659.25,
        698.46,
        739.99,
        783.99,
        830.61,
        880.0,
        932.33,
        987.77,
        1046.5,
        1108.73,
        1174.66,
        1244.51,
        1318.51,
        1396.91,
        1479.98,
        1567.98,
        1661.22,
        1760.0,
        1864.66,
        1975.53,
        2093.0,
    ]
)

frequency_to_note = {
    32.7: "C1",
    34.65: "#C1",
    36.71: "D1",
    38.89: "#D1",
    41.2: "E1",
    43.65: "F1",
    46.25: "#F1",
    49.0: "G1",
    51.91: "#G1",
    55.0: "A1",
    58.27: "#A1",
    61.74: "B1",
    65.41: "C2",
    69.3: "#C2",
    73.42: "D2",
    77.78: "#D2",
    82.41: "E2",
    87.31: "F2",
    92.5: "#F2",
    98.0: "G2",
    103.83: "#G2",
    110.0: "A2",
    116.54: "#A2",
    123.47: "B2",
    130.81: "C3",
    138.59: "#C3",
    146.83: "D3",
    155.56: "#D3",
    164.81: "E3",
    174.61: "F3",
    185.0: "#F3",
    196.0: "G3",
    207.65: "#G3",
    220.0: "A3",
    233.08: "#A3",
    246.94: "B3",
    261.63: "C4",
    277.18: "#C4",
    293.66: "D4",
    311.13: "#D4",
    329.63: "E4",
    349.23: "F4",
    369.99: "#F4",
    392.0: "G4",
    415.3: "#G4",
    440.0: "A4",
    466.16: "#A4",
    493.88: "B4",
    523.25: "C5",
    554.37: "#C5",
    587.33: "D5",
    622.25: "#D5",
    659.25: "E5",
    698.46: "F5",
    739.99: "#F5",
    783.99: "G5",
    830.61: "#G5",
    880.0: "A5",
    932.33: "#A5",
    987.77: "B5",
    1046.5: "C6",
    1108.73: "#C6",
    1174.66: "D6",
    1244.51: "#D6",
    1318.51: "E6",
    1396.91: "F6",
    1479.98: "#F6",
    1567.98: "G6",
    1661.22: "#G6",
    1760.0: "A6",
    1864.66: "#A6",
    1975.53: "B6",
    2093.0: "C7",
}

chord_template = {
    "C": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "D": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    "E": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    "F": [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    "G": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    "A": [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "B": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    "Bmin": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    "Bb": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    "Dm": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    "Em": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "Fm": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "Am": [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "Bbm": [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    "NC": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}
