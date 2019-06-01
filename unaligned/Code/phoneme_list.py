N_STATES = 138
N_PHONEMES = N_STATES // 3
PHONEME_LIST = [
    "+BREATH+",
    "+COUGH+",
    "+NOISE+",
    "+SMACK+",
    "+UH+",
    "+UM+",
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "SIL",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH"
]

PHONEME_MAP = [
    '_',  # "+BREATH+"
    '+',  # "+COUGH+"
    '~',  # "+NOISE+"
    '!',  # "+SMACK+"
    '-',  # "+UH+"
    '@',  # "+UM+"
    'a',  # "AA"
    'A',  # "AE"
    'h',  # "AH"
    'o',  # "AO"
    'w',  # "AW"
    'y',  # "AY"
    'b',  # "B"
    'c',  # "CH"
    'd',  # "D"
    'D',  # "DH"
    'e',  # "EH"
    'r',  # "ER"
    'E',  # "EY"
    'f',  # "F"
    'g',  # "G"
    'H',  # "HH"
    'i',  # "IH"
    'I',  # "IY"
    'j',  # "JH"
    'k',  # "K"
    'l',  # "L"
    'm',  # "M"
    'n',  # "N"
    'G',  # "NG"
    'O',  # "OW"
    'Y',  # "OY"
    'p',  # "P"
    'R',  # "R"
    's',  # "S"
    'S',  # "SH"
    '.',  # "SIL"
    't',  # "T"
    'T',  # "TH"
    'u',  # "UH"
    'U',  # "UW"
    'v',  # "V"
    'W',  # "W"
    '?',  # "Y"
    'z',  # "Z"
    'Z',  # "ZH"
]

# For calculating phoneme prior distribution.
# Used to init the bias of final classification layer.
def calc_phoneme_prior(y_train, y_dev):
    labels = [y_train, y_dev]
    tot_phonemes = [0] * 46     # Excluding blank.
    for l in labels:
        for ph in l:
            tot_phonemes[ph] += 1
    ph_prior_norm = [ph_count/sum(tot_phonemes) for ph_count in tot_phonemes]
    return ph_prior_norm

# Save phoneme prior for given dataset for faster access.
# First one is 0.0 for 'blank'.
PHONEME_PRIOR = [0.0, 0.009827880070567827, 0.0015614202646841511, 0.0024273259507151196, 0.008212153868152158,
                0.00012206380499285802, 0.005086822240722777, 0.01881127789842996, 0.018447577581512466,
                0.10461615417306257, 0.011216418129813112, 0.003980774701603819, 0.013463388580906132,
                0.01665000122063805, 0.004169101715021371, 0.043564821111757135, 0.01907932004980203,
                0.02733980122033912, 0.028599300400020528, 0.016047155489856996, 0.0170027407060868,
                0.006547602143739347, 0.010518412453098851, 0.06145837848448569, 0.035650104352097776,
                0.0055352198917169495, 0.03800817777671491, 0.03651302072045545, 0.029558871373148055,
                0.06839708302381439, 0.007846958892398016, 0.012590507820304143, 0.001790103066691179,
                0.026379732027599374, 0.04327136976016206, 0.047691574159740166, 0.008636636977759975,
                0.03646568985729496, 0.06400178960484708, 0.004748032904416069, 0.0026849054902306608,
                0.01209776862382277, 0.017076975428306947, 0.014234632540616108, 0.00823905772721181,
                0.02912940606741808, 0.0007024896532242033]

assert len(PHONEME_LIST) == len(PHONEME_MAP)
assert len(set(PHONEME_MAP)) == len(PHONEME_MAP)
