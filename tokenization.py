import math
from common import *

midi_groups = 1


def join_str_keys(x, y):
    return ','.join([x, y])


def get_str_key(z, note_diff):
    return ','.join([':'.join(str(note + note_diff) for note in x[0]) + '/' + '{0:g}'.format(x[1]) for x in z])


def parse_str_key(x):
    sequence = []
    for y in x.split(','):
        z = y.split('/')
        if y[0] != '/':
            notes = [int(note) for note in z[0].split(':')]
            length = float(z[1])
        else:
            notes = []
            length = float(z[1])
        sequence.append([notes, length])
    return sequence


def get_token_maps(round):
    token_map_file = "{0}/token_map_{1}.json".format(out_dir, round)
    if not os.path.isfile(token_map_file):
        return None
    return json.load(open(token_map_file, 'r'))


def detokenize(tokens):
    global token_maps
    global token_maps_round
    if not token_maps:
        token_maps = get_token_maps(token_maps_round)
    channel = []
    for i, t in enumerate(tokens):
        if t != silence_token:
            channel += parse_str_key(list(token_maps[0].keys())[list(token_maps[0].values()).index(int(t))])
    return channel


def tokenize(inst, channels):
    global token_maps
    global token_maps_round
    if not token_maps:
        token_maps = get_token_maps(token_maps_round)
    token_channels = {}
    token_instruments = {}
    for k in channels:
        if k not in inst:
            continue
        midi_group = get_midi_group(inst[k][1], False)
        if midi_group < 0:
            continue
        token_instruments[k] = midi_group
        token_channels[k] = get_tokens(token_maps[midi_group], channels[k])

    return token_channels, token_instruments


def calc_entropy(m):
    s = float(sum(list(m.values())))
    p = [x / s for x in list(m.values())]
    return sum([-math.log(x) * x for x in p])


def get_tokens(m, bar, note_diff=0, initialize=False):
    tokens = []
    i = 0
    best_x = None
    x = None
    while i < len(bar):
        if x is None:
            x = get_str_key([bar[i]], note_diff)
            i += 1
        if x in m:
            if i < len(bar):
                best_x = x
                x = join_str_keys(x, get_str_key([bar[i]], note_diff))
                i += 1
            else:
                tokens.append(m[x])
                best_x = None
        elif initialize and not best_x:
            m[x] = len(m)
            tokens.append(m[x])
            x = None
        else:
            tokens.append(m[best_x])
            best_x = None
            x = None
            i -= 1

    return tokens


def get_pairs_hist(bars, ms, note_diff):
    mm = ms[1]
    m = ms[0]
    initialize = ms[2]
    tokens = get_tokens(m, bars, note_diff, initialize)
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        if pair in mm:
            mm[pair] += 1
        else:
            mm[pair] = 1
    return [m, mm, initialize]


def count_file(f, ms):
    inst, channels = load_json(f)
    for k in channels:
        if k not in inst:
            continue
        midi_group = get_midi_group(inst[k][1], False)
        if midi_group < 0:
            continue
        for note_diff in range(-6, 6):
            ms[midi_group] = get_pairs_hist(channels[k], ms[midi_group], note_diff)


def iterate_counts(rounds):
    token_map_file = None
    token_map_index = 0
    for i in reversed(range(rounds)):
        if os.path.isfile(os.path.join(out_dir, 'token_map_{0}.json'.format(i))):
            token_map_file = os.path.join(out_dir, "token_map_{0}.json".format(i))
            token_map_index = i + 1
            break
    ms = []
    for i in range(midi_groups):
        ms.append([{}, {}, True])
    if token_map_file:
        m = json.load(open(token_map_file, 'r'))
        for i in range(midi_groups):
            ms[i][0] = m[i]
            ms[i][2] = False

    for i in range(token_map_index, rounds):
        iterate_all_files(count_file, file_type='.notes', param=ms, run_in_threads=False)
        e = []
        for j in range(midi_groups):
            e.append(calc_entropy(ms[j][1]))
            y = [[ms[j][1][x], x] for x in ms[j][1]]
            if y:
                y.sort(reverse=True)
                k = 0
                while k < min(48, len(y)):
                    k1 = list(ms[j][0].keys())[list(ms[j][0].values()).index(y[k][1][0])]
                    k2 = list(ms[j][0].keys())[list(ms[j][0].values()).index(y[k][1][1])]
                    k3 = join_str_keys(k1, k2)
                    if k3 in ms[j][0]:
                        del y[k]
                        continue
                    ms[j][0][k3] = len(ms[j][0])
                    k += 1
            else:
                print('error instrument', j)
            ms[j][1] = {}
            ms[j][2] = False
        token_map_file = os.path.join(out_dir, "token_map_{0}.json".format(i))
        entropy_file = os.path.join("entropy_{0}.json".format(i))
        m = []
        for j in range(midi_groups):
            m.append(ms[j][0])
        print(i, e, len(m[0]))
        with open(token_map_file, 'w') as outfile:
            outfile.write(json.dumps(m))
        with open(entropy_file, 'w') as outfile:
            outfile.write(json.dumps(e))


token_maps_round = 222
token_maps = get_token_maps(token_maps_round)
vocab_size = len(token_maps)
silence_token = token_maps(',/4')
