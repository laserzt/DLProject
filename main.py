import mido
import pygame
import time
import os
import random
import json
import math
import glob
import copy

instruments = ['Acoustic Grand Piano', 'Bright Acoustic Piano', 'Electric Grand Piano', 'Honky-tonk Piano',
               'Electric Piano 1', 'Electric Piano 2', 'Harpsichord', 'Clavinet', 'Celesta', 'Glockenspiel',
               'Music Box', 'Vibraphone', 'Marimba', 'Xylophone', 'Tubular Bells', 'Dulcimer', 'Drawbar Organ',
               'Percussive Organ', 'Rock Organ', 'Church Organ', 'Reed Organ', 'Accordion', 'Harmonica',
               'Tango Accordion', 'Acoustic Guitar (nylon)', 'Acoustic Guitar (steel)', 'Electric Guitar (jazz)',
               'Electric Guitar (clean)', 'Electric Guitar (muted)', 'Overdriven Guitar', 'Distortion Guitar',
               'Guitar harmonics', 'Acoustic Bass', 'Electric Bass (finger)', 'Electric Bass (pick)',
               'Fretless Bass', 'Slap Bass 1', 'Slap Bass 2', 'Synth Bass 1', 'Synth Bass 2', 'Violin', 'Viola',
               'Cello', 'Contrabass', 'Tremolo Strings', 'Pizzicato Strings', 'Orchestral Harp', 'Timpani',
               'String Ensemble 1', 'String Ensemble 2', 'Synth Strings 1', 'Synth Strings 2', 'Choir Aahs',
               'Voice Oohs', 'Synth Voice', 'Orchestra Hit', 'Trumpet', 'Trombone', 'Tuba', 'Muted Trumpet',
               'French Horn', 'Brass Section', 'Synth Brass 1', 'Synth Brass 2', 'Soprano Sax', 'Alto Sax',
               'Tenor Sax', 'Baritone Sax', 'Oboe', 'English Horn', 'Bassoon', 'Clarinet', 'Piccolo', 'Flute',
               'Recorder', 'Pan Flute', 'Blown Bottle', 'Shakuhachi', 'Whistle', 'Ocarina', 'Lead 1 (square)',
               'Lead 2 (sawtooth)', 'Lead 3 (calliope)', 'Lead 4 (chiff)', 'Lead 5 (charang)', 'Lead 6 (voice)',
               'Lead 7 (fifths)', 'Lead 8 (bass + lead)', 'Pad 1 (new age)', 'Pad 2 (warm)', 'Pad 3 (polysynth)',
               'Pad 4 (choir)', 'Pad 5 (bowed)', 'Pad 6 (metallic)', 'Pad 7 (halo)', 'Pad 8 (sweep)', 'FX 1 (rain)',
               'FX 2 (soundtrack)', 'FX 3 (crystal)', 'FX 4 (atmosphere)', 'FX 5 (brightness)', 'FX 6 (goblins)',
               'FX 7 (echoes)', 'FX 8 (sci-fi)', 'Sitar', 'Banjo', 'Shamisen', 'Koto', 'Kalimba', 'Bag pipe',
               'Fiddle', 'Shanai', 'Tinkle Bell', 'Agogo', 'Steel Drums', 'Woodblock', 'Taiko Drum', 'Melodic Tom',
               'Synth Drum', 'Reverse Cymbal', 'Guitar Fret Noise', 'Breath Noise', 'Seashore', 'Bird Tweet',
               'Telephone Ring', 'Helicopter', 'Applause', 'Gunshot']


def get_midi_group(x):
    if x < 8:  # piano
        return 0
    elif x < 16:  # Chromatic Percussion
        return -1
    elif x < 24:  # organ
        return 0
    elif x < 32:  # Guitar
        return 0
    elif x < 40:  # Bass
        return 1
    elif x < 56:  # Strings
        return 3
    elif x < 96:  # Solo
        return 2
    elif x < 104:  # effects
        return -1
    elif x < 119:  # percussion
        return -1
    else:  # effects
        return -1


full_dir = 'C:\\lmd_full'
matched_dir = 'C:\\lmd_matched'
part_dir = 'C:\\lmd_full\\0'

out_dir = 'C:\\Users\\Eleizerovich\\OneDrive - Gita Technologies LTD\\Desktop\\School\\DLProjectData'


def iterate_all_files(func, file_type='.json', directory=part_dir, fail_func=lambda x: None, param=None, verbose=False,
                      prefix=None):
    if prefix:
        y = glob.glob(directory + '\\' + prefix + '*')
    else:
        y = glob.glob(directory + '\\*')
    for f in y:
        if os.path.isfile(f):
            if os.path.splitext(f)[1] == file_type:
                try:
                    if verbose:
                        print('Running on file', f)
                    if param:
                        func(f, param)
                    else:
                        func(f)
                except Exception as e:
                    print('Failed on file', f, type(e), e)
                    fail_func(f)
                    raise e
        else:
            iterate_all_files(func, file_type, f, fail_func)


def get_random_file(dir_name=part_dir, file_type='.json'):
    x = os.listdir(dir_name)
    while x:
        i = random.randint(0, len(x) - 1)
        f = dir_name + '\\' + x[i]
        if os.path.isfile(f):
            if os.path.splitext(f)[1] == file_type:
                return f
        else:
            y = get_random_file(f)
            if y:
                return y
        del x[i]


mixer_init = False


def play_midi(filename):
    global mixer_init
    if not mixer_init:
        mixer_init = True
        pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.stop()


def get_instruments(filename):
    mid = mido.MidiFile(filename)
    res = {}
    for msg in mid:
        if msg.type == 'program_change':
            res[msg.channel] = (instruments[msg.program], msg.program)
    return res


def convert_to_lengths(channels):
    for k in channels.keys():
        for j, bar in enumerate(channels[k]):
            beats_per_bar = bar[0]
            new_bar = [beats_per_bar]
            if len(bar) > 1 and bar[1][1] > 0:
                new_bar.append([[], bar[1][1]])
            else:
                new_bar.append([[], beats_per_bar])
                channels[k][j] = new_bar
                continue
            for i in range(1, len(bar) - 1):
                new_bar.append([bar[i][0], bar[i + 1][1] - bar[i][1]])
            new_bar.append([bar[-1][0], beats_per_bar - sum([x[1] for x in new_bar[1:]])])
            channels[k][j] = new_bar
    return channels


def get_notes(filename):
    mid = mido.MidiFile(filename)
    tempo = 500000
    ticks_per_bar = 4 * mid.ticks_per_beat
    next_ticks_per_bar = ticks_per_bar
    cum_ticks = 0
    channels = {}
    ticks_in_bar = 0
    bar = 0
    last_bar = 0
    bar_offset = 0
    ignore_ticks = 0
    for msg in mid:
        ticks = 0
        if msg.time > 0:
            ticks = int(round(mido.second2tick(msg.time, ticks_per_beat=mid.ticks_per_beat, tempo=tempo)))
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        if msg.type == 'time_signature':
            next_ticks_per_bar = msg.numerator * 4.0 / msg.denominator * mid.ticks_per_beat
        if ticks:
            ignore_ticks += ticks
            if ignore_ticks >= mid.ticks_per_beat / 8.0:  # 32th notes
                cum_ticks += ignore_ticks
                ignore_ticks = 0
                ticks_in_bar = cum_ticks % ticks_per_bar
                bar = bar_offset + int(cum_ticks / ticks_per_bar)
                if bar > last_bar and next_ticks_per_bar != ticks_per_bar:
                    bar_offset += int(cum_ticks / ticks_per_bar)
                    cum_ticks = cum_ticks % ticks_per_bar
                    ticks_per_bar = next_ticks_per_bar
                last_bar = bar
        if msg.type == 'note_on':
            if msg.channel not in channels.keys():
                channels[msg.channel] = []
            if len(channels[msg.channel]) <= bar:
                for i in range(1 + bar - len(channels[msg.channel])):
                    channels[msg.channel].append([round(ticks_per_bar)])
            if len(channels[msg.channel][bar]) <= 1 or channels[msg.channel][bar][-1][1] != ticks_in_bar:
                channels[msg.channel][bar].append([[msg.note], ticks_in_bar])
            elif channels[msg.channel][bar][-1][1] == ticks_in_bar:
                channels[msg.channel][bar][-1][0].append(msg.note)
    return channels, mid.ticks_per_beat


def check_midi_file(filename):
    mido.MidiFile(filename)


def get_best_round(x):
    y1 = math.ceil(x * 8) / 8  # 32th notes
    y2 = math.ceil(x * 6) / 6  # 8th triplets
    if abs(x - y1) < abs(x - y2):
        return y1
    return y2


def fix_timings(channels, ticks_per_beat):
    for k in channels:
        for j, bar in enumerate(channels[k]):
            err = 0
            s = 0
            for i, x in enumerate(bar[1:]):
                d = (max(0.0, x[1] + err * 1.0 / (len(bar[1:]) - i)))
                if i < len(bar[1:]) - 1:
                    d = get_best_round(d * 1.0 / ticks_per_beat) * ticks_per_beat
                else:
                    d = bar[0] - s
                channels[k][j][i + 1][1] = d / ticks_per_beat
                s += d
                err = x[1] - d
            del channels[k][j][0]
    return channels


def fix_values(channels):
    for k in channels:
        for i in range(len(channels[k])):
            silent = False
            j = 0
            while j < len(channels[k][i]):
                s = set()
                for x in channels[k][i][j][0]:
                    s.add(x)
                x = list(s)
                if not x:
                    if silent:
                        channels[k][i][j - 1][1] += channels[k][i][j][1]
                        del channels[k][i][j]
                        continue
                    silent = True
                else:
                    silent = False
                x.sort()
                channels[k][i][j][0] = x
                j += 1
    return channels


def get_data(f):
    inst = get_instruments(f)
    ch, tpb = get_notes(f)
    ch = convert_to_lengths(ch)
    ch = fix_timings(ch, tpb)
    ch = fix_values(ch)
    return ch, inst


def serialize_file(f):
    ch, inst = get_data(f)
    write_json(os.path.splitext(f)[0], inst, ch)


def write_tokens(f, inst, channels):
    song = {'Instruments': inst, 'Tokens': channels}
    json_object = json.dumps(song)
    with open("{0}.tokens".format(f), "w") as outfile:
        outfile.write(json_object)


def load_tokens(f):
    data = json.load(open(f, 'r'))
    inst = data['Instruments']
    channels = data['Tokens']
    return inst, channels


def tokenize_json(f):
    inst, channels = load_json(f)
    token_channels, token_instruments = tokenize(inst, channels)
    write_tokens(os.path.splitext(f)[0], token_instruments, token_channels)


def write_json(f, inst, channels):
    song = {'Instruments': inst, 'Notes': channels}
    json_object = json.dumps(song)
    with open("{0}.json".format(f), "w") as outfile:
        outfile.write(json_object)


def load_json(f):
    data = json.load(open(f, 'r'))
    inst = data['Instruments']
    channels = data['Notes']
    channels = fix_values(channels)
    return inst, channels


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


def join_str_keys(x, y):
    return ','.join([x, y])


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

    return tokens, m


def get_token_maps(round):
    token_map_file = "{0}\\token_map_{1}.json".format(out_dir, round)
    if not os.path.isfile(token_map_file):
        return None
    return json.load(open(token_map_file, 'r'))


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
        midi_group = get_midi_group(inst[k][1])
        if midi_group < 0:
            continue
        token_instruments[k] = midi_group
        for bar in channels[k]:
            token_channels[k].append(get_tokens(token_maps[midi_group], bar))
    return token_channels, token_instruments


def get_pairs_hist(channel, ms, note_diff):
    mm = ms[1]
    m = ms[0]
    initialize = ms[2]
    for bar in channel:
        tokens, m = get_tokens(m, bar, note_diff, initialize)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in mm:
                mm[pair] += 1
            else:
                mm[pair] = 1
    return ms


def count_file(f, ms):
    inst, channels = load_json(f)
    for k in channels:
        if k not in inst:
            continue
        midi_group = get_midi_group(inst[k][1])
        if midi_group < 0:
            continue
        for note_diff in range(-6, 6):
            ms[midi_group] = get_pairs_hist(channels[k], ms[midi_group], note_diff)


def calc_entropy(m):
    s = float(sum(list(m.values())))
    p = [x / s for x in list(m.values())]
    return sum([-math.log(x) * x for x in p])


def iterate_counts(rounds):
    token_map_file = None
    token_map_index = 0
    for i in reversed(range(rounds)):
        if os.path.isfile("{0}\\token_map_{1}.json".format(out_dir, i)):
            token_map_file = "{0}\\token_map_{1}.json".format(out_dir, i)
            token_map_index = i + 1
            break
    ms = []
    for i in range(4):
        ms.append([{}, {}, True])
    if token_map_file:
        m = json.load(open(token_map_file, 'r'))
        for i in range(4):
            ms[i][0] = m[i]
            ms[i][2] = False

    for i in range(token_map_index, rounds):
        iterate_all_files(count_file, param=ms)
        e = []
        for j in range(4):
            e.append(calc_entropy(ms[j][1]))
        for j in range(4):
            y = [[ms[j][1][x], x] for x in ms[j][1]]
            if y:
                y.sort(reverse=True)
                for k in range(min(48, len(y))):
                    k1 = list(ms[j][0].keys())[list(ms[j][0].values()).index(y[k][1][0])]
                    k2 = list(ms[j][0].keys())[list(ms[j][0].values()).index(y[k][1][1])]
                    ms[j][0][join_str_keys(k1, k2)] = len(ms[j][0])
            else:
                print('error instrument', j)
            ms[j][1] = {}
            ms[j][2] = False
        token_map_file = "{0}\\token_map_{1}.json".format(out_dir, i)
        entropy_file = "{0}\\entropy_{1}.json".format(out_dir, i)
        m = []
        for j in range(4):
            m.append(ms[j][0])
        print(e)
        with open(token_map_file, 'w') as outfile:
            outfile.write(json.dumps(m))
        with open(entropy_file, 'w') as outfile:
            outfile.write(json.dumps(e))


if __name__ == '__main__':
    token_maps = None
    token_maps_round = 99
    iterate_counts(token_maps_round + 1)
