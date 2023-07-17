import mido
import pygame
import time
import os
import random
import json
import math
import glob

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

midi_groups = 1


def get_midi_group(x):
    if x < 8:  # piano
        return 0
    elif x < 16:  # Chromatic Percussion
        return -1
    elif x < 24:  # organ
        return -1  # 0
    elif x < 32:  # Guitar
        return -1  # 0
    elif x < 40:  # Bass
        return -1  # 1
    elif x < 56:  # Strings
        return -1  # 3
    elif x < 96:  # Solo
        return -1  # 2
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


def iterate_all_files(func, file_type='.tokens', directory=part_dir, fail_func=lambda x: None, param=None,
                      verbose=False,
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


def get_random_file(dir_name=part_dir, file_type='.tokens', prefix=None):
    if prefix:
        x = glob.glob(dir_name + '\\' + prefix + '*')
    else:
        x = glob.glob(dir_name + '\\*')
    while x:
        i = random.randint(0, len(x) - 1)
        f = x[i]
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
        if channels[k][0][1] != 0:
            channels[k] = [[[], channels[k][0][1]]] + channels[k]
        for i in range(1, len(channels[k])):
            channels[k][i - 1][1] = channels[k][i][1] - channels[k][i - 1][1]
    return channels


def write_midi(channel):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.Message('program_change', program=0, time=0))
    t = 0
    notes = []
    tt = 0
    for group in channel:
        t = max(0, int(round(group[1] * mid.ticks_per_beat)))
        t0 = t
        for i, note in enumerate(group[0]):
            ignore = False
            for j, n in enumerate(notes):
                if note == n[0]:
                    n[1] = tt
                    ignore = True
                    break
            if ignore:
                continue
            track.append(mido.Message('note_on', note=note, velocity=100, time=t0))
            t0 = 0
            notes.append([note, tt])
        tt += t
        i = 0
        while i < len(notes):
            if tt - notes[i][1] >= 2 * mid.ticks_per_beat:
                track.append(mido.Message('note_off', note=notes[i][0], velocity=0, time=0))
                del notes[i]
            else:
                i += 1
        if t0:
            track.append(mido.Message('note_off', note=1, velocity=0, time=t0))
    mid.save(out_dir + '\\new_song.mid')


def get_notes(filename):
    mid = mido.MidiFile(filename)
    cum_ticks = 0
    channels = {}
    ignore_ticks = 0
    tempo = 500000
    for msg in mid:
        ticks = 0
        if msg.time > 0:
            ticks = int(round(mido.second2tick(msg.time, ticks_per_beat=mid.ticks_per_beat, tempo=tempo)))
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        if ticks:
            ignore_ticks += ticks
            if ignore_ticks >= mid.ticks_per_beat / 8:  # 16th notes
                cum_ticks += ignore_ticks
                ignore_ticks = 0
        if msg.type == 'note_on':
            if msg.channel not in channels.keys():
                channels[msg.channel] = []
            if len(channels[msg.channel]) == 0 or channels[msg.channel][-1][1] != cum_ticks:
                channels[msg.channel].append([[msg.note], cum_ticks])
            else:
                channels[msg.channel][-1][0].append(msg.note)
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
        err = 0
        for i, x in enumerate(channels[k]):
            d = (max(0.0, x[1] + err * 1.0 / (len(channels[k]) - i)))
            d = get_best_round(d * 1.0 / ticks_per_beat) * ticks_per_beat
            channels[k][i][1] = d / ticks_per_beat
            err = x[1] - d
    return channels


def fix_values(channels):
    for k in channels:
        silent = False
        j = 0
        while j < len(channels[k]):
            s = set()
            for x in channels[k][j][0]:
                s.add(x)
            x = list(s)
            if not x:
                if silent:
                    channels[k][j - 1][1] += channels[k][j][1]
                    del channels[k][j]
                    continue
                silent = True
            else:
                silent = False
            x.sort()
            channels[k][j][0] = x
            j += 1
    return channels


def get_data(f):
    inst = get_instruments(f)
    if len(list(inst)) != 1 or list(inst)[0] >= 8:
        return None, None
    ch, tpb = get_notes(f)
    ch = convert_to_lengths(ch)
    ch = fix_timings(ch, tpb)
    ch = fix_values(ch)
    return ch, inst


def serialize_file(f):
    ch, inst = get_data(f)
    if ch is not None:
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
    return inst, channels


def fix_json(f):
    inst, channels = json(f)
    channels = fix_values(channels)
    write_json(os.path.splitext(f)[0], inst, channels)


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

    return tokens


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
        token_channels[k] = get_tokens(token_maps[midi_group], channels[k])

    return token_channels, token_instruments


def detokenize(tokens):
    global token_maps
    global token_maps_round
    if not token_maps:
        token_maps = get_token_maps(token_maps_round)
    channel = []
    for t in tokens:
        channel += parse_str_key(list(token_maps[0].keys())[list(token_maps[0].values()).index(int(t))])
    return channel


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
    #channels, inst = get_data(f)
    #if channels is None:
    #    return
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


def iterate_counts(rounds, prefix='0'):
    token_map_file = None
    token_map_index = 0
    for i in reversed(range(rounds)):
        if os.path.isfile("{0}\\token_map_{1}.json".format(out_dir, i)):
            token_map_file = "{0}\\token_map_{1}.json".format(out_dir, i)
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
        iterate_all_files(count_file, directory=part_dir, file_type='.json', param=ms, prefix=prefix)
        e = []
        y = []
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
        token_map_file = "{0}\\token_map_{1}.json".format(out_dir, i)
        entropy_file = "{0}\\entropy_{1}.json".format(out_dir, i)
        m = []
        for j in range(midi_groups):
            m.append(ms[j][0])
        print(e, len(y))
        with open(token_map_file, 'w') as outfile:
            outfile.write(json.dumps(m))
        with open(entropy_file, 'w') as outfile:
            outfile.write(json.dumps(e))


if __name__ == '__main__':
    token_maps = None
    token_maps_round = 99
    iterate_all_files(serialize_file, file_type='.mid')
    iterate_counts(token_maps_round + 1)
    iterate_all_files(tokenize_json, file_type='.json')
    # f = get_random_file(prefix='000', file_type='.tokens')
    # print(f)
    #f = 'C:\\lmd_full\\0\\000902bf7e1e85540a5e8d864196748f.tokens'
    #f = 'C:\\lmd_full\\0\\00012722c199ae2a628ebb792ccc617a.tokens'
    #inst, tokens = load_tokens(f)
    # for k in ch:
    #    print(ch[k])
    #channels = {}
    #for k in tokens:
    #    channels[k] = detokenize(tokens[k])
    #    print(channels[k])
    #    write_midi(channels[k])
    #    break
    #play_midi(out_dir + '\\new_song.mid')
    #play_midi(os.path.splitext(f)[0] + '.mid')
