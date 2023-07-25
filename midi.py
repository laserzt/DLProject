import pygame
import time
from tokenization import *
from common import *

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
            res[msg.channel] = (midi_instruments[msg.program], msg.program)
    return res


def convert_to_lengths(channels):
    for k in channels.keys():
        if channels[k][0][1] != 0:
            channels[k] = [[[], channels[k][0][1]]] + channels[k]
        for i in range(1, len(channels[k])):
            channels[k][i - 1][1] = channels[k][i][1] - channels[k][i - 1][1]
    return channels


def get_notes_map(filename, channels_to_get):
    mid = mido.MidiFile(filename)
    total_ticks = 0
    channels = []
    channels_index = {}
    tempo = 500000
    mid_len = int(
        math.ceil(
            mido.second2tick(mid.length, ticks_per_beat=mid.ticks_per_beat, tempo=tempo)) / mid.ticks_per_beat * 16)
    for i in range(len(channels_to_get)):
        channels.append([])
        for j in range(mid_len):
            channels[-1].append([0] * 128)
        channels_index[channels_to_get[i]] = i
    cur_place = 0
    mono_channels = []
    for msg in mid:
        ticks = 0
        if msg.time > 0:
            ticks = int(round(mido.second2tick(msg.time, ticks_per_beat=mid.ticks_per_beat, tempo=tempo)))
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        if ticks:
            total_ticks += ticks
            while total_ticks >= mid.ticks_per_beat / 8:
                cur_place += 1
                if mid_len <= cur_place:
                    for ch in range(len(channels)):
                        for j in range(8):
                            channels[ch].append([0] * 128)
                    mid_len += 8
                for note in range(128):
                    for ch in range(len(channels)):
                        channels[ch][cur_place][note] = channels[ch][cur_place - 1][note]
                total_ticks -= mid.ticks_per_beat / 8
        if not (msg.type == 'note_on' or msg.type == 'note_off' or msg.type == 'control_change'):
            continue
        if msg.channel in channels_index:
            channel = channels_index[msg.channel]
            if msg.type == 'note_on':
                if channel in mono_channels:
                    for note in range(128):
                        channels[channel][cur_place][note] = 0
                channels[channel][cur_place][msg.note] = 1
            if msg.type == 'note_off':
                channels[channel][cur_place][msg.note] = 0
            if msg.type == 'control_change' and msg.control == 93:
                mono_channels.append(channel)
        if msg.type == 'control_change' and msg.control >= 90:
            for ch in range(len(channels)):
                for note in range(128):
                    channels[ch][cur_place][note] = 0
    for ch in range(len(channels)):
        for i in range(mid_len - 1, cur_place, -1):
            del channels[ch][i]
    return channels, mid.ticks_per_beat


def get_notes(filename, channels):
    m, ticks_per_beat = get_notes_map(filename, channels, False)
    if len(m) > 1:
        channel = []
        for i in range(len(m[0])):
            channel.append([0] * 128)
        for x in m:
            for i in range(len(x)):
                for j in range(128):
                    channel[i][j] = channel[i][j] | x[i][j]
    elif len(m) == 1:
        channel = m[0]
    else:
        return []

    t = 0
    last_n = None
    res = {}
    res[0] = []
    for i in range(len(channel)):
        n = []
        for j in range(128):
            if channel[i][j]:
                n.append(j)
        if last_n == n:
            t += ticks_per_beat / 8
        elif last_n is not None:
            res[0].append([last_n, t])
            t = ticks_per_beat / 8
        last_n = n
    return res, ticks_per_beat


def get_best_round(x):
    y1 = round(x * 8) / 8  # 32th notes
    y2 = round(x * 6) / 6  # 8th triplets
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
            if channels[k][j][1] == 0:
                del channels[k][j]
                continue
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
    instr = get_instruments(f)
    if len(instr) > 2:
        return [], []
    instruments = {}
    channels = []
    single_instr = -1
    for k in instr:
        if single_instr >= 0 and single_instr != instr[k][1]:
            return [], []
        single_instr = instr[k][1]
        instruments[k] = instr[k]
        midi_group = get_midi_group(instr[k][1], False)
        if midi_group >= 0:
            channels.append(k)
    if channels:
        channels, ticks_per_beat = get_notes(f, channels)
        channels = fix_timings(channels, ticks_per_beat)
        channels = fix_values(channels)
        return channels, instruments
    else:
        return [], []


def serialize_file(f):
    ch, inst = get_data(f)
    if ch:
        write_json(os.path.splitext(f)[0], inst, ch, '.notes')


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


def write_json(f, inst, channels, file_type='.json'):
    song = {'Instruments': inst, 'Notes': channels}
    json_object = json.dumps(song)
    json_bytes = json_object.encode('utf-8')

    with gzip.open(f + file_type, "w") as outfile:
        outfile.write(json_bytes)


def read_notes_map(f):
    instr = get_instruments(f)
    channels = []
    real_instruments = []
    for k in instr:
        midi_group = get_midi_group(instr[k][1], True)
        if midi_group >= 0:
            channels.append(k)
            real_instruments.append(instr[k][1])
    if channels:
        return get_notes_map(f, channels)[0], real_instruments
    else:
        return [], []


def serialize_notes_map(f):
    channels, instruments = read_notes_map(f)
    if channels:
        write_json(os.path.splitext(f)[0], channels, instruments)


if __name__ == '__main__':
    # gan preprocess
    iterate_all_files(serialize_notes_map, file_type='.mid')
    # gpt preprocess
    iterate_all_files(serialize_file, file_type='.mid')
    iterate_counts(token_maps_round + 1)
    iterate_all_files(tokenize_json, file_type='.notes')
