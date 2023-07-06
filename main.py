import mido
import pygame
import time
import os
import random
import json
import math

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
        return 1
    elif x < 24:  # organ
        return 2
    elif x < 32:  # Guitar
        return 3
    elif x < 40:  # Bass
        return 4
    elif x < 56:  # Strings
        return 5
    elif x < 96:  # Solo
        return 6
    elif x < 104:  # effects
        return -1
    elif x < 119:  # percussion
        return 7
    else:  # effects
        return -1


full_dir = 'C:\\lmd_full'
matched_dir = 'C:\\lmd_matched'
part_dir = 'C:\\lmd_full\\0'


def iterate_all_files(func, file_type='.mid', directory=full_dir, fail_func=lambda x: None, param=None):
    for x in os.listdir(directory):
        f = directory + '\\' + x
        if os.path.isfile(f):
            if os.path.splitext(f)[1] == file_type:
                try:
                    if param:
                        func(f, param)
                    else:
                        func(f)
                except TypeError as e:
                    print('Failed on file', x, type(e), e)
                    fail_func(f)
        else:
            iterate_all_files(func, file_type, f, fail_func)


def get_random_file(dir_name=full_dir, file_type='.json'):
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


def get_data(f):
    inst = get_instruments(f)
    ch, tpb = get_notes(f)
    ch = convert_to_lengths(ch)
    return fix_timings(ch, tpb), inst


def serialize_file(f):
    ch, inst = get_data(f)
    write_json(os.path.splitext(f)[0], inst, ch)


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


if __name__ == '__main__':
    iterate_all_files(serialize_file, directory=part_dir)
