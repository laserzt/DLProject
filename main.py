import mido
import pygame
import time
import os
import random
import json

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


def iterate_all_files(func, file_type='.mid', directory=full_dir, fail_func=lambda x: None):
    for x in os.listdir(directory):
        f = directory + '\\' + x
        if os.path.isfile(f):
            if os.path.splitext(f)[1] == file_type:
                try:
                    func(f)
                except Exception as e:
                    print('Failed on file', x, e)
                    fail_func(f)
        else:
            iterate_all_files(func, file_type, f, fail_func)


def get_random_file(dir_name=full_dir):
    f = dir_name + '\\' + random.choice(os.listdir(dir_name))
    if os.path.isfile(f):
        return f
    else:
        return get_random_file(f)


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


def convert_to_lengths(channels, beats_per_bar):
    for k in channels.keys():
        j = 0
        while j < len(channels[k]) - 1:
            bar = channels[k][j]
            for i in range(len(bar) - 1):
                bar[i][1] = bar[i + 1][1] - bar[i][1]
            if len(bar) == 0:
                j += 1
                continue
            l = beats_per_bar - bar[-1][1]
            while not channels[k][j + 1]:
                j += 1
                l += beats_per_bar
            bar[-1][1] = l + channels[k][j + 1][0][1]
            j += 1
        bar = channels[k][-1]
        for i in range(len(bar) - 1):
            bar[i][1] = bar[i + 1][1] - bar[i][1]
        bar[-1][1] = beats_per_bar - bar[-1][1]
    return channels


def get_notes(filename):
    mid = mido.MidiFile(filename)
    tempo = 500000
    beats_per_bar = 4
    beats = 0
    channels = {}
    beats_in_bar = 0
    bar = 0
    for msg in mid:
        ticks = 0
        if msg.time > 0:
            ticks = int(round(mido.second2tick(msg.time, ticks_per_beat=mid.ticks_per_beat, tempo=tempo)))
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        if msg.type == 'time_signature':
            beats_per_bar = msg.numerator * 4 / msg.denominator
        if ticks:
            beats += ticks / mid.ticks_per_beat
            beats_in_bar = beats % beats_per_bar
            bar = int(beats / beats_per_bar)
        if msg.type == 'note_on' or msg.type == 'note_off':
            if msg.channel not in channels.keys():
                channels[msg.channel] = []
            if len(channels[msg.channel]) <= bar:
                for i in range(1 + bar - len(channels[msg.channel])):
                    channels[msg.channel].append([])

            if msg.type == 'note_off':
                if (not channels[msg.channel][bar]) or channels[msg.channel][bar][-1][1] != beats_in_bar:
                    channels[msg.channel][bar].append([[], beats_in_bar])
            else:
                if (not channels[msg.channel][bar]) or channels[msg.channel][bar][-1][1] != beats_in_bar:
                    channels[msg.channel][bar].append([[msg.note], beats_in_bar])
                elif channels[msg.channel][bar][-1][1] == beats_in_bar:
                    channels[msg.channel][bar][-1][0].append(msg.note)
    return convert_to_lengths(channels, beats_per_bar), beats_per_bar


def check_midi_file(filename):
    mido.MidiFile(filename)


def fix_timings(channels):
    last_k = None
    for k in channels.keys():
        for i, bar in enumerate(channels[k]):
            if bar:
                continue_loop = True
                while continue_loop:
                    continue_loop = False
                    for j in range(len(bar) - 1):
                        if not bar[j][0] and not bar[j + 1][0]:
                            channels[k][i][j] = [[], bar[j][1] + bar[j + 1][1]]
                            del channels[k][i][j + 1]
                            continue_loop = True
                            break
                    if continue_loop or not bar:
                        break
                    if not bar[0][0] and bar[0][1] < 1 / 16:
                        if i == 0:
                            if last_k:
                                l = 0
                                while not channels[last_k][-1 - l]:
                                    l += 1
                                channels[last_k][-1 - l][-1][1] += bar[0][1]
                                continue_loop = True
                                del channels[k][0][0]
                            else:
                                channels[k][0][1][1] += bar[0][1]
                                continue_loop = True
                                del channels[k][0][0]
                        else:
                            l = 0
                            while not channels[k][i - 1 - l]:
                                l += 1
                            channels[k][i - 1 - l][-1][1] += bar[0][1]
                            continue_loop = True
                            del channels[k][i][0]
                    for j in range(1, len(bar)):
                        if not bar[j][0] and bar[j][1] < 1 / 16:
                            channels[k][i][j - 1][1] += bar[j][1]
                            del channels[k][i][j]
                            continue_loop = True
                            break

                t = [round(x[1] * 3 * 16) / 3.0 / 16 for x in bar]
                for j in range(len(bar)):
                    channels[k][i][j][1] = t[j]
                    channels[k][i][j][0].sort()
        last_k = k
    return channels


def serialize_file(f):
    js = os.path.splitext(f)[0] + '.json'
    if os.path.exists(js):
        fix_json(js)
    else:
        inst = get_instruments(f)
        ch, bpb = get_notes(f)
        ch = fix_timings(ch)
        write_json(os.path.splitext(f)[0], inst, ch, bpb)


def write_json(f, inst, channels, bpb):
    song = {'Instruments': inst, 'Notes': channels, 'BeatsPerBar': bpb}
    json_object = json.dumps(song)
    with open("{0}.json".format(f), "w") as outfile:
        outfile.write(json_object)


def load_json(f):
    data = json.load(open(f, 'r'))
    inst = data['Instruments']
    channels = data['Notes']
    bpb = data['BeatsPerBar']
    return inst, channels, bpb


def fix_json(f):
    inst, channels, bpb = load_json(f)
    channels = fix_timings(channels)
    write_json(os.path.splitext(f)[0], inst, channels, bpb)


if __name__ == '__main__':
    # for i in range(100):
    # serialize_file(get_random_file())
    iterate_all_files(serialize_file)
    # play_midi(f)
    # remove_bad_files()
