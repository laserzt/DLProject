import mido
import os
import gzip
import json
import glob
import random
import threading
import gc

class MidiException(Exception):
    pass

home_dir = '/home/producer'
full_dir = os.path.join(home_dir, 'lmd_full')
part_dir = os.path.join(full_dir, '3')
out_dir = os.path.join(home_dir, 'School', 'Data')

midi_instruments = ['Acoustic Grand Piano', 'Bright Acoustic Piano', 'Electric Grand Piano', 'Honky-tonk Piano',
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
                    'Pad 4 (choir)', 'Pad 5 (bowed)', 'Pad 6 (metallic)', 'Pad 7 (halo)', 'Pad 8 (sweep)',
                    'FX 1 (rain)',
                    'FX 2 (soundtrack)', 'FX 3 (crystal)', 'FX 4 (atmosphere)', 'FX 5 (brightness)', 'FX 6 (goblins)',
                    'FX 7 (echoes)', 'FX 8 (sci-fi)', 'Sitar', 'Banjo', 'Shamisen', 'Koto', 'Kalimba', 'Bag pipe',
                    'Fiddle', 'Shanai', 'Tinkle Bell', 'Agogo', 'Steel Drums', 'Woodblock', 'Taiko Drum', 'Melodic Tom',
                    'Synth Drum', 'Reverse Cymbal', 'Guitar Fret Noise', 'Breath Noise', 'Seashore', 'Bird Tweet',
                    'Telephone Ring', 'Helicopter', 'Applause', 'Gunshot']


def get_midi_group(x, GAN):
    if x < 8:  # piano
        return 0
    elif not GAN:
        return -1
    elif x < 16:  # Chromatic Percussion
        return -1
    elif x < 24:  # organ
        return -1
    elif x < 32:  # Guitar
        return 1
    elif x < 40:  # Bass
        return 2
    elif x < 56:  # Strings
        return -1
    elif x < 96:  # Solo
        return -1
    elif x < 104:  # Esoteric
        return -1
    elif x < 120:  # percussion
        return -1
    else:  # effects
        return -1


def load_json(f):
    with gzip.open(f, 'r') as infile:
        json_bytes = infile.read()
    json_object = json_bytes.decode('utf-8')
    data = json.loads(json_object)
    inst = data['Instruments']
    channels = data['Notes']
    return inst, channels


def write_midi_maps(midi_maps, instruments):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    num_channels = len(midi_maps)
    midi_len = len(midi_maps[0])

    for i in range(num_channels):
        track.append(mido.Message('program_change', channel=i, program=instruments[i], time=0))
    notes = []
    for i in range(num_channels):
        notes.append([0] * 128)

    t = 0
    for i in range(midi_len):
        for k in range(num_channels):
            for j in range(128):
                if midi_maps[k][i][j] != notes[k][j]:
                    if midi_maps[k][i][j]:
                        track.append(mido.Message('note_on', channel=k, note=j, time=round(t), velocity=100))
                    else:
                        track.append(mido.Message('note_off', channel=k, note=j, time=round(t), velocity=0))
                    t = 0
                    notes[k][j] = midi_maps[k][i][j]
        t += mid.ticks_per_beat / 4
    mid.save(os.path.join(out_dir, 'new_song.mid'))


def write_midi(channel, filename):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.Message('program_change', program=0, time=0))
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
    mid.save(f'{filename}.mid')


def iterate_all_files(func, file_type='.mid', directory=full_dir, fail_func=lambda x: None, param=None,
                      verbose=False,
                      prefix=None, run_in_threads=True):
    if run_in_threads:
        threads = []
        for i in range(16):
            if prefix:
                new_prefix = prefix + hex(i)[2].lower()
            else:
                new_prefix = hex(i)[2].lower()

            x = threading.Thread(target=iterate_all_files,
                                 args=(func, file_type, directory, fail_func, param, verbose, new_prefix, False))
            x.start()
            threads.append(x)
        for x in threads:
            x.join()
        return
    if prefix:
        if verbose:
            print('iterate ', os.path.join(directory, prefix))
        y = glob.glob(os.path.join(directory, prefix + '*'))
    else:
        if verbose:
            print('iterate ', directory)
        y = glob.glob(os.path.join(directory, '*'))
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
                    gc.collect()
                except MidiException:
                    continue
                except Exception as e:
                    print('Failed on file', f, type(e), e)
                    fail_func(f)
                    raise e
        else:
            iterate_all_files(func, file_type, f, fail_func, param, verbose, prefix, False)

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

def get_random_file(dir_name=full_dir, file_type='.mid', prefix=None):
    if prefix:
        x = glob.glob(os.path.join(dir_name, prefix + '*'))
    else:
        x = glob.glob(os.path.join(dir_name, '*'))
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
