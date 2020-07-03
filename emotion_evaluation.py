import os
from music21 import converter, corpus, instrument, midi, note, chord, pitch, environment
import librosa
import vamp
import argparse
from midiutil.MidiFile import MIDIFile
from scipy.signal import medfilt
import numpy as np

'''
Wav to MIDI and chord detection

Requirements: pip install numpy music21 vamp librosa midiutil matplotlib scipy
sudo apt-get install timidity

Melodia plugin in Mac:
    cd /Library/Audio/Plug-Ins
    mkdir Vamp && cd Vamp
    cp Melodia/* ./

Linux:
    Go to ./venv36/lib/python3.6/site-packages/music21/graph/primitives.py
        > Line 502 (first line of callDoneAction) 
        > Add `fp = self.savedKeywords['fp']`

References:
Chord detection with Music21: https://www.kaggle.com/wfaria/midi-music-data-extraction-using-music21
'''


# Open MIDI files
# Some helper methods.
def concat_path(path, child):
    return path + "/" + child


# Music21 library: robust platform to explore music files and music theory.
def open_midi(midi_path, remove_drums):
    # There is an one-line method to read MIDIs
    # but to remove the drums we need to manipulate some
    # low level MIDI events.
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if (remove_drums):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]

    return midi.translate.midiFileToStream(mf)


def save_midi(outfile, notes, tempo):
    track = 0
    time = 0
    midifile = MIDIFile(1)

    # Add track name and tempo.
    midifile.addTrackName(track, time, "MIDI TRACK")
    midifile.addTempo(track, time, tempo)

    channel = 0
    volume = 100

    for note in notes:
        onset = note[0] * (tempo / 60.)
        duration = note[1] * (tempo / 60.)
        # duration = 1
        pitch = int(note[2])
        midifile.addNote(track, channel, pitch, onset, duration, volume)

    # And write it to disk.
    binfile = open(outfile, 'wb')
    midifile.writeFile(binfile)
    binfile.close()


def midi_to_notes(midi, fs, hop, smooth, minduration):
    # smooth midi pitch sequence first
    if (smooth > 0):
        filter_duration = smooth  # in seconds
        filter_size = int(filter_duration * fs / float(hop))
        if filter_size % 2 == 0:
            filter_size += 1
        midi_filt = medfilt(midi, filter_size)
    else:
        midi_filt = midi
    # print(len(midi),len(midi_filt))

    notes = []
    p_prev = None
    duration = 0
    onset = 0
    for n, p in enumerate(midi_filt):
        if p == p_prev:
            duration += 1
        else:
            # treat 0 as silence
            if p_prev is not None and p_prev > 0:
                # add note
                duration_sec = duration * hop / float(fs)
                # only add notes that are long enough
                if duration_sec >= minduration:
                    onset_sec = onset * hop / float(fs)
                    notes.append((onset_sec, duration_sec, p_prev))

            # start new note
            onset = n
            duration = 1
            p_prev = p

    # add last note
    if p_prev is not None and p_prev > 0:
        # add note
        duration_sec = duration * hop / float(fs)
        onset_sec = onset * hop / float(fs)
        notes.append((onset_sec, duration_sec, p_prev))

    return notes


def hz2midi(hz):
    # convert from Hz to midi note
    hz_nonneg = hz.copy()
    idx = hz_nonneg <= 0
    hz_nonneg[idx] = 1
    midi = 69 + 12 * np.log2(hz_nonneg / 440.)
    midi[idx] = 0

    # round
    midi = np.round(midi)

    return midi


def audio_to_midi_melodia(infile, outfile, bpm, smooth=0.25, minduration=0.1):
    # define analysis parameters
    fs = 44100
    hop = 128

    # load audio using librosa
    print("Loading audio...")
    data, sr = librosa.load(infile, sr=fs, mono=True)

    # extract melody using melodia vamp plugin
    print("Extracting melody f0 with MELODIA...")
    melody = vamp.collect(data, sr, "mtg-melodia:melodia",
                          parameters={"voicing": 0.2})

    # hop = melody['vector'][0]
    pitch = melody['vector'][1]

    # impute missing 0's to compensate for starting timestamp
    pitch = np.insert(pitch, 0, [0] * 8)

    # debug
    # np.asarray(pitch).dump('f0.npy')
    # print(len(pitch))

    # convert f0 to midi notes
    print("Converting Hz to MIDI notes...")
    midi_pitch = hz2midi(pitch)

    # segment sequence into individual midi notes
    notes = midi_to_notes(midi_pitch, fs, hop, smooth, minduration)

    # save note sequence to a midi file
    print("Saving MIDI to disk...")
    save_midi(outfile, notes, bpm)

    print("Conversion complete.")


def init_parse():
    sample = 7
    duration = 3
    em_type, em_type_short, em_type_int = 'positive', 'pos', 1
    #em_type, em_type_short, em_type_int = 'negative', 'neg', 0
    parser = argparse.ArgumentParser(description='Convert wav to midi')
    parser.add_argument('--data_dir', default='./datasets/{}secs/{}/{}_sample{}/'.format(duration, em_type, em_type_short, sample),  # './data_1/',
                        help='Directory with data')
    #parser.add_argument('--infile', default='epTest_cnnseq2sample-s{}-em{}.wav'.format(sample, em_type_int),
    parser.add_argument('--infile', default='epTest_cnnseq2sample-s{}-em{}_cnnseq2seqAudio.wav'.format(sample, em_type_int),
                        # 'set6_epTest_cnnseq2sample-s16-em1_cnnseq2seqAudio.wav',
                        help='Filename waveform')
    parser.add_argument('--outfile', default='epTest_cnnseq2sample-s{}-em{}_cnnseq2seqAudio.mid'.format(sample, em_type_int),
                        # 'set6_epTest_cnnseq2sample-s16-em1_cnnseq2seqAudio.mid',
                        help='Filename MIDI')

    args = parser.parse_args()
    return args


# For linux
# Fix: "music21.environment.EnvironmentException: Cannot find a valid application path for format None. Specify this in your Environment by calling environment.set('graphicsPath', '/path/to/application')"
# https://web.mit.edu/music21/doc/moduleReference/moduleEnvironment.html
us = environment.UserSettings()
environment.set('graphicsPath', '/usr/bin/timidity')

args = init_parse()
data_dir = args.data_dir
# print(os.listdir(data_dir))

# Save as test.mid
audio_to_midi_melodia(infile=data_dir + args.infile,
                      outfile=data_dir + args.outfile,
                      bpm=146, smooth=0.25, minduration=0.1)

# base_midi = open_midi(concat_path(sonic_path, "green-hill-zone.mid"), True)
midi_path = concat_path(data_dir, args.outfile)
print(midi_path)
base_midi = open_midi(midi_path, True)
# print(base_midi)

# We can take a look on the pitch histogram to see which notes are more used.
fp = concat_path(data_dir, args.outfile.split('.mid')[0] + '.png')
base_midi.plot('histogram', 'pitchClass', 'count', fp=fp)

# Extra composition parameters
timeSignature = base_midi.getTimeSignatures()[0]
music_analysis = base_midi.analyze('key')
str = "Music time signature: {0}/{1}\n".format(timeSignature.beatCount, timeSignature.denominator)
str += "Expected music key: {0}\n".format(music_analysis)
str += "Music key confidence: {0}\n".format(music_analysis.correlationCoefficient)
str += "Other music key alternatives:\n"
for analysis in music_analysis.alternateInterpretations:
    if (analysis.correlationCoefficient > 0.5):
        str += '    {}\n'.format(analysis)

print(str)
fp = concat_path(data_dir, args.outfile.split('.mid')[0] + '.txt')
f = open(fp, 'w')
f.write(str)
f.close()
