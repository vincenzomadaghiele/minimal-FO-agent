import mido
import time
import pretty_midi
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from FO import FactorOracle
from signalflow import *

def quantize_midi(midi_data, fs=16):
    tempo = midi_data.get_tempo_changes()[1][0] if len(midi_data.get_tempo_changes()[1]) > 0 else 120
    seconds_per_beat = 60.0 / tempo
    grid_step = seconds_per_beat / (fs / 4) # Adjust fs based on standard 4/4
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note.start = round(note.start / grid_step) * grid_step
                note.end = round(note.end / grid_step) * grid_step
                if note.end <= note.start:
                    note.end = note.start + grid_step
    return midi_data

class NotePatch (Patch):
  def __init__(self):
    super().__init__()
    note = self.add_input("note", 60)
    amplitude = self.add_input("amplitude", 0.5)
    gate = self.add_input("gate", 1.0)
    freq = MidiNoteToFrequency(note)
    env = ADSREnvelope(0.001, 0.5, 0.9, 0.2, gate=gate)
    signal = SineOscillator(frequency=freq)
    output = signal * env * amplitude
    self.set_output(output)
    self.set_auto_free(True)

if __name__ == "__main__":

    # LOAD MIDI FILE
    midi_file_path = 'bach_988-v10.mid'
    pm = pretty_midi.PrettyMIDI(midi_file_path)
    pm = quantize_midi(pm)
    piano_roll = pm.get_piano_roll(fs=100)

    # VISUALIZE MIDI FILE
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(piano_roll, hop_length=1, 
                            sr=100, x_axis='time', y_axis='cqt_note', 
                            fmin=pretty_midi.note_number_to_hz(0))
    plt.title('Variatio 10 a 1 Clav. Fughetta (J. S. Bach, 1741)')
    plt.show()

    # EXTRACT SEQUENCES FOR TRAINING
    LENGTH_TRAINING_SEQUENCE = 100
    pitches, durations = [], []
    prev_end = 0
    for note in pm.instruments[0].notes[:LENGTH_TRAINING_SEQUENCE]:
        if note.start >= prev_end:
            pitches.append(str(note.pitch))
            durations.append(str(round(note.end - note.start, 3)))
            prev_end = note.end

    print(f'unique pitches: {list(set(pitches))}')
    print(f'unique durations: {list(set(durations))}')

    # TRAIN FACTOR ORACLES
    pitchFO = FactorOracle()
    pitchFO.train(pitches)
    pitchVocab = pitchFO.symbols
    pitchFO.visualize()

    durationFO = FactorOracle()
    durationFO.train(durations)
    durationVocab = durationFO.symbols
    durationVocab = [float(x) for x in durationVocab]
    durationFO.visualize()


    # audio graph
    graph = AudioGraph()
    graph.poll(0.5)

    # MIDI input
    MIDI_device_name = "Arturia KeyStep 32" # substitute with the name of your MIDI device
    default_input = mido.open_input(MIDI_device_name)

    # initialize note patch
    patch = NotePatch()
    spec = patch.to_spec()
    voices = [ None ] * 128

    # initialize circular buffers to the first 10 values of the original sequence
    MEMORY_LENGTH = 10
    PITCH_MEMORY = pitches[:MEMORY_LENGTH] 
    DURATION_MEMORY = durations[:MEMORY_LENGTH]

    # lists used for prediction
    LENGTH_PREDICTED_SEQUENCE = 10
    PITCH_RESPONSES = []
    DURATION_RESPONSES = []

    responding = False
    voices_response = [ None ] * 128

    # receive non-blocking MIDI messages
    with mido.open_input(MIDI_device_name) as port:
        # interaction loop
        while True:
            for message in port.iter_pending():
                if message.type == 'note_on':
                    voice = Patch(spec)
                    voice.set_input("note", message.note)
                    voice.set_input("amplitude", message.velocity / 127)
                    voice.play()
                    voices[message.note] = voice
                    note_start = time.time()
                elif message.type == 'note_off':
                    if voices[message.note] is not None:
                        voices[message.note].set_input("gate", 0)
                        voices[message.note] = None
                        latest_note_duration = time.time() - note_start
                        if str(message.note) in pitchVocab: 
                            PITCH_MEMORY[:-1] = PITCH_MEMORY[1:]
                            PITCH_MEMORY[-1] = str(message.note)
                            latest_note_duration = durationVocab[np.abs(np.array(durationVocab) - latest_note_duration).argmin()]
                            DURATION_MEMORY[:-1] = DURATION_MEMORY[1:]
                            DURATION_MEMORY[-1] = str(latest_note_duration)

                            # PREDICT NEW NOTE
                            PITCH_RESPONSES = pitchFO.predict(PITCH_MEMORY, LENGTH_PREDICTED_SEQUENCE, 0.8)
                            DURATION_RESPONSES = durationFO.predict(DURATION_MEMORY, LENGTH_PREDICTED_SEQUENCE, 0.8)

            if not responding:
                if PITCH_RESPONSES != [] and DURATION_RESPONSES != []:
                    responding_pitch = float(PITCH_RESPONSES[0])
                    responding_duration = float(DURATION_RESPONSES[0])
                    voice = Patch(spec)
                    voice.set_input("note", responding_pitch)
                    voice.set_input("amplitude", 60 / 127)
                    voice.play()
                    voices_response[int(responding_pitch)] = voice
                    response_note_start_time = time.time()
                    responding = True
                    
                    # update playing queue of response sequences
                    PITCH_RESPONSES[:-1] = PITCH_RESPONSES[1:]
                    PITCH_RESPONSES.pop()
                    DURATION_RESPONSES[:-1] = DURATION_RESPONSES[1:]
                    DURATION_RESPONSES.pop()
            else:
                if time.time() - response_note_start_time >= responding_duration:
                    voices_response[int(responding_pitch)].set_input("gate", 0)
                    voices_response[int(responding_pitch)] = None
                    responding = False

            time.sleep(0.001)