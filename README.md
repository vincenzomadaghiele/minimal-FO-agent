# A minimal co-improvising musical agent in Python with Factor Oracle
### A short tutorial to make a simple musical agent

![](https://miro.medium.com/v2/resize:fit:1400/1*_a5SNAte-DmuiF9MFpA5iw.png)
*A player piano, a mechanical piano that can be activated with MIDI messages sent by a computer program. In this picture a Yamaha Disklavier from University of Oslo, photo by the author.*

## Musical agents

With the recent advances in artificial intelligence, the idea of developing of autonomous musical agents is becoming increasingly popular.

> **Musical agents** are artificial agents that tackle musical creative tasks, partially or completely [(Tatar and Pasquier, 2019)](https://www.tandfonline.com/doi/pdf/10.1080/09298215.2018.1511736?casa_token=GV4AAQM1YF0AAAAA%3AETC7zd8PHphbe7NVzU8Lbp7bD0SfTiy2FLgXqt6PFK0rCSl-8IDerepnGO9R7kV4ds0fHOJt3l8a).

These computer programs can perform tasks like continuing a melody, playing in rhythm with a live musician, or generating accompaniment. This topic has gained wider popularity recently, with projects like [Holly+ by Holly Herndon](https://holly.plus/) and [DADABOTS](https://dadabots.com/) capturing attention outside specialist circles, however it is far from new. The first musical piece composed with a highly-automated computer program — the [Illiac Suite (Lejaren Hiller and Leonard Isaacson)](https://www.youtube.com/watch?v=fojKZ1ymZlo&t=363s) — dates back to 1957, while the first live music co-improvisation system — [Voyager (George Lewis)](https://www.youtube.com/watch?v=o9UsLbsdA6s) — was developed in 1987. The use of algorithmic processes to write music predates computers, with historical examples like [musical dice games](https://en.wikipedia.org/wiki/Musikalisches_W%C3%BCrfelspiel) and [mechanical musical automatons](https://archive.org/details/TheBookOfKnowledgeOfIngeniousMechaniIbnAlRazzazAlJazari/page/207/mode/2up).

In this article, I will outline a step-by-step procedure to make a minimal live music co-improvisation agent using the Factor Oracle model, with examples in the Python programming language.

> A **co-improvisation system** is a musical agent that generates music in response to a guiding track produced by a live musician in real-time ([paraphrased from Assayag et al., 2010](https://inria.hal.science/hal-00694801/file/AL-chapter4.pdf)).

The aim of the article is to introduce the basic concepts of live musical agents, and provide an extremely basic working software example, that can be expanded by interested computer musicians by modifying its basic components.

*Voyager (1987) by George Lewis, is considered the first co-improvisation live musical agent. George Lewis has been working on improvised music and computer music systems since the 1970s.*

## Components of a co-improvising agent

A live musical agent needs to able to interpret a live musician’s sonic input, process it, and generate an appropriate musical response. These are the basic components of the software we will develop: (1) live input analysis, (2) response generation, (3) sound synthesis ([Blackwell et al., 2012](https://link.springer.com/chapter/10.1007/978-3-642-31727-9_6)).

### 1\. Live input analysis

The two most used format of live input for co-improvising agents are audio and symbolic music. Live audio is usually processed to extract musically significant information ([Music Information Retrieval](https://en.wikipedia.org/wiki/Music_information_retrieval)), while symbolic music formats like MIDI and [musicXML](https://www.musicxml.com/) can be used without much preprocessing. For simplicity, in this tutorial we will use MIDI, by far the most popular symbolic music format. We use a MIDI keyboard as an input device, rendering the live sound in Python with a basic synthesizer.

![MIDI keyboard Arturia KeyStep mk2 is used as live input device for playing notes.](https://miro.medium.com/v2/resize:fit:1400/1*Rzqtn9R_jTF6Gnmk615NpA.png)
*MIDI keyboard Arturia KeyStep mk2 is used as live input device for playing notes. Image from the Arturia Website.*

### 2\. Response generation

We want to process live MIDI input, and generate appropriate musical responses in MIDI format. To do so, we use Factor Oracle (FO), a sequence model [initially developed for string matching in biology](https://hal.science/hal-00619846v1/document), later applied to music co-improvisation because of its ability to model variations of sequences in a compact and easily accessible way.

> Basically, **Factor Oracle** is a compact structure which represents at least all the factors in a word *w*. It is an acyclic automaton with an optimal *(m+1)* number of states and is linear in the number of transitions (at most *2m-1* transitions), where *m = |w|*. ([Assayag and Dubnov, 2004](https://hal.science/hal-01161221v1/document))

For example, given the word [CACIOCAVALLO](https://en.wikipedia.org/wiki/Caciocavallo), we can build the FO of this word by considering each letter of the word as a unique symbol.

![](https://miro.medium.com/v2/resize:fit:2000/1*HsZ_FNlOvyRgBwXKFqSzGQ.png)
*Factor Oracle of the word ‘CACIOCAVALLO’.*

The symbols contained in the word CACIOCAVALLO are the unique letters: (‘C’, ‘A’, ‘I’, ‘O’, ‘V’, ‘L’). The number of *states* (the circled numbers in the diagram) of the FO model is equal to the number of letters in the original word. Navigating the FO, we can reconstruct all the *factors* of the original word, that is, all the possible subsequences contained in the original word. For example, we can generate ‘ACI’ and ‘AVAL’, by starting in the first state of the oracle and following the right arrows.

## Get Vincenzo Madaghiele’s stories in your inbox

Join Medium for free to get updates from this writer.

FO can be used in music as a generator of variations over an original sequence of notes. For example, instead of a word, we can train a FO on the sequence of MIDI pitches of a melody. To use it live, given an input sequence of notes played by a live musician, we search for the corresponding state of the input sequence — or of a subset of it — in the FO model, and try to *continue it* by following arrows in the model, as described [in the original paper by Assayag and Dubnov](https://hal.science/hal-01161221v1/document).

### 3\. Sound synthesis

A live co-improvisation response is, in our demo case, a sequence of MIDI notes. MIDI is a symbolic format, that can be used to control an arbitrary synthesizer. In this tutorial, we will render the MIDI notes to sound with a basic synthesizer in Python, however any synthesizer can theoretically be used, and MIDI notes can also be processed in the symbolic domain, for example transposing pitch or modifying their duration, for a wider variation of musical effects.

## Finally coding!

We can finally start putting together our live system in Python. We will use the following Python libraries:

-   [**Signalflow**](https://signalflow.dev/) for live sound and synthesis
-   [**Mido**](https://mido.readthedocs.io/en/stable/) for live MIDI controls
-   [**Pretty\_midi**](https://craffel.github.io/pretty-midi/) for offline analysis of MIDI files
-   [**Librosa**](https://librosa.org/doc/latest/index.html) and [**Matplotlib**](https://matplotlib.org/) for visualization

### Training a FO on a MIDI file

We start by loading and inspecting a MIDI file, which we will use to train a FO. In this case, we use *Variatio 10 a 1 Clav. Fughetta (J. S. Bach, 1741),* from J.S. Bach’s Goldberg variations, downloaded from [this free online MIDI resource](http://www.jsbach.net/midi/midi_goldbergvariations.html).

```
import pretty_midi
import librosa.display
import matplotlib.pyplot as plt

# LOAD MIDI FILE
midi_file_path = '988-v10.mid'
pm = pretty_midi.PrettyMIDI(midi_file_path)
```

We quantize the duration of the notes in the MIDI file, a step that will be useful later.

```
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

pm = quantize_midi(pm)
```

We can now extract a *piano roll* representation of the MIDI file and visualize it.

```
piano_roll = pm.get_piano_roll(fs=100)

# VISUALIZE MIDI FILE
plt.figure(figsize=(10, 5))
librosa.display.specshow(piano_roll, hop_length=1,
                        sr=100, x_axis='time', y_axis='cqt_note',
                        fmin=pretty_midi.note_number_to_hz(0))
plt.title('Variatio 10 a 1 Clav. Fughetta (J. S. Bach, 1741)')
plt.show()
```

![Piano roll representation of MIDI file.](https://miro.medium.com/v2/resize:fit:1400/1*ZQ7PB-i2ImkBmvNzQVn1Nw.png)
*Piano roll representation of MIDI file.*

This whole score can be used to train a FO, however we want to start with a smaller model for demonstration purposes. We decide to train a model with 100 states. To do so, we extract the first 100 notes (pitch and duration separate), from the score. To simplify even more, we ignore polyphony (when more than one note plays at the same time), by extracting only the note with the highest pitch between two simultaneous notes. To train the FO, we need to obtain **discrete classes** from values of pitches and duration. These classes are the unique values of pitch and duration.

```
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
```

We can now use these sequences of pitch and duration to train our model. While there are excellent [Python implementations of Factor Oracle online](https://github.com/wangsix/vmo), for simplicity we implement the basic algorithm from scratch. The Python class below implements the basic algorithm as described [in the original paper](https://hal.science/hal-01161221v1/document). A step-by-step tutorial of this implementation is out of the scope of this article.

```
import random

class FactorOracle:
    def __init__(self):
        # symbols of the oracle
        self.symbols = []
        # for each state, sigma stores a list of transitions for each symbol
        self.sigma = []
        # suffix links for each state
        self.S = []

    def train(self, sequence):
        w = sequence
        self.S.append(None)
        for i in range(1,len(w)+1):
            i_word = i-1
            if w[i_word] not in self.symbols: # add new symbol to dictionary
                self.symbols.append(w[i_word])
                for ss in self.sigma: # add new word space to past sigmas
                    ss.append(None)
            for j in range(len(self.symbols)): # find index of current symbol
                if self.symbols[j] == w[i_word]:
                    symbolIdx = j
            # initialize empty states
            self.S.append(None)
            self.sigma.append([None for _ in range(len(self.symbols))])
            self.sigma[i-1][symbolIdx] = i
            k = self.S[i-1]
            while k is not None and self.sigma[k][symbolIdx] is None:
                self.sigma[k][symbolIdx] = i
                k = self.S[k]
            if k is not None:
                self.S[i] = self.sigma[k][symbolIdx]
            else:
                self.S[i] = 0
        self.sigma.append([None for _ in range(len(self.symbols))])

    def predict(self, sequence=[], num_predictions=1, p=1):
        state = 0
        v = sequence
        # look for factor matching input sequence
        if all(char in sequence for char in self.symbols):
            foundMatch = False
            while not foundMatch:
                for i in range(len(v)):
                    # find index of current symbol
                    for j in range(len(self.symbols)):
                        if self.symbols[j] == v[i]:
                            symbolIdx = j
                    if state is not None:
                        state = self.sigma[state][symbolIdx]
                        foundMatch = True
                    else:
                        foundMatch = False
                        break
                if v != []:
                    v = v[1:]
                else:
                    break
        i = state if state is not None else 0
        # generate predicted sequence
        preds = []
        seq_len = num_predictions
        for n in range(seq_len):
            q = 1 if self.S[i] == None else p
            if random.random() < q and i < len(self.S)-1:
                idx = self.sigma[i].index(i+1)
                preds.append(self.symbols[idx])
                i += 1
            else:
                sig = self.sigma[self.S[i]]
                not_none_idxs = [j for j in range(len(sig)) if sig[j] is not None]
                idx = random.choice(not_none_idxs)
                preds.append(self.symbols[idx])
                i = self.sigma[self.S[i]][idx]
        return preds
```

We train two separate FOs for the sequences of pitch and duration.

```
pitchFO = FactorOracle()
pitchFO.train(pitches)
pitchVocab = pitchFO.symbols

durationFO = FactorOracle()
durationFO.train(durations)
durationVocab = durationFO.symbols
durationVocab = [float(x) for x in durationVocab]
```

The result will be two parallel FOs that look like these:

![](https://miro.medium.com/v2/resize:fit:2000/1*FFi81w4uNUR2SsD0M0Cw3A.png)
*Pitch FO with 100 states. Pitches are represented as MIDI values over transitions.*

![](https://miro.medium.com/v2/resize:fit:2000/1*A3xoR7oPSBM3Om_DUgtLsA.png)
*Duration FO with 100 states. Durations are represented as values in milliseconds over transitions.*

### Live interaction with the model using a MIDI controller

We can now use our trained model to respond to a live musician. To begin with, we need to make sound using our MIDI keyboard. First, we make a simple sine wave synthesizer patch in Signalflow that will play a MIDI note.

```
from signalflow import *

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
```

We then use the MIDI keyboard to play the patch using code inspired by the [MIDI keyboard example from the Signalflow repository](https://github.com/ideoforms/signalflow/blob/master/examples/midi-keyboard-example.py). The *while* loop is where we code the live interactions with the model, where we do operations in musical time. It is updated every 0.001 seconds.

```
import mido
import time
import numpy as np

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

with mido.open_input(MIDI_device_name) as port:
  while True:
    for message in port.iter_pending():
      if message.type == 'note_on':
        voice = Patch(spec)
        voice.set_input("note", message.note)
        voice.set_input("amplitude", message.velocity / 127)
        voice.play()
        voices[message.note] = voice
      elif message.type == 'note_off':
        if voices[message.note] is not None:
          voices[message.note].set_input("gate", 0)
          voices[message.note] = None
  time.sleep(0.001)
```

Now comes the trickiest part of live interaction: coordinating the timings of inputs and output responses. To activate the FO, we save the sequence of the latest 10 live MIDI inputs in a list that will function as a [circular buffer](https://en.wikipedia.org/wiki/Circular_buffer). **For simplicity, we skip input MIDI pitches that are not in the vocabulary of the pitch FO, and we quantize the duration of the selected pitches to the closest values (computed using euclidean distance) in the vocabulary of the duration FO.**

At the end of each new input MIDI note, we use the updated sequence to generate a novel sequence response with the FOs. To do so, we modify the interaction loop as follows.

```
# ...

# initialize circular buffers to the first 10 values of the original sequence
MEMORY_LENGTH = 10
PITCH_MEMORY = pitches[:MEMORY_LENGTH]
DURATION_MEMORY = durations[:MEMORY_LENGTH]

# lists used for prediction
LENGTH_PREDICTED_SEQUENCE = 10
PITCH_RESPONSES = []
DURATION_RESPONSES = []

with mido.open_input(MIDI_device_name) as port:
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
 time.sleep(0.001)
```

At the same time as we are analyzing incoming input MIDI notes, we want to play back the responses generated by the FOs. We use the sequence of MIDI responses to trigger notes in a second synthesizer, identical to the one used for the keyboard inputs. When a new response sequence is generated, the previously playing sequence is erased and the new sequence is substituted to it. We add the following code section to the *while* loop.

```
# ...
responding = False
voices_response = [ None ] * 128

with mido.open_input(MIDI_device_name) as port:
  while True:

    # ...

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
```

There it is! A minimal live co-improvisation musical agent coded entirely in Python.

## Putting it all together (and personalization ideas)

We can now put all the elements together and customize the code to make this system more expressive, functional and personal. Some suggestions of improvements, left to the reader as an exercise, are:

-   Using different Signalflow synthesis patches for playback of source and agent sounds;
-   Modifying the agent’s MIDI responses before playback, for example transposing pitch or increasing/decreasing notes duration;
-   Detecting rests in pitch sequences;
-   Implementing polyphony by encoding simultaneous pitches in unique FO classes;
-   Using a [multichannel FO](https://watermark02.silverchair.com/comj_a_00460.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAzkwggM1BgkqhkiG9w0BBwagggMmMIIDIgIBADCCAxsGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMCoRGYI_HSVUr4AXeAgEQgIIC7Cku6P-r5z1yc1xDtOWhSDSdbSDxSm8bfT2FUPQDx27R7tp6EXRHV9U5SoV7M6vPNstyyRG4_15Wez6xxZy_CHm585aaJm9CGiZJuE3cgZc9UpPO-KgaytV0vomH_cE-24-uxSkG9l9C54SInaiuLSBDlzbwOwlk8bfoa5jIOH0mPdZ8oLx7NIO17g6yjcuqHOMbxA_OcjFShGJwFusjh9o-dLRAazaXLp0Dz5cZxTt0NN75FQZOmPL93_-FY4xGrhGhMz0i3z7nu231tRuZkD73OOnxwHmmTNzMZELayaikWPzPS8w4pP-tvP0IcFsFuFyRFuh4n3MFNZgSeIoZEGLCsd8gSsI_TI45JNixQj4khif1H1KWXH9-H7vFcgkGn6dRfumiGmsBgOy9g_ym7GWB1cvt9CbvRBCGv3V54fZaRBi0jXJGdu6NFsmh4TRwGQflnjouori1igxUqH1qYTsGG3Ly0nS95mm_Vhf_6c8A0QPbq6Gcy_fGYGIeRf0DSou6okUVej9THx2OGAUQRiKD38ygFJyoh-eC77_haf9jLip5Gl1c0f7q9tqTlvtfEbr7YcmxB7v3KJT92bJ4eZwhMyHqnrY-HqtQS_TXV2l42UybUa1gryc0f148fgrkR6z0_rEPP5DxSMDsjET95a0hVXSoM-xHVEjzcY6qpJDcK8mdWi-xn0X58rqcSYxPjFRQD7GUBkMva94Z0RbBMTGX65r8D2L7EXmruaDjvTIfAX_ExS0jAUoJ_pajGmsaohgb5VnNcUlfMT6BHHzwYtMFCueHtGP9BYL7VVXBQ8pBj5eckvbb56ixCWJ71BBmQkjAybLVB2QMjm_GApiLokmfx8UbWRW_lTExgsd0j8HM-_SMv5kOkzFmjktndhtzpnNT8RFUrVFIoo3bs4G9oIzTNBc_85YU-Oj7E4OdgF_oXV7sNRqvnY6XnU-ElL2syX4d7Ok6_Y_wmxMH2S33eVP6wGxZ7qNvCZFAxRA) model for coherent pitch-duration training;
-   Improving scheduling of model responses for playback;
-   Implementing live audio input instead of MIDI.

This video shows an improvisation by the author of this article, using a modified version of the program (pardon my nonexistent piano skills).

*A short video demo of a slightly modified version of the live system. The input sound is on the left channel, the sound produced by the artificial agent is on the right channel for demonstration purposes.*


This code is released under [open-source license LGPL-3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html). This work it is funded by the University of Oslo (Norway).