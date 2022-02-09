import tkinter as Tk
from PIL import ImageTk, Image
from numpy.core.getlimits import _KNOWN_TYPES
from numpy.lib.shape_base import kron
import pyaudio, wave, struct
from matplotlib import style
from math import pi, floor
from cmath import exp
import numpy as np
from scipy import signal

style.use("ggplot")

# Initialization
root = Tk.Tk()
root.title("Audio Sampler")

# Number of Trigger Pad
N = 8
PAD = list(range(1, N + 1))  # PAD = [1, 2, 3, ... 8]

# Audio Parameters
BLOCKLEN = 1024
DURATION = 5
RATE = 16000
LEN = DURATION * RATE
WIDTH = 2
CHANNELS = 1
BUFFERSIZE = 128
MAXVALUE = 2 ** 15 - 1

# For playing wavfile
num_playing_blocks = [0] * N
byte_data = [bytes(0)] * N
index = [0] * N

# For recording wavfile
num_recording_blocks = int(floor(LEN / BLOCKLEN))
counter = 0

# Streams for playing, recording, monitoring
p = pyaudio.PyAudio()
PA_FORMAT = pyaudio.paInt16

streams = []
for i in range(N):
    stream = p.open(
        format=PA_FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=False,
        output=True,
        # frames_per_buffer=BUFFERSIZE,
    )
    streams.append(stream)

recording_stream = p.open(
    format=p.get_format_from_width(WIDTH),
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    # frames_per_buffer=BUFFERSIZE,
)

monitor_stream = p.open(
    format=p.get_format_from_width(WIDTH),
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    # frames_per_buffer=BUFFERSIZE,
)

# Play audio

TRIGGERED = [False] * N
PLAYING = [False] * N
LOADED = [False] * N
END = [False] * N


def play(num):
    global RECORDING
    global TRIGGERED
    global PLAYING
    global LOADED
    global PAD
    global byte_data

    if not RECORDING:
        for i in range(N):
            if num == PAD[i]:
                print("Play " + str(num) + ".wav")
                TRIGGERED[i] = True
                if not LOADED[i]:
                    LOADED[i] = True
                    (num_playing_blocks[i], byte_data[i]) = load(num)


def load(filename):

    input_wavfile = str(filename) + ".wav"

    wf = wave.open(input_wavfile, "rb")
    LEN = wf.getnframes()
    num_blocks = int(floor(LEN / BLOCKLEN))
    total_output = []

    print("* Start playing wave file: %s." % input_wavfile)

    for i in range(0, num_blocks):

        input_bytes = wf.readframes(BLOCKLEN)
        total_output.append(input_bytes)

    print("* Finished.")
    wf.close()

    return (num_blocks, total_output)


# record audio
RECORDING = False
FINISHED = False


def record(filename):
    global RECORDING
    global FINISHED
    global RATE
    global WIDTH
    global CHANNELS
    global wf
    global output_wavfile

    if filename:  # filename != 0

        output_wavfile = str(filename) + ".wav"
        wf = wave.open(output_wavfile, "w")

        wf.setnchannels(CHANNELS)
        wf.setsampwidth(WIDTH)
        wf.setframerate(RATE)

        RECORDING = True
        FINISHED = False
        LOADED[filename - 1] = False  # if new recording, reload wavfile

        print("start recording...")
        print(RB.get())


def stop():
    global RECORDING
    global FINISHED
    global counter
    global wf
    global recording_stream

    if RECORDING:
        RECORDING = False
        FINISHED = True
        counter = 0
        wf.close()
        print("* Stopped.")


def keyboard_trigger(event):
    global CONTINUE
    global TRIGGERED
    global PLAYING
    global LOADED
    global PAD
    global byte_data

    print("You pressed " + event.char)
    if event.char == "q":
        print("Good bye")
        CONTINUE = False

    if event.char == "w":
        record(RB.get())

    if event.char == "e":
        stop()

    if event.char == "r":
        Input_level_Slider.set(0.0)
        De_noise_Slider.set(0.0)
        Pitch_Slider.set(0.0)
        EQ_band1_Slider.set(0.0)
        EQ_band2_Slider.set(0.0)
        EQ_band3_Slider.set(0.0)
        EQ_band4_Slider.set(0.0)
        EQ_band5_Slider.set(0.0)
        Delay_band1_Slider.set(DEFAULT_DELAY_TIME)
        Delay_band2_Slider.set(0.0)
        Reverb_band_Slider.set(0.00)
        print("Reset all the sliders")

    for i in range(N):
        if event.char == str(PAD[i]):
            print("Play " + event.char + ".wav")
            TRIGGERED[i] = True
            if not LOADED[i]:
                LOADED[i] = True
                (num_playing_blocks[i], byte_data[i]) = load(int(event.char))


root.bind("<Key>", keyboard_trigger)

# effects
# input -> dynamic range -> tone adjust -> time -> output
# input -> level -> gate -> pitchshifter -> equalizer -> delay -> reverb -> output


def effects(x):

    y0 = np.asanyarray(x)  # all bypass

    y1 = level(x)

    y2 = gate(y1)

    y3 = pitchshitfer(y2)

    y4 = equalizer(y3)

    y5 = delay(y4)

    y6 = reverb(y5)

    y = np.asanyarray(y6)

    return y


def level(x):

    input_gain = 10.0 ** (Input_level.get() / 10.0)

    y = np.zeros(BLOCKLEN)
    for n in range(0, BLOCKLEN):
        y[n] = input_gain * x[n]

    return y


def gate(x):

    threshold = De_noise.get()
    ratio = 0.3

    y = np.zeros(BLOCKLEN)
    for n in range(0, BLOCKLEN):
        if abs(x[n]) <= threshold:
            y[n] = ratio * x[n]
        else:
            y[n] = x[n]

    return y


# Complex Modulation Parameters:
K = 7
a_lpf = [1.0000, -1.2762, 2.6471, -2.2785, 2.1026, -1.1252, 0.4876, -0.1136]
b_lpf = [0.0423, 0.1193, 0.2395, 0.3208, 0.3208, 0.2395, 0.1193, 0.0423]
p_states = np.zeros(K)
I = 1j
a = []
b = []
for i in range(K + 1):
    a.append(a_lpf[i] * (I ** i))
    b.append(b_lpf[i] * (I ** i))
t = 0
g = np.zeros(BLOCKLEN, dtype="complex_")


def pitchshitfer(x):
    global a, b, p_states, t

    f = Pitch.get()

    y, p_states = complex_modulation(x, a, b, p_states, f, t)

    return y


def complex_modulation(x, a, b, p_states, f, t):

    [r, p_states] = signal.lfilter(b, a, x, zi=p_states)

    y = np.zeros(BLOCKLEN)
    for n in range(0, BLOCKLEN):

        g[n] = r[n] * exp(I * 2 * pi * f * t)
        t = t + (1 / RATE)
        y[n] = g[n].real

    return y, p_states


# EQ parameters:
states = [np.zeros(4)] * 5


def equalizer(x):

    y1, states[0] = bandpassfilter(x, 1, states[0])
    y2, states[1] = bandpassfilter(x, 2, states[1])
    y3, states[2] = bandpassfilter(x, 3, states[2])
    y4, states[3] = bandpassfilter(x, 4, states[3])
    y5, states[4] = bandpassfilter(x, 5, states[4])

    EQ_gain1 = 10.0 ** (EQ_band1.get() / 10.0)
    EQ_gain2 = 10.0 ** (EQ_band2.get() / 10.0)
    EQ_gain3 = 10.0 ** (EQ_band3.get() / 10.0)
    EQ_gain4 = 10.0 ** (EQ_band4.get() / 10.0)
    EQ_gain5 = 10.0 ** (EQ_band5.get() / 10.0)

    y = EQ_gain1 * y1 + EQ_gain2 * y2 + EQ_gain3 * y3 + EQ_gain4 * y4 + EQ_gain5 * y5

    return y


def bandpassfilter(x, band, states):

    if band == 1:  # 20-200Hz
        a = [1.0000, -3.8989, 5.7027, -3.7087, 0.9049]
        b = [0.0012, 0, -0.0024, 0, 0.0012]
    elif band == 2:  # 200-400Hz
        a = [1.0000, -3.8651, 5.6264, -3.6561, 0.8949]
        b = [0.0015, 0, -0.0029, 0, 0.0015]
    elif band == 3:  # 400-800Hz
        a = [1.0000, -3.6856, 5.1840, -3.2970, 0.8008]
        b = [0.0055, 0, -0.0111, 0, 0.0055]
    elif band == 4:  # 800-1600Hz
        a = [1.0000, -3.2124, 4.1671, -2.5654, 0.6414]
        b = [0.0201, 0, -0.0402, 0, 0.0201]
    else:  # 1600-7999.5Hz
        a = [1.0000, 0.8568, -0.8731, -0.3172, 0.4127]
        b = [0.6389, 0, -1.2777, 0, 0.6389]

    return signal.lfilter(b, a, x, zi=states)


# Delay Parameters:
DEFAULT_DELAY_TIME = 0.5
delay_buffer = np.zeros(RATE)  # fixed buffer size = 16000
delay_kr = 0
delay_kw = int(DEFAULT_DELAY_TIME * RATE)


def delay(x):
    global delay_kr, delay_kw, delay_buffer

    M = int(Time.get() * RATE)  # new delay time
    if delay_kw > delay_kr:
        current_delay_time = delay_kw - delay_kr
    else:
        current_delay_time = RATE - delay_kr + delay_kw

    # adjust the write index
    if current_delay_time > M:  # delay become shorter
        diff = current_delay_time - M

        if delay_kw > delay_kr:
            delay_kw -= diff
        elif delay_kw - diff < 0:
            delay_kw = RATE - 1 - (diff - delay_kw)
        else:
            delay_kw -= diff
    else:  # delay become longer
        diff = M - current_delay_time

        if delay_kw < delay_kr:
            delay_kw += diff
        elif delay_kw + diff > RATE - 1:
            delay_kw = diff - (RATE - delay_kw)
        else:
            delay_kw += diff

    y, delay_kr, delay_kw, delay_buffer = delayloop(x, delay_kr, delay_kw, delay_buffer)

    return y


def delayloop(x, kr, kw, u_buffer):

    g = Wet.get()
    f = 0.5

    y = np.zeros(BLOCKLEN)
    for n in range(0, BLOCKLEN):
        # y[n] = x[n] + g * u[n]
        # u[n] = x[n-M] + f * u[n-M]

        u = u_buffer[kr]
        y[n] = x[n] + g * u
        u_buffer[kw] = x[n] + f * u

        kr += 1
        if kr == len(u_buffer):
            kr = 0

        kw += 1
        if kw == len(u_buffer):
            kw = 0

    return y, kr, kw, u_buffer


# Reverb Parameters:
reverb_kr1 = 0
reverb_kr2 = 0
reverb_kw = [3301, 2749, 3767, 2399, 1051, 337]
reverb_buffer_index = (reverb_kr1, reverb_kr2, reverb_kw)

REVERB_BUFFER_SIZE1 = 4800
REVERB_BUFFER_SIZE2 = 1600

comb_buffer = [np.zeros(REVERB_BUFFER_SIZE1)] * 4
allpass_buffer = [np.zeros(REVERB_BUFFER_SIZE2)] * 2


def reverb(x):
    global reverb_buffer_index, comb_buffer, allpass_buffer

    ratio = Mix.get()

    y = np.zeros(BLOCKLEN)
    (y, reverb_buffer_index, comb_buffer, allpass_buffer) = reverbloop(
        x, reverb_buffer_index, comb_buffer, allpass_buffer
    )

    y = (1 - ratio) * x + ratio * y

    return y


def reverbloop(x, index, comb_buffer, allpass_buffer):

    kr1 = index[0]
    kr2 = index[1]
    kw = index[2]

    comb1_buffer = comb_buffer[0]
    comb2_buffer = comb_buffer[1]
    comb3_buffer = comb_buffer[2]
    comb4_buffer = comb_buffer[3]

    u_buffer = allpass_buffer[0]
    v_buffer = allpass_buffer[1]

    comb_loop_gain1 = 0.342
    comb_loop_gain2 = 0.333
    comb_loop_gain3 = 0.315
    comb_loop_gain4 = 0.397
    allpass_loop_gain = -0.7

    y = np.zeros(BLOCKLEN)
    for n in range(0, BLOCKLEN):
        # comb: y[n] = x[n-M] + g*y[n-M]
        comb1 = comb1_buffer[kr1]
        comb2 = comb2_buffer[kr1]
        comb3 = comb3_buffer[kr1]
        comb4 = comb4_buffer[kr1]

        y1 = comb1 + comb2 + comb3 + comb4

        # all-pass: y[n] = u[n](1-g**2) - g*x[n]
        #            u[n] = x[n-M] + g*u[n-M]
        u = u_buffer[kr2]
        v = v_buffer[kr2]
        y2 = u * (1 - allpass_loop_gain ** 2) - allpass_loop_gain * y1
        y[n] = v * (1 - allpass_loop_gain ** 2) - allpass_loop_gain * y2

        comb1_buffer[kw[0]] = x[n] + comb_loop_gain1 * comb1
        comb2_buffer[kw[1]] = x[n] + comb_loop_gain2 * comb2
        comb3_buffer[kw[2]] = x[n] + comb_loop_gain3 * comb3
        comb4_buffer[kw[3]] = x[n] + comb_loop_gain4 * comb4

        u_buffer[kw[4]] = y1 + allpass_loop_gain * u
        v_buffer[kw[5]] = y2 + allpass_loop_gain * v

        kr1 += 1
        if kr1 == REVERB_BUFFER_SIZE1:
            kr1 = 0

        kr2 += 1
        if kr2 == REVERB_BUFFER_SIZE2:
            kr2 = 0

        for i in range(4):
            kw[i] += 1
            if kw[i] == REVERB_BUFFER_SIZE1:
                kw[i] = 0

        for i in range(4, 6):
            kw[i] += 1
            if kw[i] == REVERB_BUFFER_SIZE2:
                kw[i] = 0

    return (
        y,
        (kr1, kr2, kw),
        [comb1_buffer, comb2_buffer, comb3_buffer, comb4_buffer],
        [u_buffer, v_buffer],
    )


# left top frame (plot)
Plot_Frame = Tk.LabelFrame(root)
Plot_Frame.grid(row=0, column=0, sticky="ew")
fig = ImageTk.PhotoImage(Image.open("nyu.png").resize((200, 170), Image.ANTIALIAS))

canvas = Tk.Canvas(Plot_Frame, width=170, height=200)
canvas.pack()
canvas.create_image(85, 100, image=fig, anchor="center")


# right top frame (left)
EQ_Frame = Tk.LabelFrame(root, text="Equalizer")
EQ_Frame.grid(row=0, column=1, sticky="ns")

EQ_band1 = Tk.DoubleVar()
EQ_band2 = Tk.DoubleVar()
EQ_band3 = Tk.DoubleVar()
EQ_band4 = Tk.DoubleVar()
EQ_band5 = Tk.DoubleVar()

EQ_band1_Slider = Tk.Scale(
    EQ_Frame, from_=10.0, to=-10.0, variable=EQ_band1, resolution=0.1
)
EQ_band2_Slider = Tk.Scale(
    EQ_Frame, from_=10.0, to=-10.0, variable=EQ_band2, resolution=0.1
)
EQ_band3_Slider = Tk.Scale(
    EQ_Frame, from_=10.0, to=-10.0, variable=EQ_band3, resolution=0.1
)
EQ_band4_Slider = Tk.Scale(
    EQ_Frame, from_=10.0, to=-10.0, variable=EQ_band4, resolution=0.1
)
EQ_band5_Slider = Tk.Scale(
    EQ_Frame, from_=10.0, to=-10.0, variable=EQ_band5, resolution=0.1
)

EQ_band1_Slider.set(0.0)
EQ_band2_Slider.set(0.0)
EQ_band3_Slider.set(0.0)
EQ_band4_Slider.set(0.0)
EQ_band5_Slider.set(0.0)

EQ_band1_Label = Tk.Label(EQ_Frame, text="200Hz")
EQ_band2_Label = Tk.Label(EQ_Frame, text="400Hz")
EQ_band3_Label = Tk.Label(EQ_Frame, text="800Hz")
EQ_band4_Label = Tk.Label(EQ_Frame, text="1600Hz")
EQ_band5_Label = Tk.Label(EQ_Frame, text="3200Hz")
EQ_band_unit_Label1 = Tk.Label(EQ_Frame, text="dB")
EQ_band_unit_Label2 = Tk.Label(EQ_Frame, text="dB")
EQ_band_unit_Label3 = Tk.Label(EQ_Frame, text="dB")
EQ_band_unit_Label4 = Tk.Label(EQ_Frame, text="dB")
EQ_band_unit_Label5 = Tk.Label(EQ_Frame, text="dB")

EQ_band1_Label.grid(row=0, column=0)
EQ_band2_Label.grid(row=0, column=1)
EQ_band3_Label.grid(row=0, column=2)
EQ_band4_Label.grid(row=0, column=3)
EQ_band5_Label.grid(row=0, column=4)
EQ_band1_Slider.grid(row=1, column=0)
EQ_band2_Slider.grid(row=1, column=1)
EQ_band3_Slider.grid(row=1, column=2)
EQ_band4_Slider.grid(row=1, column=3)
EQ_band5_Slider.grid(row=1, column=4)
EQ_band_unit_Label1.grid(row=2, column=0)
EQ_band_unit_Label2.grid(row=2, column=1)
EQ_band_unit_Label3.grid(row=2, column=2)
EQ_band_unit_Label4.grid(row=2, column=3)
EQ_band_unit_Label5.grid(row=2, column=4)


Time = Tk.DoubleVar()
Wet = Tk.DoubleVar()
Mix = Tk.DoubleVar()

# right top frame (middle)
Delay_Frame = Tk.LabelFrame(root, text="Delay")
Delay_Frame.grid(row=0, column=2, sticky="ns")
Delay_band1_Slider = Tk.Scale(
    Delay_Frame, from_=0.99, to=0.01, variable=Time, resolution=0.01
)
Delay_band2_Slider = Tk.Scale(
    Delay_Frame, from_=1.5, to=0.0, variable=Wet, resolution=0.01
)

Delay_band1_Label = Tk.Label(Delay_Frame, text="Time")
Delay_band2_Label = Tk.Label(Delay_Frame, text="Mix")
Delay_band1_unit_Label = Tk.Label(Delay_Frame, text="Sec")
Delay_band2_unit_Label = Tk.Label(Delay_Frame, text="Ratio")

Delay_band1_Label.grid(row=0, column=0)
Delay_band2_Label.grid(row=0, column=1)
Delay_band1_Slider.grid(row=1, column=0)
Delay_band2_Slider.grid(row=1, column=1)
Delay_band1_unit_Label.grid(row=2, column=0)
Delay_band2_unit_Label.grid(row=2, column=1)

# right top frame (right)
Reverb_Frame = Tk.LabelFrame(root, text="Reverb")
Reverb_Frame.grid(row=0, column=3, sticky="ns")
Reverb_band_Slider = Tk.Scale(
    Reverb_Frame, from_=1.00, to=0.00, variable=Mix, resolution=0.01
)

Reverb_band_Label = Tk.Label(Reverb_Frame, text="Mix")
Reverb_band_unit_Label = Tk.Label(Reverb_Frame, text="Ratio")

Reverb_band_Label.grid(row=0, column=0)
Reverb_band_Slider.grid(row=1, column=0)
Reverb_band_unit_Label.grid(row=2, column=0)

Delay_band1_Slider.set(DEFAULT_DELAY_TIME)
Delay_band2_Slider.set(0.0)
Reverb_band_Slider.set(0.00)

# left bottom frame
Rec_Frame = Tk.LabelFrame(root)
Rec_Frame.grid(row=1, column=0, sticky="ns")

MONITOR = Tk.BooleanVar()
# Image for record, stop, and play on headphones
Rec_Button_img = ImageTk.PhotoImage(
    Image.open("rec.png").resize((38, 20), Image.ANTIALIAS)
)
Stop_Button_img = ImageTk.PhotoImage(
    Image.open("stop.png").resize((40, 20), Image.ANTIALIAS)
)
HeadPhone_Button_img = ImageTk.PhotoImage(
    Image.open("headphone.png").resize((28, 28), Image.ANTIALIAS)
)


Rec_Button = Tk.Button(
    Rec_Frame, image=Rec_Button_img, command=lambda: record(RB.get())
)
Stop_Button = Tk.Button(Rec_Frame, image=Stop_Button_img, command=lambda: stop())
Check_Box = Tk.Checkbutton(Rec_Frame, image=HeadPhone_Button_img, variable=MONITOR)

LR_ratio = Tk.DoubleVar()
Input_level = Tk.DoubleVar()
De_noise = Tk.DoubleVar()
Pitch = Tk.DoubleVar()

# Pan_Slider = Tk.Scale(
#     Rec_Frame,
#     from_=-1.0,
#     to=1.0,
#     length=110,
#     width=15,
#     showvalue=1,
#     orient=Tk.HORIZONTAL,
#     variable=LR_ratio,
#     resolution=0.25,
# )
Input_level_Slider = Tk.Scale(Rec_Frame, from_=10.0, to=-10.0, variable=Input_level)
De_noise_Slider = Tk.Scale(Rec_Frame, from_=1000, to=0, variable=De_noise)
Pitch_Slider = Tk.Scale(Rec_Frame, from_=400.0, to=-400.0, variable=Pitch)

# Set initial values of sliders
# Pan_Slider.set(0.0)
Input_level_Slider.set(0.0)
De_noise_Slider.set(0.0)
Pitch_Slider.set(0.0)

Empty_Label = Tk.Label(Rec_Frame, text=" ")
# Pan_Label = Tk.Label(Rec_Frame, text="Pan")
# L_Label = Tk.Label(Rec_Frame, text="L")
# R_Label = Tk.Label(Rec_Frame, text="R")
Input_level_Label = Tk.Label(Rec_Frame, text="Gain")
De_noise_Label = Tk.Label(Rec_Frame, text="Threshold")
Pitch_Label = Tk.Label(Rec_Frame, text="Pitch")
Input_level_unit_Label = Tk.Label(Rec_Frame, text="dB")
De_noise_unit_Label = Tk.Label(Rec_Frame, text="Level")
Pitch_unit_Label = Tk.Label(Rec_Frame, text="Hz")

Empty_Label.grid(row=0, column=0)
Rec_Button.grid(row=1, column=0)
Stop_Button.grid(row=1, column=1)
Check_Box.grid(row=1, column=2)
# L_Label.grid(row=2, column=0)
# Pan_Slider.grid(row=2, column=1)
# R_Label.grid(row=2, column=2)
# Pan_Label.grid(row=3, column=1)
Input_level_Label.grid(row=4, column=0)
De_noise_Label.grid(row=4, column=1)
Pitch_Label.grid(row=4, column=2)
Input_level_Slider.grid(row=5, column=0)
De_noise_Slider.grid(row=5, column=1)
Pitch_Slider.grid(row=5, column=2)
Input_level_unit_Label.grid(row=6, column=0)
De_noise_unit_Label.grid(row=6, column=1)
Pitch_unit_Label.grid(row=6, column=2)


# right bottom frame
Pad_Frame = Tk.LabelFrame(root)
Pad_Frame.grid(row=1, column=1, columnspan=3, sticky="ns")

RB = Tk.IntVar()
RB.set(0)

Radio_Button_1 = Tk.Radiobutton(Pad_Frame, variable=RB, value=1)
Radio_Button_2 = Tk.Radiobutton(Pad_Frame, variable=RB, value=2)
Radio_Button_3 = Tk.Radiobutton(Pad_Frame, variable=RB, value=3)
Radio_Button_4 = Tk.Radiobutton(Pad_Frame, variable=RB, value=4)
Radio_Button_5 = Tk.Radiobutton(Pad_Frame, variable=RB, value=5)
Radio_Button_6 = Tk.Radiobutton(Pad_Frame, variable=RB, value=6)
Radio_Button_7 = Tk.Radiobutton(Pad_Frame, variable=RB, value=7)
Radio_Button_8 = Tk.Radiobutton(Pad_Frame, variable=RB, value=8)
# Radio_Button_9 = Tk.Radiobutton(
#     Pad_Frame, variable=RB, value=9, text="Stop recording..."
# )  # None

Radio_Button_1.grid(row=0, column=0)
Radio_Button_2.grid(row=0, column=1)
Radio_Button_3.grid(row=0, column=2)
Radio_Button_4.grid(row=0, column=3)
Radio_Button_5.grid(row=2, column=0)
Radio_Button_6.grid(row=2, column=1)
Radio_Button_7.grid(row=2, column=2)
Radio_Button_8.grid(row=2, column=3)
# Radio_Button_9.grid(row=4, column=3)

Pad_Button_1 = Tk.Button(Pad_Frame, text="1", padx=40, pady=40, command=lambda: play(1))
Pad_Button_2 = Tk.Button(Pad_Frame, text="2", padx=40, pady=40, command=lambda: play(2))
Pad_Button_3 = Tk.Button(Pad_Frame, text="3", padx=40, pady=40, command=lambda: play(3))
Pad_Button_4 = Tk.Button(Pad_Frame, text="4", padx=40, pady=40, command=lambda: play(4))
Pad_Button_5 = Tk.Button(Pad_Frame, text="5", padx=40, pady=40, command=lambda: play(5))
Pad_Button_6 = Tk.Button(Pad_Frame, text="6", padx=40, pady=40, command=lambda: play(6))
Pad_Button_7 = Tk.Button(Pad_Frame, text="7", padx=40, pady=40, command=lambda: play(7))
Pad_Button_8 = Tk.Button(Pad_Frame, text="8", padx=40, pady=40, command=lambda: play(8))

Pad_Button_1.grid(row=1, column=0)
Pad_Button_2.grid(row=1, column=1)
Pad_Button_3.grid(row=1, column=2)
Pad_Button_4.grid(row=1, column=3)
Pad_Button_5.grid(row=3, column=0)
Pad_Button_6.grid(row=3, column=1)
Pad_Button_7.grid(row=3, column=2)
Pad_Button_8.grid(row=3, column=3)

# root.mainloop()

CONTINUE = True

while CONTINUE:

    root.update()

    MONITORING = MONITOR.get()

    # Monitor Block:
    if MONITORING and not RECORDING:

        input = monitor_stream.read(BLOCKLEN, exception_on_overflow=False)
        original = struct.unpack("h" * BLOCKLEN, input)

        processed = effects(original)

        clipped = np.clip(processed.astype(int), -MAXVALUE, MAXVALUE)
        output = struct.pack("h" * BLOCKLEN, *clipped)

        monitor_stream.write(output)

    # Recording Block:
    if RECORDING and not FINISHED:

        input_bytes = recording_stream.read(BLOCKLEN, exception_on_overflow=False)
        input_tuple = struct.unpack("h" * BLOCKLEN, input_bytes)

        processed_input = effects(input_tuple)

        output_tuple = np.clip(processed_input.astype(int), -MAXVALUE, MAXVALUE)
        output_bytes = struct.pack("h" * BLOCKLEN, *output_tuple)

        if MONITORING:
            recording_stream.write(output_bytes)

        wf.writeframes(output_bytes)

        counter = counter + 1

    if counter >= num_recording_blocks:
        counter = 0
        FINISHED = True
        RECORDING = False
        print("* Finished.")
        print("Saved the wave file: %s." % output_wavfile)
        wf.close()

    # Play Block
    if not RECORDING:
        for i in range(N):
            if TRIGGERED[i]:
                TRIGGERED[i] = False
                PLAYING[i] = True
                END[i] = False
                index[i] = 0
                streams[i].write(byte_data[i][index[i]])
                index[i] = index[i] + 1

        for i in range(N):
            if PLAYING[i] and not TRIGGERED[i] and not END[i]:
                streams[i].write(byte_data[i][index[i]])
                index[i] = index[i] + 1

        for i in range(N):
            if index[i] >= num_playing_blocks[i]:
                END[i] = True
                PLAYING[i] = False
                index[i] = 0


# Close and terminate
for i in range(N):
    streams[i].stop_stream()
    streams[i].close()

recording_stream.stop_stream()
recording_stream.close()
monitor_stream.stop_stream()
monitor_stream.close()
p.terminate()
