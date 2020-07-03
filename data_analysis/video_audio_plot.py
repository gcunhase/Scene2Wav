from moviepy.editor import VideoFileClip
from cnnseq.utils import project_dir_name, ensure_dir
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

"""
    Result section: Plot video frames and audio waveforms
"""


__author__ = "Gwena Cunha"


params = {
    'data_dir': project_dir_name() + 'samples/data_0/',
    'filename_target': 'set8_epTest_cnnseq2sample-s26-em0_targetAudio_500x500.avi',
    'filename_baseline': 'set8_epTest_cnnseq2sample-s26-em0_cnnseq2seqAudio_500x500.avi',
    'filename_proposed': 'set8_epTest_cnnseq2sample-s26-em0_500x500.avi',
    'plot_filename': 'em0_plot_26.png',
    'audio_fps': 16000,
}


def plot_waveform(data_dir, filename, idx, shared_x=None, colour='b', trim=False, x_axis=True, title=None):
    video_clip = VideoFileClip(data_dir + filename)
    fps = params['audio_fps']
    audio_clip = video_clip.audio
    # https://github.com/Zulko/moviepy/issues/260
    a = audio_clip.to_soundarray(nbytes=4, buffersize=1000, fps=fps)
    a = np.array(a[:, 0]) + np.array(a[:, 1])
    if trim:
        a = a[:-28000]
    if shared_x is None or idx != 4:
        # ax1 = plt.subplot(num_plots, 1, idx)
        ax1 = plt.subplot(outer_grid[(idx-1)*num_frames:num_frames*idx])
        plt.setp(ax1.get_xticklabels(), visible=False)
    else:
        # ax1 = plt.subplot(num_plots, 1, idx, sharex=shared_x)
        ax1 = plt.subplot(outer_grid[(idx-1)*num_frames:num_frames*idx], sharex=shared_x)
    if not x_axis:
        ax1.spines['bottom'].set_visible(False)  # no spine
        ax1.xaxis.set_ticks_position('none')  # tick markers
    else:
        ax1.set_xlabel("Time (s)")
        # Arrow head for bottom spine
        ax1.annotate('', xy=(1, 0), xycoords='axes fraction', xytext=(5, 0),
                     textcoords='offset points', ha='center',
                     arrowprops=dict(arrowstyle='<|-', shrinkA=0, shrinkB=0, facecolor='black'))
    # ax1.get_xaxis().set_ticks([])
    # ax1.get_yaxis().set_ticks([])
    ax1.set_yticks([])
    # plt.ylim(-1.0, 1.0)
    t = np.arange(0.0, float(len(a)), 1.0)/16000  # time
    # t = list(range(0, float(len(a)), 1.0)) / 16000  # time
    plt.plot(t, a, colour)
    if title is not None:
        plt.title(title)
    # plt.title('Audio')
    # Axis line not visible
    for side in ['right', 'top', 'left']:
        ax1.spines[side].set_visible(False)
    # ax1.spines['left'].set_bounds(-0.5, 0.5)
    return ax1, a


data_dir = params['data_dir']
filename_target = params['filename_target']
video_clip = VideoFileClip(data_dir + filename_target)
audio_clip = video_clip.audio

num_plots = 4
num_frames = 7

# fig = plt.figure(figsize=(9, 4))  # change here if images are too far apart
fig = plt.figure(figsize=(8.7, 4))  # change here if images are too far apart
outer_grid = gridspec.GridSpec(num_plots, num_frames, wspace=0.0, hspace=0.8, height_ratios=[3, 1, 1, 1])

# Plot frames spread evenly in video
frame_arr = []
frames = int(video_clip.fps * video_clip.duration)
total_frames = int(frames / num_frames)
print("Frames {}, total wanted {}, period {}".format(frames, num_frames, total_frames))
for f, frame in enumerate(video_clip.iter_frames()):
    if f % total_frames == 0:
        frame_arr.append(frame)
        print(f)

    if len(frame_arr) == num_frames:
        break

for i in range(1, num_frames+1):
    # plt.subplot(2, num_frames, i)
    plt.subplot(outer_grid[i-1])
    plt.imshow(frame_arr[i-1])
    plt.axis('off')

ax1, audio1 = plot_waveform(data_dir, params['filename_target'], idx=2, colour='g', x_axis=False, title="Original audio")
ax2, audio2 = plot_waveform(data_dir, params['filename_baseline'], idx=3, shared_x=ax1, colour='y', x_axis=False, title="Baseline model")
_, audio3 = plot_waveform(data_dir, params['filename_proposed'], idx=4, shared_x=ax1, colour='b', x_axis=True, title="Proposed model")

plt.tight_layout()
plt.savefig(params['data_dir']+params['plot_filename'])
# plt.show()

# Correlation
# R: correlation coefficient matrix of the variables
r1 = np.corrcoef(audio1, audio2)
r2 = np.corrcoef(audio1, audio3)
r3 = np.corrcoef(audio2, audio3)
print("Correlation coefficient between")
print("- target and baseline {}".format(r1[0, 1]))
print("- target and proposed {}".format(r2[0, 1]))
print("- baseline and proposed {}".format(r3[0, 1]))

plt.show()

'''
# Spectrogram: ["linear", "linear_grayscale", "log_power"]
plt.figure()
plt.specgram(audio1, NFFT=256, Fs=16000, noverlap=128)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.savefig(data_dir+params['plot_filename'].split('.png')[0]+"_audio1.png")
plt.figure()
plt.specgram(audio2, NFFT=256, Fs=16000, noverlap=128)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.savefig(data_dir+params['plot_filename'].split('.png')[0]+"_audio2.png")
plt.figure()
plt.specgram(audio3, NFFT=256, Fs=16000, noverlap=128)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.savefig(data_dir+params['plot_filename'].split('.png')[0]+"_audio3.png")
'''

