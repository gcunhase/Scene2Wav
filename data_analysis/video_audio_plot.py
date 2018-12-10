from cnnseq.utils import project_dir_name, ensure_dir

"""
    Result section: Video frames and audio waveform plot for paper
"""

__author__ = "Gwena Cunha"

params = {
    'root_dir': project_dir_name() + 'data_analysis/res_paper/',
    'videos_filename_pattern': 'video{}.avi',
    'target_audio_filename_pattern': 'target_audio{}.wav',
    'baseline_audio_filename_pattern': 'baseline_audio{}.wav',
    'proposed_audio_filename_pattern': 'proposed_audio{}.wav',
}

