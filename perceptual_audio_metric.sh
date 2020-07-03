# Settings
EMOTION_TYPE_SHORT='neg'
MODEL_TYPE="ours"

if [[ $EMOTION_TYPE_SHORT == "neg" ]]; then
  EMOTION_TYPE='negative'
  EMOTION_NUM=0
else
  EMOTION_TYPE='positive'
  EMOTION_NUM=1
fi

# Files path
ROOT="./datasets/cognimuse-3secs/"
# cognimuse-3secs - extra: Neg (14, 28, 33, 36, 38, 40), Pos (4, 5, 26, 29, 35, 37)
# cognimuse-10secs-ep16: Neg (27, 29, 34, 35, 38), Pos (4, 8, 26, 29, 41)
# deap-3secs-25frames-ep25: Neg (9, 12, 14, 31, 33), Pos (4, 5, 14, 19, 24)
SAMPLE=14
if [[ $EMOTION_TYPE_SHORT == "baseline" ]]; then
  AUDIO="${ROOT}/${EMOTION_TYPE}/${EMOTION_TYPE_SHORT}_sample${SAMPLE}/epTest_cnnseq2sample-s${SAMPLE}-em${EMOTION_NUM}_cnnseq2seqAudio.wav"  # baseline
else
  AUDIO="${ROOT}/${EMOTION_TYPE}/${EMOTION_TYPE_SHORT}_sample${SAMPLE}/epTest_cnnseq2sample-s${SAMPLE}-em${EMOTION_NUM}.wav"  # ours
fi
AUDIO_REF="${ROOT}/${EMOTION_TYPE}/${EMOTION_TYPE_SHORT}_sample${SAMPLE}/epTest_cnnseq2sample-s${SAMPLE}-em${EMOTION_NUM}_targetAudio.wav"

# Obtain perceptual audio metric (PAM)
python metric_use_simple.py --e0 $AUDIO --e1 $AUDIO_REF
