#!/bin/bash

DATA_TYPE='deap_seq2seq_ep20'
NUM_FRAMES=25
EP_NUM=45
EM_TYPE='negative'
DURATION=3

echo "Dataset ${DATA_TYPE} with ${EM_TYPE} emotion for ${DURATION} seconds"

if [[ $DATA_TYPE = "cognimuse" ]]; then
  if [[ $EM_TYPE = "negative" ]]; then
    EM_TYPE_SHORT='neg'
    EM_TYPE_INT=0
    if [ $DURATION = 3 ]; then
      SAMPLES=(2 14 20 27 28 33 34 36 38 40)
    else  # 10secs
      if [[ $EP_NUM = 11 ]]; then
        SAMPLES=(3 5 7 9 13 19 20 35 36 41)
      elif [[ $EP_NUM = 12 ]]; then
        SAMPLES=(14 15 16 22 35 39)
      elif [[ $EP_NUM = 16 ]]; then
        SAMPLES=(1 3 4 5 6 7 10 11 14 16 18 19 24 26 27 28 29 31 32 34 35 37 38 40)
      elif [[ $EP_NUM = 17 ]]; then
        SAMPLES=(2 4 5 7 10 15 21 23)
      else
        SAMPLES=(1 7 16 24 27)
      fi
    fi
  else
    EM_TYPE_SHORT='pos'
    EM_TYPE_INT=1
    if [ $DURATION = 3 ]; then
      SAMPLES=(4 5 7 13 26 29 32 35 37)
    else  # 10secs
      if [[ $EP_NUM = 11 ]]; then
        SAMPLES=(7 9 11 21 26 41)
      elif [[ $EP_NUM = 12 ]]; then
        SAMPLES=(17 27 28 31 41 42)
      elif [[ $EP_NUM = 16 ]]; then
        SAMPLES=(2 4 5 6 7 8 9 10 12 13 16 18 20 21 22 25 26 27 29 31 32 36 37 41)
      elif [[ $EP_NUM = 17 ]]; then
        SAMPLES=(4 9 10 11 18 22)
      else
        SAMPLES=(7 13 14 15)
      fi
    fi
  fi
elif [[ $DATA_TYPE = "deap" ]]; then
  if [[ $EM_TYPE = "negative" ]]; then
    EM_TYPE_SHORT='neg'
    EM_TYPE_INT=0
    if [[ $NUM_FRAMES = 16 ]]; then
      if [[ $EP_NUM = 25 ]]; then
        SAMPLES=(10 11 15 25)
      else  # 23
        SAMPLES=(6 7 8 9 13 22 25)
      fi
    else  # 25 frames
      if [[ $EP_NUM = 19 ]]; then
        SAMPLES=(1 6 14 15 17 23 24)
      elif [[ $EP_NUM = 25 ]]; then
        SAMPLES=(1 2 3 4 5 7 8 9 12 13 14 17 18 19 21 28 31 33 34)
      else
        SAMPLES=(6 10 18 20 21 23)
      fi
    fi
  else
    EM_TYPE_SHORT='pos'
    EM_TYPE_INT=1
    if [[ $NUM_FRAMES = 16 ]]; then
      if [[ $EP_NUM = 25 ]]; then
        SAMPLES=(8 13 14 24)
      else  # 23
        SAMPLES=(1 10 14 16 21 22 24)
      fi
    else # 25 frames
      if [[ $EP_NUM = 19 ]]; then
        SAMPLES=(8 14 21 23)
      elif [[ $EP_NUM = 25 ]]; then
        SAMPLES=(2 3 4 5 7 10 14 19 21 24 26 27 31 32 34)
      else
        SAMPLES=(5 7 12 18 19 24)
      fi
    fi
  fi
elif [[ $DATA_TYPE = "deap_seq2seq_ep20" ]]; then
  if [[ $EM_TYPE = "negative" ]]; then
    EM_TYPE_SHORT='neg'
    EM_TYPE_INT=0
    if [[ $EP_NUM = 25 ]]; then
      SAMPLES=(23 24 25 26 27 33)
    elif [[ $EP_NUM = 45 ]]; then
      SAMPLES=(12 14 23 24 25 26 28 29)
    fi
  else
    EM_TYPE_SHORT='pos'
    EM_TYPE_INT=1
    if [[ $EP_NUM = 25 ]]; then
      SAMPLES=(15 21 22 23)
    elif [[ $EP_NUM = 45 ]]; then
      SAMPLES=(2 4 9 12 19 21 24 27)
    fi
  fi
else  # deap_raw
  if [[ $EM_TYPE = "negative" ]]; then
    EM_TYPE_SHORT='neg'
    EM_TYPE_INT=0
    if [[ $EP_NUM = 8 ]]; then
      SAMPLES=(3 4 8 9 16 18 20 22 24)
    elif [[ $EP_NUM = 10 ]]; then
      SAMPLES=(4 5 6 15 18 25)
    elif [[ $EP_NUM = 17 ]]; then
      SAMPLES=(1 3 4 5 9 15 16 19 20 22 24)
    else
      SAMPLES=(7 13 15 16 17 20 22 24)
    fi
  else
    EM_TYPE_SHORT='pos'
    EM_TYPE_INT=1
    if [[ $EP_NUM = 8 ]]; then
      SAMPLES=(7 9 10 14 15 23 25)
    elif [[ $EP_NUM = 10 ]]; then
      SAMPLES=(10 12 17 20 25)
    elif [[ $EP_NUM = 17 ]]; then
      SAMPLES=(1 4 5 8 19 20 22)
    else
      SAMPLES=(7 10 16 21)
    fi
  fi
fi
# Original samples
#SAMPLES=(1 2 3 4)
#DATA_DIR="./datasets/original_${DURATION}secs/${EM_TYPE}/"

for SAMPLE in ${SAMPLES[*]}; do
  # Original sample
  #FILE_ROOT="${EM_TYPE_SHORT}${SAMPLE}_baseline"
  #FILE_ROOT="${EM_TYPE_SHORT}${SAMPLE}_ours_scene2wav"
  #FILE_ROOT="${EM_TYPE_SHORT}${SAMPLE}_target"

  # New samples
  if [[ $DATA_TYPE = "cognimuse" ]]; then
    DATA_DIR="./datasets/${DATA_TYPE}-${DURATION}secs-ep${EP_NUM}/${EM_TYPE}/${EM_TYPE_SHORT}_sample${SAMPLE}/"
  elif [[ $DATA_TYPE = "deap_seq2seq_ep20" ]]; then
    DATA_DIR="./datasets/deap-${DURATION}secs-${NUM_FRAMES}frames-ep${EP_NUM}-seq2seq_ep20_resume/${EM_TYPE}/${EM_TYPE_SHORT}_sample${SAMPLE}/"
  else
    DATA_DIR="./datasets/${DATA_TYPE}-${DURATION}secs-${NUM_FRAMES}frames-ep${EP_NUM}/${EM_TYPE}/${EM_TYPE_SHORT}_sample${SAMPLE}/"
  fi

  # Proposed: Scene2Wav
  FILE_ROOT="epTest_cnnseq2sample-s${SAMPLE}-em${EM_TYPE_INT}"
  INFILE="${FILE_ROOT}.wav"
  OUTFILE="${FILE_ROOT}.mid"
  python emotion_evaluation.py --data_dir $DATA_DIR --infile $INFILE --outfile $OUTFILE

  # Baseline
  FILE_ROOT="epTest_cnnseq2sample-s${SAMPLE}-em${EM_TYPE_INT}_cnnseq2seqAudio"
  INFILE="${FILE_ROOT}.wav"
  OUTFILE="${FILE_ROOT}.mid"
  python emotion_evaluation.py --data_dir $DATA_DIR --infile $INFILE --outfile $OUTFILE

  # Target
  FILE_ROOT="epTest_cnnseq2sample-s${SAMPLE}-em${EM_TYPE_INT}_targetAudio"
  INFILE="${FILE_ROOT}.wav"
  OUTFILE="${FILE_ROOT}.mid"
  python emotion_evaluation.py --data_dir $DATA_DIR --infile $INFILE --outfile $OUTFILE
done
