# Phoneme Prediction from Speech

## Introduction
As letters are the atomic elements of written language, phonemes are the atomic elements of speech. It is crucial for us to have a means to distinguish different sounds in speech that may or may not represent the same letter or combinations of letters in the written alphabet. This mini-project supports phoneme prediction from speech in two manners, aligned (per frame) and unaligned.

### Phoneme and Phoneme States
For this DNN system we consider 46 phonemes in the english language as shown in the table below.

|  Phoneme  | Phoneme Map | Phoneme | Phoneme Map |
|:---------:|:-----------:|:-------:|:-----------:|
|  +BREATH+ |      _      |    IY   |      I      |
|  +COUGH+  |      +      |    JH   |      j      |
|  +NOISE+  |      ~      |    K    |      k      |
|  +SMACK+  |      !      |    L    |      l      |
|    +UH+   |      -      |    M    |      m      |
|    +UM+   |      @      |    N    |      n      |
|     AA    |      a      |    NG   |      G      |
|     AE    |      A      |    OW   |      O      |
|     AH    |      h      |    OY   |      Y      |
|     AO    |      o      |    P    |      p      |
|     AW    |      w      |    R    |      R      |
|     AY    |      y      |    S    |      s      |
|     B     |      b      |    SH   |      S      |
|     CH    |      c      |   SIL   |      .      |
|     D     |      d      |    T    |      t      |
|     DH    |      D      |    TH   |      T      |
|     EH    |      e      |    UH   |      u      |
|     ER    |      r      |    UW   |      U      |
|     EY    |      e      |    V    |      v      |
|     F     |      f      |    W    |      W      |
|     G     |      g      |    Y    |      ?      |
|     HH    |      H      |    Z    |      z      |
|     IH    |      i      |    ZH   |      Z      |

- __Aligned Phoneme Prediction__: For each phoneme, there are 3 respective phoneme states. Therefore for our 46 phonemes, there exist 138 respective phoneme states. The prediction is made as integers [0-137].

- __Unaligned Phoneme Prediction__: For unaligned prediction, each of the phoneme is mapped to a single character as specified in the "Phoneme Map" columns. The final prediction is a string consisting of these mapped phoneme characters.

## Dataset
Dataset consists of audio recordings (utterances, raw mel spectrogram frames) and corresponding labels sourced from Wall Street Journal (WSJ) dataset.

### Utterance Representation
Note: Below steps are not included in the code and data is expected to already have been preprocessed.

Utterances are expected to be converted to "mel- spectrograms", which are pictorial representations that characterise how the frequency content of the signal varies with time. The frequency domain of the audio signal provides more useful features for distinguishing phonemes.

To convert the speech to a mel-spectrogram, it is segmented into little "frames", each 25ms wide, where the "stride" between adjacent frames is 10ms. Thus we get 100 such frames per second of speech.

From each frame, we compute a single "mel spectral" vector, where the components of the vector represent the (log) energy in the signal in different frequency bands. This DNN system expects as input 40- dimensional mel-spectral vectors, i.e. energies computed in 40 frequency bands.

Thus, we get 100 40-dimensional mel spectral (row) vectors per second of speech in the recording. Each one of these vectors is referred to as a frame. Thus, for a T-second recording, the entire spectrogram is a 100*T x 40 matrix, comprising 100*T 40-dimensional vectors (at 100 vectors (frames) per second).

### Label Representation
- __Aligned Phoneme Prediction__: In aligned phoneme prediction, the labels have a direct mapping to each time step of the feature, that is, they consist of phoneme state (subphoneme) for each frame. These labels are provided as integers [0-137].

- __Unaligned Phoneme Prediction__: In unaligned phoneme prediction, the labels do not have a direct mapping to each time step of the feature, instead they are simply the list of phonemes in the utterance [0-45]. The phoneme array is as long as however many phonemes are in the utterance. The feature data is an array of utterances, whose dimensions are (frames, time step, 40), and the labels will be of the dimension (frames, frequencies). The second dimension, viz., frequencies has variable length which has no correlation to the time step dimension in feature data.

## Models
- __Aligned phoneme prediction__: This system uses a naive MLP. It consists of only linear, batchnorm and activation layers. Similar to using CNN for feature extraction in unaligned phoneme prediction, we use "frame context" in aligned phoneme prediction. Temporal context is important for distinguishing elements of speech. The best accuracy was achieved by using a context of 12 frames on both sides of the key frame. That is, concatenating k=12 mel spectrogram frames around the current time step. This technique makes the size of the input vector 40 * (2k + 1). Since each mel spectrogram frame is only 25ms of the speech audio data, a single frame is unlikely to represent a complete phoneme. Concatenating nearby "k" frames is thus helpful. Here "k" is a hyperparameter.
- __Unaligned phoneme prediction__: On the other hand, the unaligned phoneme prediction uses a CNN layer for feature extraction, followed by four bidirectional LSTM layers and a linear layer for classification. As described above, there is no alignment between utterances and their corresponding phonemes. Thus, this network is trained using CTC loss. The predictions are decoded using beam search.

## Evaluation
- __Aligned Phoneme Prediction__: Performance is measured by classification accuracy on a held out set of labelled mel spectrogram frames.
- __Aligned Phoneme Prediction__: Performance is evaluated using CER - character error rate (edit distance).

## Results
Results are summarized in the table below:

|  Approach |         Criteria        | Test Results |
|:---------:|:-----------------------:|:------------:|
|  Aligned  | Classification Accuracy |     65.91    |
| Unaligned |   CER - Edit Distance   |     6.98     |
