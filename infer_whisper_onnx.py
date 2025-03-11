import time
import os
import numpy as np
import audio2numpy as a2n  # import here, as this requires ffmpeg to be installed on host machine
from scipy import special as scipy_special  # type: ignore
# import sys
# sys.path.append(r"D:\algorithm\robot_det_face\whisper_learn\whisperkittools\whisper")
import tokenizer_pt 


from onnxruntime import InferenceSession
class whisperEncoderOnnx:
    def __init__(self,path=''):
        super().__init__()
        #path=r"D:\algorithm\robot_det_face\whisper_learn\models\whisper-base-en\WhisperEncoder.onnx"

        self.yolo_session = InferenceSession(path, providers=['CPUExecutionProvider',"CUDAExecutionProvider"])  #CUDAExecutionProvider
        self.input_names = [input.name for input in self.yolo_session.get_inputs()]

    def __call__(self, audio):
        onnx_inputs = {
            self.input_names[0]: audio,
        }
        output = self.yolo_session.run(None, onnx_inputs)
        return output

class whisperDecoderOnnx:
    def __init__(self,path=''):
        super().__init__()
        #path=r"D:\algorithm\robot_det_face\whisper_learn\models\whisper-base-en\WhisperDecoder.onnx"

        self.yolo_session = InferenceSession(path, providers=['CPUExecutionProvider',"CUDAExecutionProvider"])  #CUDAExecutionProvider
        self.input_names = [input.name for input in self.yolo_session.get_inputs()]

    def __call__(self, x,index,k_cache_cross,v_cache_cross,k_cache_self,v_cache_self):
        onnx_inputs = {
                self.input_names[0]: x,
                self.input_names[1]: index,
                self.input_names[2]: k_cache_cross,
                self.input_names[3]: v_cache_cross,
                self.input_names[4]: k_cache_self,
                self.input_names[5]: v_cache_self,
            }
        output = self.yolo_session.run(None, onnx_inputs)
        return output
####
import torch

# Adopted from https://github.com/openai/whisper/blob/main/whisper/audio.py
def log_mel_spectrogram(
    mel_filter: np.ndarray,
    audio_np: np.ndarray,
    pad_to_length: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio_np: np.ndarray, shape = (*)

    pad_to_length: int
        Add zero samples to the right till this length. No op if
        len(audio) >= pad_to_length

    Returns
    -------
    np.ndarray, shape = (1, 80, n_frames)
        A Tensor that contains the Mel spectrogram. n_frames = 3000 for whisper
    """
    audio = torch.from_numpy(audio_np)
    assert isinstance(audio, torch.Tensor)

    if pad_to_length is not None:
        padding = pad_to_length - len(audio)
        if padding > 0:
            audio = torch.nn.functional.pad(audio, (0, padding))
    window = torch.hann_window(n_fft)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    # mel_spec = torch.from_numpy(mel_filter) @ magnitudes
    mel_spec = mel_filter @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.unsqueeze(0).detach().float().numpy()





import samplerate
import torch.nn.functional as F


CHUNK_LENGTH = 30
# 20ms sample rate
SAMPLE_RATE = 16000

N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk

MEAN_DECODE_LEN = 224


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array

def chunk_and_resample_audio(
    audio: np.ndarray,
    audio_sample_rate: int,
    model_sample_rate=SAMPLE_RATE,
    model_chunk_seconds=CHUNK_LENGTH,
) -> list[np.ndarray]:
    """
    Parameters
    ----------
    audio: str
        Raw audio numpy array of shape [# of samples]

    audio_sample_rate: int
        Sample rate of audio array, in samples / sec.

    model_sample_rate: int
        Sample rate (samples / sec) required to run Whisper. The audio file
        will be resampled to use this rate.

    model_chunk_seconds: int
        Split the audio in to N sequences of this many seconds.
        The final split may be shorter than this many seconds.

    Returns
    -------
    List of audio arrays, chunked into N arrays of model_chunk_seconds seconds.
    """
    if audio_sample_rate != model_sample_rate:
        audio = samplerate.resample(audio, model_sample_rate / audio_sample_rate)
        audio_sample_rate = model_sample_rate

    number_of_full_length_audio_chunks = (
        audio.shape[0] // audio_sample_rate // model_chunk_seconds
    )
    last_sample_in_full_length_audio_chunks = (
        audio_sample_rate * number_of_full_length_audio_chunks * model_chunk_seconds
    )

    if number_of_full_length_audio_chunks == 0:
        return [audio]

    return [
        *np.array_split(
            audio[:last_sample_in_full_length_audio_chunks],
            number_of_full_length_audio_chunks,
        ),
        audio[last_sample_in_full_length_audio_chunks:],
    ]
########

# Whisper constants
TOKEN_SOT = 50257  # Start of transcript
TOKEN_EOT = 50256  # end of transcript
TOKEN_BLANK = 220  # " "
TOKEN_NO_TIMESTAMP = 50362
TOKEN_TIMESTAMP_BEGIN = 50363
TOKEN_NO_SPEECH = 50361

# Above this prob we deem there's no speech in the audio
NO_SPEECH_THR = 0.6

# https://github.com/openai/whisper/blob/v20230314/whisper/decoding.py#L600
NON_SPEECH_TOKENS = [
    1,
    2,
    7,
    8,
    9,
    10,
    14,
    25,
    26,
    27,
    28,
    29,
    31,
    58,
    59,
    60,
    61,
    62,
    63,
    90,
    91,
    92,
    93,
    357,
    366,
    438,
    532,
    685,
    705,
    796,
    930,
    1058,
    1220,
    1267,
    1279,
    1303,
    1343,
    1377,
    1391,
    1635,
    1782,
    1875,
    2162,
    2361,
    2488,
    3467,
    4008,
    4211,
    4600,
    4808,
    5299,
    5855,
    6329,
    7203,
    9609,
    9959,
    10563,
    10786,
    11420,
    11709,
    11907,
    13163,
    13697,
    13700,
    14808,
    15306,
    16410,
    16791,
    17992,
    19203,
    19510,
    20724,
    22305,
    22935,
    27007,
    30109,
    30420,
    33409,
    34949,
    40283,
    40493,
    40549,
    47282,
    49146,
    50257,
    50357,
    50358,
    50359,
    50360,
    50361,
]

SAMPLE_BEGIN = 1  # first token is TOKEN_SOT

# https://github.com/openai/whisper/blob/v20230314/whisper/decoding.py#L545
precision = 0.02  # in second
max_initial_timestamp = 1.0  # in second
max_initial_timestamp_index = int(max_initial_timestamp / precision)

def apply_timestamp_rules(
    logits: np.ndarray, tokens: list[int]
) -> tuple[np.ndarray, float]:
    """
    When predicting timestamps, there are a few post processing rules /
    heuristics to ensure well-formed timestamps. See in-line comments for details

    Args:
    - logits: of shape (51864,)

    Returns:

    - modified logits
    - log probability of modified logits (log(softmax(logits)))
    """
    # Require producing timestamp
    logits[TOKEN_NO_TIMESTAMP] = -np.inf

    # timestamps have to appear in pairs, except directly before EOT
    seq = tokens[SAMPLE_BEGIN:]
    last_was_timestamp = len(seq) >= 1 and seq[-1] >= TOKEN_TIMESTAMP_BEGIN
    penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= TOKEN_TIMESTAMP_BEGIN
    if last_was_timestamp:
        if penultimate_was_timestamp:  # has to be non-timestamp
            logits[TOKEN_TIMESTAMP_BEGIN:] = -np.inf
        else:  # cannot be normal text tokens
            logits[:TOKEN_EOT] = -np.inf

    timestamps = [t for t in tokens if t >= TOKEN_TIMESTAMP_BEGIN]
    if len(timestamps) > 0:
        # timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
        # also force each segment to have a nonzero length, to   prevent infinite looping
        if last_was_timestamp and not penultimate_was_timestamp:
            timestamp_last = timestamps[-1]
        else:
            timestamp_last = timestamps[-1] + 1
        logits[TOKEN_TIMESTAMP_BEGIN:timestamp_last] = -np.inf

    if len(tokens) == SAMPLE_BEGIN:
        # suppress generating non-timestamp tokens at the beginning
        logits[:TOKEN_TIMESTAMP_BEGIN] = -np.inf

        # apply the `max_initial_timestamp` option
        last_allowed = TOKEN_TIMESTAMP_BEGIN + max_initial_timestamp_index
        logits[(last_allowed + 1) :] = -np.inf

    # if sum of probability over timestamps is above any other token, sample timestamp
    logprobs = scipy_special.log_softmax(logits)
    timestamp_logprob = scipy_special.logsumexp(logprobs[TOKEN_TIMESTAMP_BEGIN:])
    max_text_token_logprob = logprobs[:TOKEN_TIMESTAMP_BEGIN].max()
    if timestamp_logprob > max_text_token_logprob:
        # Mask out all but timestamp tokens
        logits[:TOKEN_TIMESTAMP_BEGIN] = -np.inf

    return logits, logprobs




class WhisperPredict:
    def __init__(self,encoder,decoder,type_model ="base"):
        mel_filter: np.ndarray | None = None,
        sample_rate: int = 16000  #SAMPLE_RATE,
        max_audio_seconds: int = 30 #CHUNK_LENGTH,
        n_fft: int =400 # N_FFT,
        hop_length: int =160 # HOP_LENGTH,
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds
        self.mean_decode_len = MEAN_DECODE_LEN
        self.type_model = type_model
        if self.type_model =="tiny":
             ## base
            self.num_decoder_blocks = 4  #len(self.decoder.blocks)
            self.num_decoder_heads = 6# self.decoder.blocks[0].attn.n_head
            self.attention_dim = 384  #self.decoder.blocks[0].attn_ln.weight.shape[0]
            # end base
        else:
            ## base
            self.num_decoder_blocks = 6  #len(self.decoder.blocks)
            self.num_decoder_heads = 8# self.decoder.blocks[0].attn.n_head
            self.attention_dim = 512  #self.decoder.blocks[0].attn_ln.weight.shape[0]
            # end base
        
        self.n_fft = n_fft
        self.max_audio_samples = self.max_audio_seconds * self.sample_rate
        self.encoder = encoder
        self.decoder = decoder
        n_mels = 80
        
        filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
        # filters_path="whisper/whisper/assets/mel_filters.npz"
        with np.load(filters_path, allow_pickle=False) as f:
            self.mel_filter =  torch.from_numpy(f[f"mel_{n_mels}"])
        
    def transcribe(self, audio: np.ndarray | str, audio_sample_rate: int | None = None
    ) -> str:
        result = " ".join(
            self._transcribe_single_chunk(x)
            for x in chunk_and_resample_audio(audio, audio_sample_rate)
        )
        return result
    
    def _transcribe_single_chunk(self, audio: np.ndarray) -> str:
        """
        Transcribe an audio chunk to text.

        Parameters:

        audio: numpy array
            A numpy array of audio of shape (number of samples).
            The sample rate of this audio must be self.sample_rate.
            The maximum length of this audio must be self.max_audio_samples.

        Returns:

        - transcribed texts
        """
        audio = pad_or_trim(audio)
        mel_input = log_mel_spectrogram(
            self.mel_filter, audio, self.max_audio_samples, self.n_fft, self.hop_length
        )
        path_f =r"onnx-models\qnn_tiny_data"

        if False:
            audio_path = os.path.join(path_f,f"audio_{mel_input.shape[0]}x{mel_input.shape[1]}x{mel_input.shape[2]}.raw")

            # mel_input.tofile(audio_path)
            tmp = np.transpose(mel_input, axes=(0, 2, 1))
            tmp.tofile(audio_path)

        #  [6, 8, 64, 1500], [6, 8, 1500, 64]
        k_cache_cross, v_cache_cross = self.encoder(mel_input)
        # k_path=r"D:\algorithm\robot_det_face\whisper_learn\whisperkittools\out_qnn\output_qnn\Result_0\k_cache.raw"
        # v_path=r"D:\algorithm\robot_det_face\whisper_learn\whisperkittools\out_qnn\output_qnn\Result_0\v_cache.raw"
        # k_cache_cross_t = np.fromfile(k_path, dtype=np.float32)
        # k_cache_cross_t=k_cache_cross_t.reshape(4, 6, 64, 1500)
        # v_cache_cross_t = np.fromfile(v_path, dtype=np.float32)
        # v_cache_cross_t=v_cache_cross_t.reshape(4, 6, 1500, 64)



        # Start decoding
        # coreml only takes float tensors
        # x = np.array([[TOKEN_SOT]],dtype=np.int32)
        x = np.array([[TOKEN_SOT]],dtype=np.float32)
        decoded_tokens = [TOKEN_SOT]
        sample_len = self.mean_decode_len  # mean # of tokens to sample
        k_cache_self = np.zeros(
            (
                self.num_decoder_blocks,
                self.num_decoder_heads,
                self.attention_dim // self.num_decoder_heads,
                sample_len,
            )
        ).astype(np.float32)    #base 6,8,64,224
        v_cache_self = np.zeros(
            (
                self.num_decoder_blocks,
                self.num_decoder_heads,
                sample_len,
                self.attention_dim // self.num_decoder_heads,
            )
        ).astype(np.float32)  # base 6,8,224,64

        sum_logprobs = 0
        t1= time.time()
        for i in range(sample_len):
            # Using i to index inside the decoder model hurts the
            # the model performance.
            # index - used to get positional embedding correctly.
            # index = torch.zeros([1, 1], dtype=torch.int32)
            # index = np.zeros([1, 1], dtype=np.int32)
            index = np.zeros([1, 1], dtype=np.float32)
            index[0, 0] = i
            if False:
                x_path = os.path.join(path_f,f"x_{x.shape[0]}x{x.shape[1]}.raw")
                x.tofile(x_path)
                index_path = os.path.join(path_f,f"index_{index.shape[0]}x{index.shape[1]}.raw")
                index.tofile(index_path)
                k_cache_cross_path = os.path.join(path_f,f"k_cache_cross_{k_cache_cross.shape[0]}x{k_cache_cross.shape[1]}x{k_cache_cross.shape[2]}x{k_cache_cross.shape[3]}.raw")
                k_cache_cross.tofile(k_cache_cross_path)
                v_cache_cross_path = os.path.join(path_f,f"v_cache_cross_{v_cache_cross.shape[0]}x{v_cache_cross.shape[1]}x{v_cache_cross.shape[2]}x{v_cache_cross.shape[3]}.raw")
                v_cache_cross.tofile(v_cache_cross_path)
                k_cache_self_path = os.path.join(path_f,f"k_cache_self_{k_cache_self.shape[0]}x{k_cache_self.shape[1]}x{k_cache_self.shape[2]}x{k_cache_self.shape[3]}.raw")
                k_cache_self.tofile(k_cache_self_path)
                v_cache_self_path = os.path.join(path_f,f"v_cache_self_{v_cache_self.shape[0]}x{v_cache_self.shape[1]}x{v_cache_self.shape[2]}x{v_cache_self.shape[3]}.raw")
                v_cache_self.tofile(v_cache_self_path)



            decoder_out = self.decoder(
                x, index, k_cache_cross, v_cache_cross, k_cache_self, v_cache_self
            )
            # logit has shape (1, decoded_len, 51864)
            logits = decoder_out[0]
            k_cache_self = decoder_out[1]
            v_cache_self = decoder_out[2]

            # logit has shape (51864,)
            logits = logits[0, -1]  # consider only the last token

            # Filters
            # SuppressBlank
            if i == 0:
                logits[[TOKEN_EOT, TOKEN_BLANK]] = -np.inf
            # SuppressTokens
            logits[NON_SPEECH_TOKENS] = -np.inf

            logits, logprobs = apply_timestamp_rules(logits, decoded_tokens)

            if i == 0:
                # detect no_speech
                no_speech_prob = np.exp(logprobs[TOKEN_NO_SPEECH])
                if no_speech_prob > NO_SPEECH_THR:
                    break

            # temperature = 0
            next_token = np.argmax(logits)
            if next_token == TOKEN_EOT:
                break

            sum_logprobs += logprobs[next_token]
            # x = np.array([[next_token]],dtype=np.int32)
            x = np.array([[next_token]],dtype=np.float32)
            decoded_tokens.append(int(next_token))
        cost_time = (time.time()-t1)*1000
        print(f"inference cost time:{cost_time}")
        tokenizer = tokenizer_pt.get_tokenizer(
            multilingual=False, language="en", task="transcribe"
        )

        text = tokenizer.decode(decoded_tokens[1:])  # remove TOKEN_SOT
        return text.strip()

import wave

def _buf_to_float(x, n_bytes=2, dtype=np.float32):
    # Invert the scale of the data
    scale = 1. / float(1 << ((8 * n_bytes) - 1))
    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)
    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)

def read_wav_file_new(file_path):
    # Load audio clip as 16-bit PCM data
    frame_rate =0
    with wave.open(file_path, mode='rb') as f:
        # Load WAV clip frames
        params = f.getparams()
        n_channels, samp_width, frame_rate, n_frames = params[:4]
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        result =[]
        for va in data:
            result.append(_buf_to_float(va))
    if result:
        return np.concatenate(result),frame_rate
    else:
        y = np.empty(0, dtype=np.float32)
        return y , 0

if __name__ == "__main__":

    type_model = "tiny"

    en_path=r"onnx-models\tiny.en\encoder_pt.onnx"
    de_path=r"onnx-models\tiny.en\decoder_pt.onnx"

    # en_path=r"onnx-models\tiny.en\encoder_pt_l.onnx"
    # de_path=r"onnx-models\tiny.en\decoder_pt_l.onnx"


    # type_model = "base"
    # # en_path =r"D:\algorithm\robot_det_face\whisper_learn\models\whisper-base-en\WhisperEncoder.onnx"
    # # de_path =r"D:\algorithm\robot_det_face\whisper_learn\models\whisper-base-en\WhisperDecoder.onnx"
    # en_path=r"onnx-models\base.en\encoder_pt_l.onnx"
    # de_path=r"onnx-models\base.en\decoder_pt_l.onnx"


    encoder_model =whisperEncoderOnnx(en_path)
    decoder_model= whisperDecoderOnnx(de_path)


    app =WhisperPredict(encoder_model,decoder_model,type_model = type_model)
    folder_path="./data"
    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('wav'))]
    # audio_path="data/998.wav"
    # audio_path="data/common_voice_en_106930.wav"
    # audio_, audio_sample_rate_ = a2n.audio_from_file(audio_path)

    for audio_f in audio_files: 
        audio_path  = os.path.join(folder_path, audio_f)  
        audio_path=r"D:\algorithm\robot_det_face\whisper_learn\whisperkittools\data\common_voice_en_113714.wav"
        # audio_path=r"D:\algorithm\robot_det_face\whisper_learn\ouput_16k.wav"
        audio, audio_sample_rate = read_wav_file_new(audio_path)
        t0=time.time()
        transcription = app.transcribe(audio, audio_sample_rate)
        t1=time.time()
        cost_time = (t1-t0)*1000
        
        print("Transcription:", transcription)
        print(f" cost time: {cost_time} ms")
    
