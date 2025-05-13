# import os
import numpy as np
import whisper
from scipy import special as scipy_special
import librosa
from transformers import WhisperProcessor
import time
import argparse
# Whisper constants
TOKEN_SOT = 50257  # Start of transcript
TOKEN_EOT = 50256  # end of transcript
TOKEN_NO_TIMESTAMP = 50362
TOKEN_BLANK = 220  # " "
TOKEN_TIMESTAMP_BEGIN = 50363
TOKEN_NO_SPEECH = 50361

MEAN_DECODE_LEN = 224

# Above this prob we deem there's no speech in the audio
NO_SPEECH_THR = 0.6

SAMPLE_BEGIN = 1  # first token is TOKEN_SOT

# https://github.com/openai/whisper/blob/v20230314/whisper/decoding.py#L545
precision = 0.02  # in second
max_initial_timestamp = 1.0  # in second
max_initial_timestamp_index = int(max_initial_timestamp / precision)

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


def run_whisper_encoder(encoder_model, audio,invoke_nums=1):
    
    invoke_time=[]
    for i in range(invoke_nums):
        t1=time.time()
        result =encoder_model(audio)
        cost_time = (time.time()-t1)*1000
        invoke_time.append(cost_time)
        if result != 0:
            print("interpreter set_input_tensor() failed")  

    # qnn_out=[]
    # TODO 尺寸不对，需要调整
    k_cache = result[0]#qnn_out_0.reshape(6,8,64,1500)
    v_cache = result[1] #qnn_out_1.reshape(6,8,1500,64)
    max_invoke_time = max(invoke_time)
    min_invoke_time = min(invoke_time)
    mean_invoke_time = sum(invoke_time)/invoke_nums
    var_invoketime=np.var(invoke_time)
    print("====================================")
    print(f"QNN encoder invoke time:\n --mean_invoke_time is {mean_invoke_time} \n --max_invoke_time is {max_invoke_time} \n --min_invoke_time is {min_invoke_time} \n --var_invoketime is {var_invoketime}")
    print("====================================")
    
    return k_cache, v_cache

def run_whisper_decoder(decoder_model, x, index, k_cache_cross, v_cache_cross, k_cache_self, v_cache_self,invoke_nums=1):
    # pdb.set_trace()
    # print("k_cache_cross:",k_cache_cross) 

    invoke_time=[]
    for i in range(invoke_nums):
        t1=time.time()
        result = decoder_model(x, index, k_cache_cross, v_cache_cross, k_cache_self, v_cache_self)
        cost_time = (time.time()-t1)*1000
        invoke_time.append(cost_time)
        if result != 0:
            print("interpreter set_input_tensor() failed")

    
    logits =result[0] #qnn_out_0.reshape(1,1,51864).copy()
    k_cache =result[1] #qnn_out_1.reshape(6,8,64,224).copy()
    v_cache =result[2] #qnn_out_2.reshape(6,8,224,64).copy()
    
    
    max_invoke_time = max(invoke_time)
    min_invoke_time = min(invoke_time)
    mean_invoke_time = sum(invoke_time)/invoke_nums
    var_invoketime=np.var(invoke_time)
    print("====================================")
    print(f"QNN decoder invoke time:\n --mean_invoke_time is {mean_invoke_time} \n --max_invoke_time is {max_invoke_time} \n --min_invoke_time is {min_invoke_time} \n --var_invoketime is {var_invoketime}")
    print("====================================")

    return logits, k_cache, v_cache



def apply_timestamp_rules(
    logits: np.ndarray, tokens: list[int]
) -> tuple[np.ndarray, float | np.ndarray]:
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


def aidlite_run(args):
    # aidlite.set_log_level(aidlite.LogLevel.INFO)
    # aidlite.log_to_stderr()

    en_path =r"D:\algorithm\robot_det_face\whisper_learn\models\whisper-base-en\WhisperEncoder.onnx"
    de_path =r"D:\algorithm\robot_det_face\whisper_learn\models\whisper-base-en\WhisperDecoder.onnx"

    encoder_model =whisperEncoderOnnx(en_path)
    decoder_model= whisperDecoderOnnx(de_path)


    data_dir = "data"
    num_decoder_blocks = 6
    attention_dim = 512
    num_decoder_heads = 8
    mean_decode_len = MEAN_DECODE_LEN
    # processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
    processor = WhisperProcessor.from_pretrained("./tokensf")
    sample1, sample_rate1 = librosa.load(args.audio, sr=16000)
    audio_data=processor(sample1, sampling_rate=16000, return_tensors="pt").input_features.numpy()
    k_cache_cross, v_cache_cross = run_whisper_encoder(encoder_model, audio_data,args.invoke_nums)
    # pdb.set_trace()

    
    # pdb.set_trace()
    x = np.array([[TOKEN_SOT]])
    decoded_tokens = [TOKEN_SOT]
    sample_len = mean_decode_len  # mean # of tokens to sample
    k_cache_self = np.zeros(
        (
            num_decoder_blocks,
            num_decoder_heads,
            attention_dim // num_decoder_heads,
            sample_len,
        )
    ).astype(np.float32)
    v_cache_self = np.zeros(
        (
            num_decoder_blocks,
            num_decoder_heads,
            sample_len,
            attention_dim // num_decoder_heads,
        )
    ).astype(np.float32)

    sum_logprobs = 0
    for i in range(sample_len):
        x = x.astype(np.int32)

        # Using i to index inside the decoder model hurts the
        # the model performance.
        # index - used to get positional embedding correctly.
        index = np.array([[0]], dtype=np.int32)
        index[0, 0] = i

        
        logits, k_cache_self, v_cache_self = run_whisper_decoder(decoder_model, x, index, k_cache_cross, v_cache_cross, k_cache_self, v_cache_self,args.invoke_nums)

        # logit has shape (51864,)
        logits = logits[0, -1]  # consider only the last token

        # Filters
        # SuppressBlank
        if i == 0:
            # pdb.set_trace()
            logits[[TOKEN_EOT, TOKEN_BLANK]] = -np.inf
        # SuppressTokens
        logits[NON_SPEECH_TOKENS] = -np.inf

        logits, logprobs = apply_timestamp_rules(logits, decoded_tokens)
        assert isinstance(logprobs, np.ndarray)

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
        x = np.array([[next_token]])
        decoded_tokens.append(int(next_token))

    print("decoder ok")
    tokenizer = whisper.decoding.get_tokenizer(
        multilingual=False, language="en", task="transcribe"
    )

    text = tokenizer.decode(decoded_tokens[1:])  # remove TOKEN_SOT
    text = text.strip()

    print("\n")
    print(f"<{args.audio}> transcription result:")
    print(text)
    print("\n")
    print("="*60)
    pass

def parser_args():
    audio_path=r"D:\algorithm\robot_det_face\whisper_learn\whisperkittools\data\common_voice_en_113714.wav"
    parser = argparse.ArgumentParser(description="Run model benchmarks")
    parser.add_argument('--audio',type=str,default=audio_path,help="Predict images path")
    parser.add_argument('--invoke_nums',type=int,default=1,help="Inference nums")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()
    aidlite_run(args)
