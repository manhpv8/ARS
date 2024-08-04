import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torchaudio
from config import PRETRAINED_MODEL, PRETRAINED_PROCESSOR, MODEL_PATH, SAMPLE_RATE


model = Wav2Vec2ForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = Wav2Vec2Processor.from_pretrained(PRETRAINED_PROCESSOR)

model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)


def wav2input(file_path):
  waveform, sample_rate = torchaudio.load(file_path)
  if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
  if sample_rate != SAMPLE_RATE:
      waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)(waveform)
  current_length = waveform.size(1)
  padding = 40000 - current_length
  waveform = torch.nn.functional.pad(waveform, (0, padding))
  input_values = processor(waveform.squeeze(0).tolist(), return_tensors="pt", padding="longest", sampling_rate=SAMPLE_RATE).input_values
  return input_values


def wave2input(waveform, sample_rate):
  if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
  if sample_rate != SAMPLE_RATE:
      waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)(waveform)
  current_length = waveform.size(1)
  padding = 40000 - current_length
  waveform = torch.nn.functional.pad(waveform, (0, padding))
  input_values = processor(waveform.squeeze(0).tolist(), return_tensors="pt", padding="longest", sampling_rate=SAMPLE_RATE).input_values
  return input_values

def predict(inputs):
    logits =  model(inputs.to(device)).logits
    _, predicted = torch.max(logits, 1)
    if predicted == 1:
        return "Bật camera lên"
    elif predicted == 2:
        return "Đóng cửa lại"
    else:
        return "Không tồn tại mệnh lệnh"
    
if __name__ == '__main__':
    predict()