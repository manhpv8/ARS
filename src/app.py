from fastapi import FastAPI, File, UploadFile
import torchaudio
import io
import uvicorn
from services.detech_command import predict, wave2input

app = FastAPI()

@app.post("/detech_command/")
async def upload_wav(file: UploadFile = File(...)):
    file_content = await file.read()
    audio_buffer = io.BytesIO(file_content)

    waveform, sample_rate = torchaudio.load(audio_buffer)
    inputs = wave2input(waveform, sample_rate)
    output = predict(inputs)
    return {
        "filename": file.filename,
        "output": output
    }
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)