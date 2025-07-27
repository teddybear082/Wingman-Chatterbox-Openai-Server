import argparse
import gc
import io
import random
import numpy as np
import pysbd
import torch
from flask import Flask, request, send_file, jsonify, render_template, render_template_string, Response
from chatterbox.tts import ChatterboxTTS
import torchaudio

# Set up Flask app
app = Flask(__name__)

parser = argparse.ArgumentParser(description="OpenAI-compatible TTS server for Chatterbox.")

# Server arguments
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the server.")
parser.add_argument("--port", type=int, default=5002, help="Port for the server.")
parser.add_argument("--model-name", type=str, default="chatterbox-tts", help="Name of the model to report in API responses.")
parser.add_argument("--debug", action="store_true", help="Run the server in debug mode.")
parser.add_argument("--device", type=str, default="cpu", help="Device to run server on. Options: cpu, cuda, cuda:0, cuda:1, mps")
parser.add_argument("--low_vram", action=argparse.BooleanOptionalAction, default=False, help="Whether to unload model to cpu when not generating.")
parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False, help="Enable audio streaming sentence by sentence.")

# Chatterbox generation arguments with reasonable defaults
parser.add_argument("--exaggeration", type=float, default=0.5, help="Exaggeration level (0.5 is neutral).")
parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
parser.add_argument("--min-p", type=float, default=0.05, help="min_p for nucleus sampling.")
parser.add_argument("--top-p", type=float, default=1.0, help="top_p for nucleus sampling (1.0 disables it).")
parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Repetition penalty.")

args = parser.parse_args()
segmenter = pysbd.Segmenter(language="en", clean=True)

DEVICE = args.device
if "cuda" in DEVICE:
    if not torch.cuda.is_available():
        DEVICE = "cpu"
if "mps" in DEVICE:
    if not torch.backends.mps.is_available():
        DEVICE = "cpu"

# Load the Chatterbox model
print(f"Loading Chatterbox TTS...")
chatterbox_model = ChatterboxTTS.from_pretrained(DEVICE)
print(f"Model loaded successfully on {DEVICE}.")
CURRENT_DEVICE = DEVICE

generation_count = 0

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def handle_vram_change(desired_device: str):
    global chatterbox_model, CURRENT_DEVICE
    if torch.cuda.is_available():
        if "cuda" in desired_device:
            if "cuda" not in CURRENT_DEVICE:
                if chatterbox_model:
                    del chatterbox_model
                gc.collect()
                chatterbox_model = ChatterboxTTS.from_pretrained(desired_device)
                CURRENT_DEVICE = desired_device
                print(f"Switched ChatterboxTTS model to {desired_device}.")
        elif "cpu" in desired_device:
            if "cpu" not in CURRENT_DEVICE:
                del chatterbox_model
                torch.cuda.empty_cache()
                gc.collect()
                chatterbox_model = None
                CURRENT_DEVICE = desired_device
                print("Unloaded ChatterboxTTS model")

if args.low_vram and "cuda" in DEVICE:
    handle_vram_change("cpu")

def _cleanup():
    """Handles VRAM and garbage collection after a generation task."""
    if args.low_vram and "cuda" in DEVICE:
        handle_vram_change("cpu")

    global generation_count
    generation_count += 1
    if generation_count >= 5 and not args.low_vram and "cuda" in DEVICE:
        generation_count = 0
        torch.cuda.empty_cache()
        gc.collect()
        print("CUDA cache cleared after 5 generations.")

def split_sentences(input_text: str) -> list:
    if len(input_text) <= 200:
        return [input_text]
    return segmenter.segment(input_text)

@app.route("/")
def index():
    return render_template(
        "index.html",
    )

@app.route("/v1/audio/speech", methods=["POST"])
def openai_tts():
    if args.low_vram and "cuda" in DEVICE:
        handle_vram_change(DEVICE)

    payload = request.get_json(force=True)
    
    text = payload.get("input", "")
    audio_prompt_path = payload.get("voice", None)
    cfg_weight = payload.get("speed", 0.5)
    stream = payload.get("stream", args.stream)

    if not text:
        return jsonify({"error":"Missing input in request"}), 400

    # --- Streaming Logic ---
    if stream:
        fmt = payload.get("response_format", "pcm").lower()
        
        # Only PCM and MP3 are supported for chunked streaming
        if fmt not in ['pcm', 'mp3']:
            return jsonify({"error": f"Streaming is only supported for 'pcm' and 'mp3' formats. Requested: {fmt}"}), 400
        
        print(f"Streaming Request: text='{text}', voice='{audio_prompt_path}', speed='{cfg_weight}', format='{fmt}'")

        def generate_stream():
            try:
                sentences = split_sentences(text)
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    print(f"Streaming sentence: {sentence}")
                    wav_tensor = chatterbox_model.generate(
                        sentence,
                        audio_prompt_path=audio_prompt_path,
                        exaggeration=args.exaggeration,
                        temperature=args.temperature,
                        cfg_weight=cfg_weight,
                        min_p=args.min_p,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                    )
                    waveform_cpu = wav_tensor.squeeze(0).cpu()

                    if fmt == 'pcm':
                        waveform_int16 = (waveform_cpu * 32767).to(torch.int16)
                        yield waveform_int16.numpy().tobytes()
                    elif fmt == 'mp3':
                        buffer = io.BytesIO()
                        waveform_2d = waveform_cpu.unsqueeze(0)
                        torchaudio.save(buffer, waveform_2d, chatterbox_model.sr, format="mp3")
                        yield buffer.getvalue()
            finally:
                _cleanup()
                print("Finished streaming response.")

        mimetype = "audio/L16" if fmt == 'pcm' else "audio/mpeg"
        return Response(generate_stream(), mimetype=mimetype)

    # --- Non-Streaming Logic ---
    else:
        fmt = payload.get("response_format", "mp3").lower()
        print(f"Request: text='{text}', voice='{audio_prompt_path}', speed='{cfg_weight}', format='{fmt}'")

        sentences = split_sentences(text)
        audio_chunks = []
        for sentence in sentences:
            wav_tensor = chatterbox_model.generate(
                sentence,
                audio_prompt_path=audio_prompt_path,
                exaggeration=args.exaggeration,
                temperature=args.temperature,
                cfg_weight=cfg_weight,
                min_p=args.min_p,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            audio_chunks.append(wav_tensor)
        
        final_audio = torch.cat(audio_chunks, dim=-1) if len(audio_chunks) > 1 else audio_chunks[0]
        waveform_cpu = final_audio.squeeze(0).cpu()
        
        mimetypes = {
            "wav": "audio/wav", "mp3": "audio/mpeg", "opus": "audio/ogg",
            "aac": "audio/aac", "flac": "audio/flac", "pcm": "audio/L16",
        }

        if fmt not in mimetypes:
            return jsonify({"error": f"Unsupported format: {fmt}"}), 400

        mimetype = mimetypes[fmt]
        buffer = io.BytesIO()

        if fmt == 'pcm':
            waveform_int16 = (waveform_cpu * 32767).to(torch.int16)
            buffer.write(waveform_int16.numpy().tobytes())
        else:
            waveform_2d = waveform_cpu.unsqueeze(0)
            format_args = {"format": fmt}
            if fmt == 'opus':
                format_args = {"format": "ogg", "encoding": "opus"}
            elif fmt == 'aac':
                format_args = {"format": "mp4", "encoding": "aac"}
            torchaudio.save(buffer, waveform_2d, chatterbox_model.sr, **format_args)

        buffer.seek(0)
        _cleanup()
        return send_file(buffer, mimetype=mimetype)

@app.route("/v1/models", methods=["GET"])
def openai_list_models():
    """
    Return a list of available models, for OpenAI client compatibility.
    """
    return jsonify({
        "data": [
            {
                "id": args.model_name,
                "object": "model",
                "created": 1677610600,  # Placeholder timestamp
                "owned_by": "user"
            }
        ]
    })

def main():
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()