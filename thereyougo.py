import requests
import random
import os
import shutil
import re
import subprocess
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import whisper
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from fastapi import Form
import uvicorn
from gradio_client import Client
import google.generativeai as genai
import json

load_dotenv()

app = FastAPI()

# Configure Gemini
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# Pydantic Models
class PromptList(BaseModel):
    prompts: List[str]

class VideoRequest(BaseModel):
    title: str

class FullVideoRequest(BaseModel):
    title: str
    script: str
    generate_image_prompts: List[str]
    tts_model: Optional[str] = "elevenlabs"

class GeneratePromptsRequest(BaseModel):
    script: str

class GenerateScriptRequest(BaseModel):
    title: str

class GenerateTitleRequest(BaseModel):
    topic: str

# Global state
current_prompts = []

# Image Generation Settings
width = 1080
height = 1920
seed = 40
enhance = "true"
nologo = 'true'
model = 'flux'

# Helper Functions
def download_image(image_url, filename):
    response = requests.get(image_url)
    if response.content and response.content.startswith(b'\xff\xd8\xff'):
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f'Downloaded {filename}')
        return True
    else:
        print(f"Corrupt image received for {filename}. Retrying...")
        return False

def generate_image(prompt, filename, initial_seed):
    current_seed = initial_seed
    while True:
        image_url = f"https://pollinations.ai/p/{prompt}, the classical era (Ancient Greece and Rome), The overall mood is one of awe, prosperity, and a romanticized view of ancient civilization, style of a grand Neoclassical painting/illustration?width={width}&height={height}&seed={current_seed}&model={model}&enhance={enhance}&nologo={nologo}"
        if download_image(image_url, filename):
            break
        else:
            current_seed = random.randint(0, 100000)
            print(f"Retrying with new seed: {current_seed}")

# Video Creation Functions
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, word_timestamps=True)
    return result["segments"]

def create_srt_from_segments(segments, srt_filename):
    os.makedirs(os.path.dirname(srt_filename), exist_ok=True)
    srt_content = ""
    for index, segment in enumerate(segments, start=1):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"].strip()
        start_srt = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{int(start_time % 60):02d},{int((start_time % 1) * 1000):03d}"
        end_srt = f"{int(end_time // 3600):02d}:{int((end_time % 3600) // 60):02d}:{int(end_time % 60):02d},{int((end_time % 1) * 1000):03d}"
        srt_content += f"{index}\n{start_srt} --> {end_srt}\n{text}\n\n"
    with open(srt_filename, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return srt_filename

def create_video_with_images_and_audio(images, audio_file, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    audio = AudioFileClip(audio_file)
    if not images:
        raise ValueError("Cannot create video with no images.")
    duration_per_image = audio.duration / len(images)
    clips = [ImageClip(str(img)).set_duration(duration_per_image) for img in images]
    video = concatenate_videoclips(clips, method="compose")
    if video.duration < audio.duration:
        extra_duration = audio.duration - video.duration
        clips[-1] = clips[-1].set_duration(clips[-1].duration + extra_duration)
        video = concatenate_videoclips(clips, method="compose")
    video = video.set_audio(audio)
    video.write_videofile(output_file, fps=24, codec="libx264")
    return output_file

def burn_subtitles_ffmpeg(input_video, srt_file, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    srt_file_escaped = str(Path(srt_file)).replace('\\', '/')
    cmd = ["ffmpeg", "-i", input_video, "-vf", f"subtitles='{srt_file_escaped}':force_style='Fontsize=10'", "-c:a", "copy", output_file]
    subprocess.run(cmd, check=True)
    return output_file

def generate_video_flow(audio_path, image_folder, title):
    title = title.lower().replace(" ", "_")
    title = re.sub(r'[^\w-]', '', title)
    subtitle_path = f"./temp/subtitles/{title}.srt"
    video_path = f"./temp/videos/temp_video_{title}.mp4"
    final_video_path = f"./output/{title}.mp4"
    images = sorted(Path(image_folder).glob('*.jpg'), key=lambda p: int(re.search(r'\d+', p.stem).group()))
    if not images:
        raise FileNotFoundError(f"No images found in {image_folder}")
    segments = transcribe_audio(audio_path)
    srt_file = create_srt_from_segments(segments, subtitle_path)
    video_file = create_video_with_images_and_audio(images, audio_path, video_path)
    final_output = burn_subtitles_ffmpeg(video_file, srt_file, final_video_path)
    os.remove(video_path)
    return final_output

# API Endpoints
@app.get("/")
def read_root():
    return FileResponse('index.html')

@app.post("/generate-title")
def generate_title_endpoint(request: GenerateTitleRequest):
    if not gemini_api_key:
        return {"error": "GEMINI_API_KEY not found in .env file"}
    try:
        model = genai.GenerativeModel('gemini-2.5-flash', 
                                      system_instruction=f"""
                                      
                                     You are a click-through-rate optimizer.
Your entire job is to pick best title {request.topic} that is most likely to be clicked by a broad,audience.
Decision stack (highest → lowest weight)

    Curiosity Gap – leaves the reader with a burning question.
    Emotional Trigger – sparks urgency, hope, or FOMO.
    Concrete Benefit – promises a clear payoff the reader already desires.
    Brevity & Clarity – ≤ 55 characters, instantly readable.
    Power Words – uses proven magnets (“now”, “easy”, “secret”, etc.).
    The title written in simple english.
 
                                      
                                      """)
        prompt = f"Return ONLY the exact text of the chosen title; no explanation, no bullet, no extra characters. Topic: '{request.topic}'"
        response = model.generate_content(prompt)
        # Clean up the response, removing quotes if the model adds them
        title = response.text.strip().strip('"')
        return {"title": title}
    except Exception as e:
        return {"error": f"Failed to generate title with Gemini: {str(e)}"}

@app.post("/generate-script")
def generate_script_endpoint(request: GenerateScriptRequest):
    if not gemini_api_key:
        return {"error": "GEMINI_API_KEY not found in .env file"}
    try:
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=f"""
            You are a short-form video script writer.
Goal
Deliver a spoken-word script in three parts: Hook, Body, Conclusion.
Constraints

    Write only plain sentences someone could read aloud.
    No scene notes, camera cues, or markdown.
    Keep words simple/easy and common.
    Do not add titles, timestamps, or sound effects.
            """
        )
        prompt = f"Create me a script base on this topic: {request.title} Provide the output as a single-line string, with no newline characters or \n escape sequences."
        response = model.generate_content(prompt)
        return {"script": response.text}
    except Exception as e:
        return {"error": f"Failed to generate script with Gemini: {str(e)}"}

@app.post("/generate-image-prompts")
def generate_image_prompts_endpoint(request: GeneratePromptsRequest):
    if not gemini_api_key:
        return {"error": "GEMINI_API_KEY not found in .env file"}
    try:
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=f"""
You are an image generation prompt creator.

Your task is to generate a list of distinct image prompts, strictly in an array format (e.g., ["prompt1", "prompt2", "etc..."]), without any JSON wrapper or code block formatting.

For each line or distinct segment of the provided script, create a corresponding image prompt. Ensure each prompt adheres to the following:

1. Features a person prominently (focus on "you" in the second person, meaning the visual subject is always a person).
2. Is easy to understand and literal, avoiding abstract or overly complex visuals. Simplicity is key.
3. Concentrates on the subject's body language and expression to convey meaning and emotion, in a **Classical Era style**, characterized by idealized forms, dramatic poses, and noble grandeur. Emotions should be clearly discernible and not neutral.
4. Includes minimal background objects, only when essential for contextual clarity (e.g., a simple wall, a single book).
5. Does NOT include any text on the image.
6. Directly complements the script content for that specific line/segment.
7. Is concise yet descriptive enough for effective image generation.

Segmentation Rule:
Break the script into **beats** (≈ one idea or line).
Treat each “\n\n” or a period followed by a capital letter as a new beat.

The very first prompt generated must be highly engaging and suitable as a visual “hook” for the script, with strong emotional focus.

Adjust the number of prompts to align with the logical flow and pacing implied by the script’s duration or segmentation.

For rhythm and pacing: Vary the beat lengths in an **alternating or random pattern** rather than keeping it perfectly regular, to create a more dynamic visual flow.

            """
        )
        prompt = f"""
        You are an image generation prompt creator.

Your task is to generate a list of distinct image prompts, strictly in an array format (e.g., ["prompt1", "prompt2", "etc..."]), without any JSON wrapper or code block formatting.

For each line or distinct segment of the provided script, create a corresponding image prompt. Ensure each prompt adheres to the following:

1.  Features a person prominently (focus on "you" in the second person, meaning the visual subject is always a person).
1.5. The you is a MAN.
2.  Is easy to understand and literal, avoiding abstract or overly complex visuals. Simplicity is key.
3.  Concentrates on the subject's body language and expression to convey meaning and emotion, in a **Classical Era style**, characterized by idealized forms, dramatic poses, and noble grandeur. Emotions should be clearly discernible and not neutral.
4.  Includes minimal background objects, only when essential for contextual clarity (e.g., a simple wall, a single book).
5.  Does NOT include any text on the image.
6.  Directly complements the script content for that specific line/segment.
7.  Is concise yet descriptive enough for effective image generation.

The very first prompt generated must be highly engaging and suitable as a visual 'hook' for the script, and emotional focus.
Adjust the number of prompts to align with the logical flow and pacing implied by the script's duration or segmentation. 


HERES THE SCRIPT:
{request.script}
        """
        response = model.generate_content(prompt)
        # Clean up the response to extract the JSON array
        json_response = response.text.strip().replace("```json", "").replace("```", "")
        prompts = json.loads(json_response)
        return {"prompts": prompts}
    except Exception as e:
        return {"error": f"Failed to generate prompts with Gemini: {str(e)}"}

@app.post("/generate")
def generate_images_from_prompts(prompt_list: PromptList):
    global current_prompts
    current_prompts = prompt_list.prompts
    image_dir = 'images'
    os.makedirs(image_dir, exist_ok=True)
    if os.path.exists(image_dir):
        for f in os.listdir(image_dir):
            os.remove(os.path.join(image_dir, f))
    generated_files = []
    for i, prompt in enumerate(current_prompts):
        filename = f"{image_dir}/image_{i}.jpg"
        generate_image(prompt, filename, seed)
        generated_files.append(filename)
    return {"message": "Images generated successfully.", "files": generated_files}

@app.get("/images")
def get_images():
    image_dir = 'images'
    if not os.path.exists(image_dir):
        return {"images": []}
    images = [f"images/{f}" for f in os.listdir(image_dir) if f.startswith('image_') and f.endswith('.jpg')]
    return {"images": images}

@app.get("/images/{image_id}")
def get_image(image_id: int):
    filename = f"images/image_{image_id}.jpg"
    if os.path.exists(filename):
        return FileResponse(filename)
    return {"error": "Image not found"}

@app.post("/regenerate/{image_id}")
def regenerate_image(image_id: int):
    if 0 <= image_id < len(current_prompts):
        prompt = current_prompts[image_id]
        filename = f"images/image_{image_id}.jpg"
        new_seed = random.randint(0, 100000)
        print(f"Regenerating image {image_id} with new seed: {new_seed}")
        generate_image(prompt, filename, new_seed)
        return {"message": f"Image {image_id} regenerated successfully."}
    return {"error": "Invalid image ID"}

@app.post("/text-to-speech")
def text_to_speech(text: str = Form(...)):
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return {"error": "ELEVENLABS_API_KEY not found in .env file"}
    client = ElevenLabs(api_key=api_key)
    audio_stream = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    output_filename = "audio/tts_output.mp3"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)
    return FileResponse(output_filename, media_type="audio/mpeg")

@app.post("/create-video")
def create_video_endpoint(request: VideoRequest):
    audio_file = 'audio/tts_output.mp3'
    image_dir = 'images'
    os.makedirs('temp/images', exist_ok=True)
    os.makedirs('temp/subtitles', exist_ok=True)
    os.makedirs('temp/videos', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    temp_audio_path = 'temp/audio.mp3'
    temp_image_folder = 'temp/images'
    if os.path.exists(temp_image_folder):
        for f in os.listdir(temp_image_folder):
            os.remove(os.path.join(temp_image_folder, f))
    if os.path.exists(image_dir):
        for img_file in os.listdir(image_dir):
            shutil.copy(os.path.join(image_dir, img_file), temp_image_folder)
    if os.path.exists(audio_file):
        shutil.copy(audio_file, temp_audio_path)
    else:
        return {"error": "Audio file not found. Please generate it first using /text-to-speech."}
    final_video_path = generate_video_flow(temp_audio_path, temp_image_folder, request.title)
    return FileResponse(final_video_path, media_type="video/mp4")

def generate_audio(script: str, tts_model: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if tts_model == "Qwen/Qwen-TTS-Demo":
        client = Client("Qwen/Qwen-TTS-Demo")
        temp_audio_path = client.predict(text=script, voice="Dylan", api_name="/predict")
        shutil.copy(temp_audio_path, output_path)
        print(f"Audio generated with {tts_model} and saved to {output_path}")
    elif tts_model == "NihalGazi/Text-To-Speech-Unlimited":
        client = Client("NihalGazi/Text-To-Speech-Unlimited")
        result = client.predict(prompt=script, voice="alloy", emotion=script, use_random_seed=True, specific_seed=12345, api_name="/text_to_speech_app")
        if isinstance(result, tuple):
            temp_audio_path = result[0]
        else:
            temp_audio_path = result
        shutil.copy(temp_audio_path, output_path)
        print(f"Audio generated with {tts_model} and saved to {output_path}")
    else:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not found for elevenlabs model")
        client = ElevenLabs(api_key=api_key)
        audio_stream = client.text_to_speech.convert(text=script, voice_id="JBFqnCBsd6RMkjVDRZzb", model_id="eleven_multilingual_v2", output_format="mp3_44100_128")
        with open(output_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        print(f"Audio generated with ElevenLabs and saved to {output_path}")
    return output_path

@app.post("/generate-full-video")
def generate_full_video_endpoint(request: FullVideoRequest):
    audio_output_path = "audio/generated_audio.mp3"
    try:
        generate_audio(request.script, request.tts_model, audio_output_path)
    except Exception as e:
        return {"error": f"Failed to generate audio: {str(e)}"}
    image_dir = 'images'
    os.makedirs(image_dir, exist_ok=True)
    for f in os.listdir(image_dir):
        os.remove(os.path.join(image_dir, f))
    for i, prompt in enumerate(request.generate_image_prompts):
        filename = f"{image_dir}/image_{i}.jpg"
        generate_image(prompt, filename, seed)
    final_video_path = generate_video_flow(audio_output_path, image_dir, request.title)
    return FileResponse(final_video_path, media_type="video/mp4")

if __name__ == "__main__":
    uvicorn.run("thereyougo:app", host="0.0.0.0", port=8000, reload=True)
