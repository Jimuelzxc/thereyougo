<div align="center">
  <img src="assets/logo.png" alt="There You Go Logo" width="150"/>
  <h1>There You Go: AI Video Generator</h1>
  <p>
This project helps you create short videos using Artificial Intelligence. Give it a topic, and it will automatically generate a title, write a script, create matching images, and combine them into a final video complete with a voiceover and subtitles.

  </p>
</div>

---

<!-- Placeholder for a GIF of the app in action -->
<!-- <div align="center">
  <img src="link-to-your-demo.gif" alt="App Demo GIF"/>
</div> -->

## ✨ Features

- **🤖 AI-Powered Content:** Uses Google's Gemini AI to generate titles, scripts, and image prompts.
- **🖼️ Image Generation:** Creates images from prompts using the Pollinations.ai API.
- **🗣️ Text-to-Speech:** Converts scripts to audio with a choice of high-quality voices from ElevenLabs and others.
- **🎬 Automated Video Assembly:** Uses MoviePy and FFmpeg to combine images and audio into a video.
- **✍️ Automatic Subtitles:** Transcribes the audio with Whisper and burns subtitles directly into the video.
- **🌐 Simple Web Interface:** An easy-to-use frontend to manage the entire video creation process.
- **🔌 API-First Design:** Expose endpoints for all major functions, allowing for automation with tools like n8n or Zapier.

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **AI & ML:**
  - **Language Model:** Google Gemini
  - **Image Generation:** Pollinations.ai
  - **Text-to-Speech:** ElevenLabs, Gradio Client
  - **Transcription:** OpenAI Whisper
- **Video Processing:** MoviePy, FFmpeg

## 🚀 Getting Started

Follow these steps to get the app running on your local machine.

### 1. Prerequisites

Make sure you have the following installed:

- **Python 3.8+**: You can download it from [python.org](https://www.python.org/).
- **FFmpeg**: A required tool for video processing. Download it from [ffmpeg.org](https://ffmpeg.org/) and ensure it's accessible in your system's PATH.
- **Git**: For cloning the repository.

### 2. Installation & Setup

```bash
# 1. Clone the repository to your local machine
git clone https://github.com/your-username/there-you-go.git
cd there-you-go

# 2. Install the required Python packages
pip install -r requirements.txt

# 3. Set up your environment variables
#    Rename the example file to .env
cp .env.example .env

# 4. Add your secret API keys to the new .env file
#    GEMINI_API_KEY="your-gemini-key"
#    ELEVENLABS_API_KEY="your-elevenlabs-key"
```

### 3. Running the Application

Once the setup is complete, start the web server:

```bash
py thereyougo.py
#or
uvicorn thereyougo:app --reload

```

Open your web browser and navigate to **`http://127.0.0.1:8000`**. You should now see the app's interface and can start creating videos!

## 🤝 Contributing

Contributions are welcome! If you have ideas for new features or find a bug, please open an issue or submit a pull request.
