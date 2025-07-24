# There You Go: AI Video Maker

This app helps you create short videos using Artificial Intelligence (AI). You can give it a topic, and it will write a script, create images, and turn it all into a video with a voiceover and captions.

## What You Need (Requirements)

Before you start, make sure you have these things:

1.  **Python:** You need Python on your computer. You can get it from [python.org](https://www.python.org/).
2.  **API Keys:** This app uses other services to work. You will need to sign up for them and get special "keys" (like passwords).
    *   **Gemini (Google AI):** To write the script and title.
    *   **ElevenLabs:** To create the voice from the script.
3.  **FFmpeg:** A tool for creating videos. You can download it from [ffmpeg.org](https://ffmpeg.org/). Make sure it's installed correctly so the app can use it.

## How to Use the App

Follow these steps to get the app running on your computer.

### Step 1: Get the Code

First, you need to download the app's files.

```bash
git clone https://github.com/your-username/there-you-go.git
cd there-you-go
```
This command copies all the files into a folder called `there-you-go` on your computer.

### Step 2: Install the Tools

Next, the app needs to install some tools it uses. Run this command in your terminal:

```bash
pip install -r requirements.txt
```
This will read the `requirements.txt` file and install everything listed there.

### Step 3: Add Your API Keys

You need to tell the app what your secret API keys are.

1.  Find the file named `.env.example` and rename it to `.env`.
2.  Open the `.env` file with a text editor.
3.  Copy and paste your keys inside the quotes:

    ```
    GEMINI_API_KEY="paste-your-gemini-key-here"
    ELEVENLABS_API_KEY="paste-your-elevenlabs-key-here"
    ```

### Step 4: Start the App

Now you are ready to run the app! Use this command:

```bash
py thereyougo.app
#OR
uvicorn thereyougo:app --reload
```

This starts a local web server. You will see a message in your terminal that looks like this: `Uvicorn running on http://127.0.0.1:8000`.

### Step 5: Open the App in Your Browser

Open your web browser (like Chrome, Firefox, or Safari) and go to this address:

**http://127.0.0.1:8000**

You should now see the app's webpage, and you can start creating videos!

