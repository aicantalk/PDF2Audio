import os
import io
import gradio as gr
from pathlib import Path
from tempfile import NamedTemporaryFile
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Literal

class DialogueItem(BaseModel):
    text: str
    speaker: Literal["speaker-1", "speaker-2"]

class Dialogue(BaseModel):
    dialogue: List[DialogueItem]

def read_text_file(file_path):
    dialogue = []
    current_speaker = None
    current_text = ""

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("speaker-1:") or line.startswith("speaker-2:"):
                if current_speaker and current_text:
                    dialogue.append(DialogueItem(text=current_text.strip(), speaker=current_speaker))
                current_speaker = line[:9]
                current_text = line[9:].strip()
            else:
                current_text += " " + line

    if current_speaker and current_text:
        dialogue.append(DialogueItem(text=current_text.strip(), speaker=current_speaker))

    return Dialogue(dialogue=dialogue)

def get_mp3(text: str, voice: str, audio_model: str, api_key: str = None) -> bytes:
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    with client.audio.speech.with_streaming_response.create(
        model=audio_model,
        voice=voice,
        input=text,
    ) as response:
        with io.BytesIO() as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
            return file.getvalue()

def generate_audio(
    file: gr.File,
    openai_api_key: str,
    audio_model: str = "tts-1",
    speaker_1_voice: str = "alloy",
    speaker_2_voice: str = "echo",
) -> tuple:
    if not os.getenv("OPENAI_API_KEY") and not openai_api_key:
        raise gr.Error("OpenAI API key is required")

    file_path = Path(file.name)
    if file_path.suffix.lower() != '.txt':
        raise gr.Error("Please upload a .txt file")

    dialogue = read_text_file(file_path)

    audio = b""
    transcript = ""
    original_text = ""

    for line in dialogue.dialogue:
        voice = speaker_1_voice if line.speaker == "speaker-1" else speaker_2_voice
        audio_chunk = get_mp3(line.text, voice, audio_model, openai_api_key)
        audio += audio_chunk
        transcript += f"{line.speaker}: {line.text}\n\n"
        original_text += f"{line.speaker}: {line.text}\n"

    temporary_directory = "./gradio_cached_examples/tmp/"
    os.makedirs(temporary_directory, exist_ok=True)

    temporary_file = NamedTemporaryFile(
        dir=temporary_directory,
        delete=False,
        suffix=".mp3",
    )
    temporary_file.write(audio)
    temporary_file.close()

    return temporary_file.name, transcript, original_text

with gr.Blocks(title="Text to Audio") as demo:
    gr.Markdown("# Convert Text File into Audio")
    gr.Markdown("Upload a text file with dialogue in the format:\nspeaker-1: [dialogue text]\nspeaker-2: [dialogue text]")
    
    with gr.Row():
        with gr.Column(scale=2):
            file = gr.File(
                label="Input Text File",
                file_types=[".txt"]
            )
            openai_api_key = gr.Textbox(
                label="OpenAI API Key",
                type="password"
            )
            audio_model = gr.Dropdown(
                label="Audio Generation Model",
                choices=["tts-1", "tts-1-hd"],
                value="tts-1",
            )
            speaker_1_voice = gr.Dropdown(
                label="Speaker 1 Voice",
                choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                value="alloy",
            )
            speaker_2_voice = gr.Dropdown(
                label="Speaker 2 Voice",
                choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                value="echo",
            )
        
        with gr.Column(scale=3):
            audio_output = gr.Audio(label="Generated Audio", format="mp3")
            transcript_output = gr.Textbox(label="Transcript", lines=10)
            original_text_output = gr.Textbox(label="Original Text", lines=10)

    submit_btn = gr.Button("Generate Audio")
    
    submit_btn.click(
        fn=generate_audio,
        inputs=[file, openai_api_key, audio_model, speaker_1_voice, speaker_2_voice],
        outputs=[audio_output, transcript_output, original_text_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=5558)
