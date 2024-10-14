import os
import io
import re
import gradio as gr
from pathlib import Path
from tempfile import NamedTemporaryFile
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict

class DialogueItem(BaseModel):
    text: str
    speaker: str

class Dialogue(BaseModel):
    dialogue: List[DialogueItem]

def read_text_file(file_path, speaker_names):
    dialogue = []
    current_speaker = None
    current_text = ""

    with open(file_path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            if line.startswith('\ufeff'):
                line = line[1:]
            line = line.strip()
            if not line:
                continue
            print(line)
            speaker_match = False
            for speaker in speaker_names:
                if line.startswith(f"{speaker}:"):
                    if current_speaker and current_text:
                        dialogue.append(DialogueItem(text=current_text.strip(), speaker=current_speaker))
                    current_speaker = speaker
                    current_text = line[len(speaker)+1:].strip()
                    speaker_match = True
                    break
            if not speaker_match:
                print("---"+line)
                if current_speaker:
                    current_text += " " + line
                else:
                    # If this is the first line and no speaker is matched, assume it's for the first speaker
                    current_speaker = speaker_names[0]
                    current_text = line

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

def get_next_file_number(directory):
    existing_files = [f for f in os.listdir(directory) if f.endswith('.mp3')]
    if not existing_files:
        return 1
    
    numbers = []
    for f in existing_files:
        match = re.search(r'(\d+)\.mp3$', f)
        if match:
            numbers.append(int(match.group(1)))
    
    return max(numbers) + 1 if numbers else 1

def generate_audio(
    file: gr.File,
    openai_api_key: str,
    audio_model: str,
    speaker_1_name: str,
    speaker_1_voice: str,
    speaker_2_name: str,
    speaker_2_voice: str,
    speaker_3_name: str,
    speaker_3_voice: str,
    speaker_4_name: str,
    speaker_4_voice: str,
    speaker_5_name: str,
    speaker_5_voice: str,
    speaker_6_name: str,
    speaker_6_voice: str,
) -> tuple:
    if not os.getenv("OPENAI_API_KEY") and not openai_api_key:
        raise gr.Error("OpenAI API key is required")

    file_path = Path(file.name)
    if file_path.suffix.lower() != '.txt':
        raise gr.Error("Please upload a .txt file")

    speaker_names = [speaker_1_name, speaker_2_name, speaker_3_name, speaker_4_name, speaker_5_name, speaker_6_name]
    speaker_voices = {
        speaker_1_name: speaker_1_voice,
        speaker_2_name: speaker_2_voice,
        speaker_3_name: speaker_3_voice,
        speaker_4_name: speaker_4_voice,
        speaker_5_name: speaker_5_voice,
        speaker_6_name: speaker_6_voice
    }

    dialogue = read_text_file(file_path, speaker_names)

    combined_audio = b""
    transcript = ""
    original_text = ""

    temporary_directory = "./gradio_cached_examples/tmp/"
    os.makedirs(temporary_directory, exist_ok=True)

    for i, line in enumerate(dialogue.dialogue, start=1):
        voice = speaker_voices[line.speaker]
        audio_chunk = get_mp3(line.text, voice, audio_model, openai_api_key)
        combined_audio += audio_chunk
        transcript += f"{line.speaker}: {line.text}\n\n"
        original_text += f"{line.speaker}: {line.text}\n"

        # Save individual audio chunks
        chunk_filename = f"audio_chunk_{i:03d}.mp3"
        chunk_path = os.path.join(temporary_directory, chunk_filename)
        with open(chunk_path, 'wb') as f:
            f.write(audio_chunk)

    # Save the combined audio file
    combined_filename = f"combined_audio_{get_next_file_number(temporary_directory):03d}.mp3"
    combined_path = os.path.join(temporary_directory, combined_filename)
    with open(combined_path, 'wb') as f:
        f.write(combined_audio)

    return combined_path, transcript, original_text

with gr.Blocks(title="Text to Audio") as demo:
    gr.Markdown("# Convert Text File into Audio")
    gr.Markdown("Upload a text file with dialogue in the format:\n[Speaker Name]: [dialogue text]")
    
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
            
            speakers = []
            for i in range(1, 7):
                with gr.Row():
                    name = gr.Textbox(
                        label=f"Speaker {i} Name",
                        value=f"Speaker {i}",
                    )
                    voice = gr.Dropdown(
                        label=f"Speaker {i} Voice",
                        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                        value="alloy",
                    )
                    speakers.extend([name, voice])
        
        with gr.Column(scale=3):
            audio_output = gr.Audio(label="Generated Audio", format="mp3")
            transcript_output = gr.Textbox(label="Transcript", lines=10)
            original_text_output = gr.Textbox(label="Original Text", lines=10)

    submit_btn = gr.Button("Generate Audio")
    
    submit_btn.click(
        fn=generate_audio,
        inputs=[file, openai_api_key, audio_model] + speakers,
        outputs=[audio_output, transcript_output, original_text_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=5558)
