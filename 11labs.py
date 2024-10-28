import os
import re
import gradio as gr
from io import BytesIO
from pathlib import Path
from typing import IO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv


casts = []
voices_dict = {}
voice_names = []
cast_to_voice = {}
models = [
    "eleven_multilingual_v2",
    "eleven_turbo_v2_5",
    "eleven_turbo_v2",
    "eleven_multilingual_v1",
    "eleven_monolingual_v1"
]
__transcript = []


load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY
)


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


def text_to_speech_stream(
        text: str,
        voice_id: str,
        model_id: str,
        output: str,
        voice_settings: VoiceSettings
    ) -> IO[bytes]:
    response = client.text_to_speech.convert(
        voice_id=voice_id,
        output_format="mp3_44100_128",
        text=text,
        model_id=model_id,
        voice_settings=voice_settings
    )

    audio_stream = BytesIO()

    for chunk in response:
        if chunk:
            audio_stream.write(chunk)

    audio_stream.seek(0)

    with open(output, 'wb') as f:
        f.write(audio_stream.getvalue())

    return audio_stream


def get_voices():
    response = client.voices.get_all()
    for voice in response.voices:
        voices_dict[voice.name] = voice.voice_id
        voice_names.append(voice.name)


def make_casts(file: gr.File):
    casts.clear()
    __transcript.clear()
    file_path = Path(file.name)
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            if line.startswith("\n") or line.startswith("#"):
                continue
            le = line.split(':')
            le[1] = le[1].lstrip()
            __transcript.append((le[0], le[1]))
            if le[0] in casts:
                continue
            casts.append(le[0])

    return gr.Textbox(value=";".join(casts))


def generate(
        file: gr.File,
        model: str,
        stability:float,
        similarity:float,
        exaggeration:float,
        boost:bool,
        *voices
    ):
    assert(__transcript)
    print("====== start generation ======")
    print("settings")
    print(f" - model={model}\n - stability={stability}\n - similarity={similarity}\n - exaggeration={exaggeration}\n - boost={boost}")
    print()
    for i, v in enumerate(voices):
        cast_to_voice[casts[i]] = voices[i]

    assert(len(cast_to_voice) != 0)

    combined_audio = b""
    voice_settings = VoiceSettings(
        stability=stability,
        similarity_boost=similarity,
        style=exaggeration,
        use_speaker_boost=boost
    )
    dir_name = Path(file.name)

    output_directory = f"./result/mp3/{dir_name.stem}/"
    os.makedirs(output_directory, exist_ok=True)

    output_transcript = []
    for i, line in enumerate(__transcript):
        voice_id = voices_dict[cast_to_voice[line[0]]]
        text = line[1]
        output_transcript.append(f"{line[0]}: {line[1]}")
        print(output_transcript[i])
        chunk_name = f"audio_chunk_{i:03d}.mp3"
        chunk_path = os.path.join(output_directory, chunk_name)
        stream = text_to_speech_stream(text, voice_id, model, chunk_path, voice_settings)
        combined_audio += stream.getvalue()

    filename = f"result_audio_{get_next_file_number(output_directory):03d}.mp3"
    file_path = os.path.join(output_directory, filename)
    with open(file_path, 'wb') as f:
        f.write(combined_audio)

    print("==============================")
    return gr.File(value=None), gr.Textbox(value=""), file_path, "\n".join(output_transcript)


with gr.Blocks(title="Text to Audio") as demo:
    gr.Markdown(
    """
    # Convert Text file into mp3
    > **Uploading a text file of which format should be: [Speaker name]: [dialogue text]**
    ---
    """
    )

    with gr.Row():
        with gr.Column(scale=2):
            file = gr.File(
                label="Input Text File",
                file_types=[".txt"]
            )
            casts_text = gr.Textbox(visible=False)
            file.upload(fn=make_casts, inputs=[file], outputs=[casts_text])

            @gr.render(inputs=casts_text)
            def show_select_voice(text):
                if len(text) == 0:
                    return

                voices = []
                cs = text.split(";")
                with gr.Row():
                    for c in cs:
                        voice = gr.Dropdown(label=c, value=voice_names[0], choices=voice_names, interactive=True)
                        voices.append(voice)

                    with gr.Group():
                        gr.Label(label="Settings")
                        model = gr.Dropdown(label="Model", value=models[0], choices=models, interactive=True)
                        stability = gr.Slider(0, 1, value=0.5, step=0.01, interactive=True, label="Stability")
                        similarity = gr.Slider(0, 1, value=0.5, step=0.01, interactive=True, label="Similarity")
                        exaggeration = gr.Slider(0, 1, value=0.5, step=0.01, interactive=True, label="Style Exaggeration")
                        boost = gr.Checkbox(label="Speaker boost")

                submit_btn.click(
                    fn=generate,
                    inputs=[file, model, stability, similarity, exaggeration, boost] + voices,
                    outputs=[file, casts_text, audio, transcript]
                )

        with gr.Column(scale=3):
            with gr.Group():
                audio = gr.Audio(label="Generated Audio", type="filepath", format="mp3", interactive=False, autoplay=False)
                transcript = gr.Textbox(label="Transcript")

    submit_btn = gr.Button("Generate Audio")


if __name__ == "__main__":
    get_voices()
    demo.launch(server_name="localhost", server_port=5558)
