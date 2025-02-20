"""
## Setup

To install the dependencies for this script, run:

```
pip install google-genai pyaudio
```

Before running this script, ensure the `GEMINI_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

## Description

This script creates an interactive session with Gemini AI model where:
- You can provide input through both text (typing) and voice (microphone)
- The model will respond in the specified modality (--response_modality):
  - AUDIO: Responses will be spoken through your speakers/headphones
  - TEXT: Responses will be displayed as text in the console

## Run

To run the script:

```
python Get_started_LiveAPI.py [--voice_name VOICE] [--system_prompt PROMPT] [--response_modality {AUDIO,TEXT}]
```

Arguments:
  --voice_name        Voice to use (Aoede, Charon, Fenrir, Kore, or Puck). Default: Aoede
  --system_prompt     Custom system prompt for the AI. Default: Helpful assistant
  --response_modality Response type (AUDIO or TEXT). Default: AUDIO
"""

import argparse
import asyncio
import os
import sys
import traceback

import pyaudio
from google import genai

if sys.version_info < (3, 11, 0):
    import exceptiongroup
    import taskgroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-exp"

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, config):
        self.config = config
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=self.config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voice_name",
        type=str,
        default="Aoede",
        help="Voice to use (Aoede, Charon, Fenrir, Kore, or Puck)",
        choices=["Aoede", "Charon", "Fenrir", "Kore", "Puck"],
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful assistant called Jarvis.",
        help="System prompt for the AI personality",
    )
    parser.add_argument(
        "--response_modality",
        type=str,
        default="AUDIO",
        help="Response type (AUDIO or TEXT)",
        choices=["AUDIO", "TEXT"],
    )
    args = parser.parse_args()

    # Create config with parsed arguments
    speech_config = {
        "voice_config": {"prebuilt_voice_config": {"voice_name": args.voice_name}}
    }
    config = {
        "generation_config": {"response_modalities": [args.response_modality]},
        "system_instruction": args.system_prompt,
        "speech_config": speech_config,
    }

    main = AudioLoop(config)
    asyncio.run(main.run())
