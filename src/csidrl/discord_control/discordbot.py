import asyncio
import os.path
import sys
from io import BytesIO
from typing import Optional

import discord
import numpy as np
from PIL import Image
from discord.ext import commands


class DiscordBot(commands.Bot):
    def __init__(self, token):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__("^", intents=intents)
        self.token = token

        self.logged = asyncio.Event()

    async def on_ready(self):
        #print(f"We have logged in as {self.user}")
        self.logged.set()

    async def start_bot(self):
        await self.start(self.token)

    async def meow_wait_for_ready(self):
        await self.logged.wait()

    async def send_message(self, channel_id, message, image=None):
        channel = self.get_channel(channel_id)
        if channel is None:
            print(f"Channel with ID {channel_id} not found.")
            return

        if image is not None:
            image_bytes = self.convert_numpy_to_bytes(image)
            file = discord.File(image_bytes, filename="image.png")
            await channel.send(message, file=file)
        else:
            await channel.send(message)

    def set_command_callback(self, callback):
        @self.event
        async def on_message(message):
            if message.author == self.user:
                return

            if message.content.startswith("^"):
                command = message.content[1:]
                await callback(command)

    @staticmethod
    def convert_numpy_to_bytes(image):
        image = image.astype(np.uint8)
        image = np.squeeze(image)
        image_pil = Image.fromarray(image, "RGBA")
        with BytesIO() as output:
            image_pil.save(output, format="PNG")
            image_pil.save("graph.png", fromat="PNG")
            image_bytes = output.getvalue()

        image_bytes = BytesIO(image_bytes)
        return image_bytes

    async def on_error(self, event, *args, **kwargs):
        # Bitch
        exc_type, exc_value, exc_traceback = sys.exc_info()
        raise exc_value.with_traceback(exc_traceback)


TOKEN_LOC = os.path.expanduser("~/.cache/discordbot_token")


def get_token() -> Optional[str]:

    if not os.path.isfile(TOKEN_LOC):
        return None

    with open(TOKEN_LOC, "r") as f:
        return ("\n".join(f.readlines())).strip()


def send_message(message):
    return None

    token = get_token()

    if token is None:
        return None

    bot = DiscordBot(token)

    async def send_message_async():
        try:
            asyncio.create_task(bot.start_bot())
            await bot.meow_wait_for_ready()
            await bot.send_message(1104132411412983848, message)
            await bot.close()
        except Exception as e:
            print("Exception in bot:", e)

    asyncio.run(send_message_async())


if __name__ == "__main__":
    send_message("Hellou")
