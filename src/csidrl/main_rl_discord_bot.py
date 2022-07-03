import sys

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from csidrl.discord_control.discordbot import DiscordBot
import asyncio

def graph():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Plot the graph
    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Dummy Graph')

    # Render the graph to a NumPy array
    fig = plt.gcf()
    fig.canvas.draw()
    graph_array = np.array(fig.canvas.renderer.buffer_rgba())
    return graph_array

def handle_exception(task):
    try:
        task.result()  # Raises an exception if the task has failed
    except Exception as e:
        print("Exception:", e)


# Initialize the Discord bot
bot = DiscordBot(
    "NTA4NjQzNTk4NTk4Nzk5Mzcw.GWVeuV.LItpmiNG9uQiriPrxrWsSaygwZoO6IPEOd46LM"
)


# Define a callback function for non-bot commands
async def command_callback(command):
    print(f"Received command: {command}")
    # Send a message with an image
    image_array = graph()
    await bot.send_message(1104132411412983848, "Hello Discord!", image=image_array)
    print(f"Finished: {command}")


async def main():
    try:
        print("Starting bot...")
        bot.set_command_callback(command_callback)
        task = asyncio.create_task(bot.start_bot())
        task.add_done_callback(handle_exception)

        await bot.meow_wait_for_ready()

        await bot.send_message(1104132411412983848, "Initial message")

        await task
    except Exception as e:
        print("Exception in main:", e)


loop = asyncio.new_event_loop()
loop.run_until_complete(main())
loop.close()