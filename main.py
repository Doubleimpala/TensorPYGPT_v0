import discord
from discord import app_commands
from discord.ext import commands
import os
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")

bot = commands.Bot(command_prefix=">", intents=discord.Intents.all())
TOKEN = "ODQ3NDMzMTQ5MjkzMjY0ODk3.G-z_D7.nOz9Y_rJa0YhUd5iG82WHJZtgDSrGopEwC0GJA"

@bot.event
async def on_ready():
    print("Bot is ready")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} commands")
    except Exception as e:
        print(e)


@bot.tree.command(name="chat")
@app_commands.describe(arg="Prompt")
async def chat(interaction: discord.Interaction, arg: str):
    bot_input_ids = tokenizer.encode(arg + tokenizer.eos_token, return_tensors='pt')

    chat_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    messagebot = tokenizer.decode(chat_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    if (messagebot) == "":
        messagebot = "TensorPY did not find a suitable response."
    print(str(interaction.user) + ": " + arg)
    print("Tensor: " + messagebot)
    await interaction.response.send_message(messagebot)

bot.run(TOKEN)
print("Running")




