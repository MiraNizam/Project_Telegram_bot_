import nest_asyncio
from telegram import Update  # new information from server
from telegram.ext import ApplicationBuilder  # to create and configure an application (telegram bot)
from telegram.ext import MessageHandler  # Handler - create acts(func) for activities
from telegram.ext import filters

from chat_bot import bot

# Update - information received from the server (new messages, new contacts)
# Updates come regularly from the server with new information
# Function for MessageHandler, call it on every message to the bot
nest_asyncio.apply()

# func for MessageHandler, call on every message
async def reply(update: Update) -> None:
    question = update.message.text
    reply = bot(question)
    print(f"> {question}")
    print(f"< {reply}")
    await update.message.reply_text(reply)  # Response User

# t.me/NotBorn_bot

TOKEN = ""  # at BotFather add TOKEN
app = ApplicationBuilder().token(TOKEN).build()

# Create text message handler
handler = MessageHandler(filters.Text(), reply)
# Add handler to App
app.add_handler(handler)
app.run_polling()
