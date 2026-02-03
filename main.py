import discord
from discord.ext import commands
from discord_buddy import core


class DiscordBuddyBot(commands.Bot):
    async def setup_hook(self):
        core.init_bot(self)
        await self.load_extension("discord_buddy.cogs.events")
        await self.load_extension("discord_buddy.cogs.commands")


def build_bot() -> commands.Bot:
    intents = discord.Intents.default()
    intents.message_content = True
    intents.emojis = True
    intents.members = True

    return DiscordBuddyBot(command_prefix="!", intents=intents)


if __name__ == "__main__":
    bot = build_bot()
    bot.run(core.DISCORD_TOKEN)
