"""Быстрый тест доставки в Telegram (для GitHub Actions / локально с VPN)."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from telegram import Bot

BOT_TOKEN = (os.environ.get("BOT_TOKEN") or "7831097053:AAE5lFirdDiDbdCa45eLh3k5tuYrWYLVS00").strip()
CHAT_ID = (os.environ.get("CHAT_ID") or "283220567").strip()


async def main() -> None:
    if not BOT_TOKEN or not CHAT_ID:
        raise SystemExit("Нужны BOT_TOKEN и CHAT_ID в окружении.")

    bot = Bot(token=BOT_TOKEN)
    me = await bot.get_me()
    print(f"Bot OK: @{me.username}")
    now = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%Y-%m-%d %H:%M:%S %Z")
    text = (
        "✅ Тест Сигналы РФ (GitHub Actions)\n"
        f"Время МСК: {now}\n"
        "Уведомления доходят."
    )
    msg = await bot.send_message(chat_id=CHAT_ID, text=text)
    print(f"Message sent, id={msg.message_id}")


if __name__ == "__main__":
    asyncio.run(main())
