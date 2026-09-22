import os
import sys
import requests
from fastapi import FastAPI, Request, Response, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from rag_engine import MasviaRAG, FALLBACK_ERROR_MESSAGE, BOT_NAME, OWNER_PHONE
from langchain_core.messages import HumanMessage, AIMessage

# UTF-8 for console output
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

app = FastAPI(
    title=f"Masvia WhatsApp RAG Chatbot ({BOT_NAME})",
    description=f"Customer Service AI '{BOT_NAME}' for Jamu Celup Masvia with token reduction and owner notifications",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance & in-memory session history
rag_bot: Optional[MasviaRAG] = None
session_histories: dict[str, List] = {}

# WhatsApp Cloud API Credentials
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "masvia_secret_verify_token")


@app.on_event("startup")
def startup_event():
    global rag_bot
    try:
        rag_bot = MasviaRAG()
        print(f"✅ {BOT_NAME} RAG Engine successfully loaded & ready!")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize RAG engine on startup: {e}")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default_user"


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    cached: bool = False
    escalate_to_human: bool = False


@app.get("/")
def health_check():
    return {
        "status": "online",
        "bot_name": BOT_NAME,
        "service": "Masvia WhatsApp RAG Chatbot",
        "rag_ready": rag_bot is not None,
        "owner_notification_target": f"wa.me/{OWNER_PHONE}",
    }


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Direct API endpoint for testing chatbot responses via HTTP JSON."""
    session_id = request.session_id
    history = session_histories.get(session_id, [])

    if not rag_bot:
        return ChatResponse(
            reply=FALLBACK_ERROR_MESSAGE,
            session_id=session_id,
            cached=False,
            escalate_to_human=True,
        )

    try:
        result = rag_bot.answer_query(request.message, chat_history=history)
        answer = result["answer"]
        cached = result.get("cached", False)
        escalate = result.get("escalate_to_human", False)

        if escalate:
            notify_owner_whatsapp(session_id, request.message)

        # Update conversation history with token trim
        history.append(HumanMessage(content=request.message))
        history.append(AIMessage(content=answer))
        if len(history) > 6:
            history = history[-6:]
        session_histories[session_id] = history

        return ChatResponse(
            reply=answer,
            session_id=session_id,
            cached=cached,
            escalate_to_human=escalate,
        )
    except Exception as e:
        print(f"⚠️ Shielded API Error: {e}")
        return ChatResponse(
            reply=FALLBACK_ERROR_MESSAGE,
            session_id=session_id,
            cached=False,
            escalate_to_human=True,
        )


# =========================================================================
# WHATSAPP NOTIFICATION & WEBHOOKS
# =========================================================================

# Fonnte Gateway Token (Optional, for Scan QR mode)
FONNTE_TOKEN = os.getenv("FONNTE_TOKEN", "")


def notify_owner_whatsapp(customer_phone: str, user_text: str):
    """Sends an urgent notification to the owner's WhatsApp when customer needs human support."""
    notification_body = (
        f"🚨 *NOTIFIKASI CS MASVIA ({BOT_NAME})*\n\n"
        f"Ada pelanggan yang ingin terhubung dengan Admin Manusia!\n"
        f"👤 *Pelanggan:* {customer_phone}\n"
        f"💬 *Pesan Pelanggan:* \"{user_text}\"\n\n"
        f"Silakan segera hubungi pelanggan tersebut ya!"
    )
    print(f"\n📲 [WHATSAPP OWNER ALERT] Sending notification to {OWNER_PHONE}...")

    # 1. Send via Fonnte if FONNTE_TOKEN is configured
    if FONNTE_TOKEN:
        try:
            requests.post(
                "https://api.fonnte.com/send",
                headers={"Authorization": FONNTE_TOKEN},
                data={"target": OWNER_PHONE, "message": notification_body},
                timeout=10,
            )
            print(f"📤 Alert sent to admin via Fonnte!")
        except Exception as e:
            print(f"⚠️ Fonnte alert error: {e}")

    # 2. Send via Meta WhatsApp Cloud API if configured
    send_whatsapp_message(OWNER_PHONE, notification_body)


def send_whatsapp_message(to_phone: str, text: str):
    """Sends a text message reply back through Meta WhatsApp Cloud API."""
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        print(f"ℹ️ [SIMULATION MODE] WhatsApp message to {to_phone}:\n{text}\n")
        return

    url = f"https://graph.facebook.com/v20.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "text",
        "text": {"body": text},
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code != 200:
            print(f"❌ Failed to send WhatsApp message: {response.text}")
        else:
            print(f"📤 Successfully sent message to {to_phone}!")
    except Exception as e:
        print(f"❌ Error sending WhatsApp message: {e}")


@app.get("/webhook")
def verify_whatsapp_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    """Webhook verification endpoint required by Meta WhatsApp Cloud API."""
    if hub_mode == "subscribe" and hub_verify_token == WHATSAPP_VERIFY_TOKEN:
        print("✅ WhatsApp webhook verified successfully!")
        return Response(content=hub_challenge, media_type="text/plain")
    raise HTTPException(status_code=403, detail="Verification token mismatch")


@app.post("/webhook")
async def handle_whatsapp_message(request: Request):
    """Handles incoming WhatsApp messages and triggers the Vivi RAG pipeline."""
    payload = await request.json()

    try:
        entry = payload.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])

        if not messages:
            return {"status": "ignored_non_message_event"}

        msg = messages[0]
        user_phone = msg.get("from")
        msg_type = msg.get("type")

        if msg_type != "text":
            reply_text = (
                f"Halo Kak! 🌿 Saat ini {BOT_NAME} baru dapat menerima pesan teks. "
                "Silakan ketik pertanyaan Kakak ya!"
            )
        else:
            user_text = msg["text"]["body"]
            print(f"\n📩 Incoming WhatsApp message from {user_phone}: '{user_text}'")

            if not rag_bot:
                reply_text = FALLBACK_ERROR_MESSAGE
            else:
                history = session_histories.get(user_phone, [])
                result = rag_bot.answer_query(user_text, chat_history=history)
                reply_text = result["answer"]

                # If customer needs human admin, notify the owner's WhatsApp!
                if result.get("escalate_to_human"):
                    notify_owner_whatsapp(user_phone, user_text)

                # Update history for this user
                history.append(HumanMessage(content=user_text))
                history.append(AIMessage(content=reply_text))
                if len(history) > 6:
                    history = history[-6:]
                session_histories[user_phone] = history

        # Send response back to customer on WhatsApp
        send_whatsapp_message(user_phone, reply_text)

    except Exception as e:
        # Never crash or show technical error to customer
        print(f"❌ Error handling WhatsApp webhook: {e}")

    return {"status": "processed"}


@app.post("/webhook/fonnte")
async def fonnte_webhook(request: Request):
    """
    Native webhook receiver for Fonnte WhatsApp Gateway (Scan QR mode).
    Fonnte posts incoming WhatsApp chat (sender, message, name).
    Returns {"reply": answer} which Fonnte automatically sends back to the user!
    """
    global rag_bot
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
        else:
            form_data = await request.form()
            data = dict(form_data)

        sender = str(data.get("sender", "unknown")).strip()
        user_message = str(data.get("message", "")).strip()

        if not user_message:
            return {"reply": "", "status": False}

        print(f"\n📩 [FONNTE] Pesan masuk dari {sender}: '{user_message}'")

        if not rag_bot:
            return {"reply": FALLBACK_ERROR_MESSAGE, "status": True}

        history = session_histories.get(sender, [])
        result = rag_bot.answer_query(user_message, chat_history=history)
        reply_text = result["answer"]

        # If customer needs human admin, notify the owner's WhatsApp!
        if result.get("escalate_to_human"):
            notify_owner_whatsapp(sender, user_message)

        # Update conversation history
        history.append(HumanMessage(content=user_message))
        history.append(AIMessage(content=reply_text))
        if len(history) > 6:
            history = history[-6:]
        session_histories[sender] = history

        # Fonnte automatically sends the text in "reply" back to the customer on WhatsApp!
        return {"reply": reply_text}

    except Exception as e:
        print(f"❌ Error handling Fonnte webhook: {e}")
        return {"reply": FALLBACK_ERROR_MESSAGE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
