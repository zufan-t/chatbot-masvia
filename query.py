import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from rag_engine import MasviaRAG

# Force UTF-8 for emoji support on Windows
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()
BOT_NAME = os.getenv("BOT_NAME", "Vivi")
OWNER_PHONE = os.getenv("OWNER_PHONE_NUMBER", "6289504573745")


def main():
    print("=" * 65)
    print(f"🍵 SIMULASI CHATBOT WHATSAPP: {BOT_NAME.upper()} (JAMU MASVIA)")
    print("=" * 65)
    print("Ketik pertanyaan Anda (misal: 'berapa harganya?', 'apakah pahit?').")
    print("Coba tanyakan hal yang sama untuk menguji Token Cache Memory ⚡")
    print("Coba ketik 'mau bicara dengan admin' untuk menguji Notifikasi WhatsApp 📲")
    print("Ketik 'exit' atau 'keluar' untuk mengakhiri sesi.\n")

    try:
        bot = MasviaRAG()
    except Exception as e:
        print(f"\n❌ Gagal menginisialisasi sistem: {e}")
        return

    chat_history = []
    print(f"\n✅ {BOT_NAME} siap melayani! Silakan mulai percakapan:\n")

    while True:
        try:
            user_input = input("Pelanggan (Anda): ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "keluar", "quit", "q"]:
                print(f"\nTerima kasih telah berkonsultasi dengan {BOT_NAME}! Salam sehat selalu! 🌿💚")
                break

            print(f"\n{BOT_NAME} sedang mengetik... 💬")
            result = bot.answer_query(user_input, chat_history=chat_history)
            answer = result["answer"]

            # Display Cache Status (Token Reduction)
            if result.get("cached"):
                c_type = result.get("cache_type", "")
                if c_type == "similar_context":
                    matched = result.get("matched_query", "")
                    sim_pct = result.get("similarity", 0.0) * 100
                    print(f"⚡ [SIMILAR CONTEXT CACHE HIT]: Konteks serupa dengan '{matched}' ({sim_pct:.1f}% match)! Menggunakan jawaban sebelumnya (0 token Groq).")
                elif c_type == "session_memory":
                    print("⚡ [SESSION MEMORY HIT]: Pertanyaan telah dijawab di percakapan ini! Menggunakan jawaban sebelumnya (0 token Groq).")
                else:
                    print("⚡ [EXACT QUERY CACHE HIT]: Pertanyaan persis terdeteksi di memori! Menggunakan jawaban sebelumnya (0 token Groq).")

            # Display Guardrail Status
            if result.get("shield_blocked"):
                reason = result.get("block_reason", "security")
                if reason == "prompt_injection":
                    print("🛡️ [GUARDRAIL AKTIF]: Upaya Prompt Injection / Manipulasi Sistem ditangkal otomatis (0 Token Groq)!")
                else:
                    print("🛡️ [GUARDRAIL AKTIF]: Pertanyaan di luar topik produk Masvia ditangkal otomatis (0 Token Groq)!")

            if result.get("escalate_to_human"):
                print(f"🚨 [NOTIFIKASI WHATSAPP TERKIRIM KE OWNER]:")
                print(f"   📲 Mengirim alert ke: wa.me/{OWNER_PHONE}")
                print(f"   ✉️ Pesan: 'Pelanggan butuh admin manusia! Request: \"{user_input}\"'")

            print(f"\n{BOT_NAME}:\n{answer}\n")
            print("-" * 65)

            # Update chat history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

            # Trim history to reduce tokens
            if len(chat_history) > 6:
                chat_history = chat_history[-6:]

        except KeyboardInterrupt:
            print("\n\nSesi diakhiri.")
            break
        except Exception as e:
            # User will never see raw crash tracebacks
            print(f"\n{BOT_NAME}:\nMohon maaf Kak, terjadi kendala teknis. Tim Masvia siap membantu Kakak via WhatsApp {OWNER_PHONE}! 🌿\n")


if __name__ == "__main__":
    main()
