import os
import torch
from fastapi import FastAPI, Request, BackgroundTasks
import telegram
from transformers import MBartForConditionalGeneration, MBartTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

TOKEN = os.environ.get("BOT_TOKEN")
if not TOKEN:
    print("ошибка: BOT_TOKEN не найден", flush=True)

bot = telegram.Bot(token=TOKEN)
app = FastAPI()

print("загрузка модели...", flush=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_mbart = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25").to(device)
state_dict = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/IlyaGusev/mbart_ru_sum_gazeta/resolve/main/pytorch_model.bin",
    map_location=device
)
model_mbart.load_state_dict(state_dict, strict=False)
tokenizer_mbart = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
tokenizer_mbart.src_lang = 'ru_RU'
tokenizer_mbart.tgt_lang = 'ru_RU'
model_mbart.eval()
print("модель загружена", flush=True)

def clean_summary(text):
    sentences = text.split('. ')
    seen = set()
    unique = []
    for s in sentences:
        if s not in seen and len(s) > 5:
            seen.add(s)
            unique.append(s)
    return '. '.join(unique)

def summarize(text, max_length=150, min_length=50):
    if not text or len(text) < 20:
        return text
    inputs = tokenizer_mbart(text, max_length=1024, truncation=True, padding=True, return_tensors='pt').to(device)
    gen_params = {
        'max_length': max_length,
        'min_length': min_length,
        'num_beams': 5,
        'early_stopping': True,
        'no_repeat_ngram_size': 3,
        'repetition_penalty': 1.4,
        'do_sample': False,
        'forced_bos_token_id': tokenizer_mbart.lang_code_to_id['ru_RU']
    }
    with torch.no_grad():
        summary_ids = model_mbart.generate(inputs['input_ids'], **gen_params)
    return clean_summary(tokenizer_mbart.decode(summary_ids[0], skip_special_tokens=True))

def hierarchical_summarize(text, chunk_size=800, chunk_overlap=100):
    if len(text) < 200:
        return summarize(text, max_length=100, min_length=30)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
        separators=['\n', '\n\n', '. ', '!', '?', ';', ' ', '']
    )
    chunks = splitter.split_text(text)
    chunk_summaries = []
    for chunk in chunks:
        if len(chunk) > 100:
            summary = summarize(chunk, max_length=80, min_length=20)
            chunk_summaries.append(summary if summary and len(summary) > 10 else chunk.split('.')[0] + '.')
        else:
            chunk_summaries.append(chunk)
    combined = ' '.join(chunk_summaries)
    return summarize(combined, max_length=150, min_length=50) if len(combined) > 400 else combined

async def process_and_reply(chat_id, text):
    try:
        await bot.send_chat_action(chat_id=chat_id, action="typing") # показываем, что бот печатает
        msg = await bot.send_message(chat_id=chat_id, text="обрабатываю...") #текстовое уведомление
        print(f"обработка текста от {chat_id}: {text[:50]}...", flush=True)
        result = hierarchical_summarize(text)
        print(f"результат ({len(result)} символов)", flush=True)
        
        await bot.delete_message(chat_id=chat_id, message_id=msg.message_id) # удаляем уведомление
        
        await bot.send_message(chat_id=chat_id, text=result[:4000]) # отправляем результат
        
    except Exception as e:
        print(f"ошибка: {e}", flush=True)
        await bot.send_message(chat_id=chat_id, text=f"Ошибка: {e}")

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    update = telegram.Update.de_json(data, bot)
    
    # игнорируем команды и пустые сообщения
    if update.message and update.message.text and not update.message.text.startswith('/'):
        print(f"добавляем задачу для {update.message.chat.id}", flush=True)
        background_tasks.add_task(process_and_reply, update.message.chat.id, update.message.text)

    return {"ok": True}

@app.get("/")
def home():
    return {"status": "ok"}