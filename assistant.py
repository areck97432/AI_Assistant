from groq import Groq
from PIL import ImageGrab, Image
from openai import OpenAI
from faster_whisper import WhisperModel
import speech_recognition as sr
import google.generativeai as genai
import pyperclip
import cv2
import pyaudio
import os 
import time
import re

wake_word = 'jarvis'
groq_client = Groq(api_key='GROQ KEY')
genai.configure(api_key='GOOGLE KEY')
openai_client = OpenAI(api_key='OPENAI KEY')
web_cam = cv2.VideoCapture(1)

sys_msg = (
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg}]

generation_config = {
'temperature': 0.7,
'top_p': 1,
'top_k': 1,
'max_output_token': 2048
}

safety_settings = [
{
'category': 'HARM_CATEGORY_HARASSMENT',
'threshold': 'BLOCK_NONE'
},
{
'category': 'HARM_CATEGORY_HATE_SPEECH',
'threshold': 'BLOCK_NONE'
},
{
'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
'threshold': 'BLOCK_NONE'
},
{
'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
'threshold': 'BLOCK_NONE'
},
]
model = genai.GenerativeModel(  'gemini-1.5-flash-latest',
                                generation_config=generation_config,
                                safety_settings=safety_settings)
num_cores = os.cpu_count()
whisper_size = 'base'

whisper_model = WhisperModel(
    whisper_size,
    device = 'cpu',
    compute_type= 'int8',
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)

r = sr.Recognizer()
source = sr.Microphone()

def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n        IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)
    
    return response.content


def function_call(prompt):
    sys_msg = (
    'You are an AI function calling model. You will determine whether extracting the users clipboard content, '
    'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond '
    'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
    'respond with only one selection from this list: ["extract clipboard", “take screenshot", "capture webcam", “None"] \n'
    'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
    'function call name exactly as I listed.'
    )
    
    function_convo = [{'role': 'system', 'content': sys_msg},
                    {'role': 'user', 'content': prompt}]
    
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    
    return response.content
    
def take_screenshot():   
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)
    
    
def web_cam_capture():
    if not web_cam.isOpened():
        print('Error: Camera did not open successfully')
        exit()
        
    path = 'webcam.jpg'
    ret, frame = web_cam.read()
    cv2.imwrite(path, frame)
    
def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('No clipboard text to copy')
        return None

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
    'You are the vision analysis AI that provides semantic meaning from images to provide context '
    'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
    'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
    'relevant to the user prompt. Then generate as much objective data about the image for the AI '
    f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )

    response = model.generate_content([prompt, img])
    return response.text

def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False
    
    with openai_client.audio.speech.with_streaming_response. create(
        model='tts-1',
        voice= 'onyx',
        response_format='pcm',
        input=text,
    ) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                if max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True


def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text


def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())
    
    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text, wake_word)
    
    if clean_prompt:
        print(f"USER: {clean_prompt}")
        call = function_call(clean_prompt)
        if 'take screenshot' in call:
            print('Taking screenshot.')
            take_screenshot()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.jpg')
        elif 'capture webcam' in call:
            print('Capturing webcam.')
            web_cam_capture()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='webcam.jpg')
        elif 'extract clipboard' in call:
            print('Extracting clipboard text.')
            paste = get_clipboard_text()
            clean_prompt = f'{clean_prompt} \n\n  CLIPBOARD CONTENT: {paste}'
            visual_context = None
        else:
            visual_context = None
        
        response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
        print(f'ASSISTANT: {response}')
        speak(response)


def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print('\n Say ', wake_word, 'followed with your prompt. \n')
    r.listen_in_background(source, callback)

    while True:
        time.sleep(.5)

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)
    
    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None
    
start_listening()

        

