from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import base64
import os
from datetime import datetime
import threading
import json

app = Flask(__name__)
CORS(app)

class GenerativeArtCreator:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.is_loading = False
        self.is_loaded = False

    def load_model(self):
        if self.is_loaded or self.is_loading:
            return
        
        self.is_loading = True
        print(f"Loading model {self.model_name}...")

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            use_safetensors=True
        ).to(self.device)

        if torch.cuda.is_available():
            self.pipe.enable_xformers_memory_efficient_attention()
            self.pipe.enable_attention_slicing()

        self.is_loaded = True
        self.is_loading = False
        print("Model loaded successfully!")

    def generate_image(self, prompt, negative_prompt=None, num_images=1,
                      guidance_scale=7.5, steps=50, width=512, height=512, painting_style="oil"):
        if not self.is_loaded:
            raise Exception("Model not loaded yet")

        style_prompts = {
            "oil": "oil painting, thick brushstrokes, canvas texture, impasto technique, rich colors, classical painting",
            "watercolor": "watercolor painting, soft edges, paper texture, transparent washes, fluid brushwork, delicate colors",
            "acrylic": "acrylic painting, bold colors, textured surface, contemporary art, vibrant brushstrokes",
            "impressionist": "impressionist painting, visible brushstrokes, light and color, outdoor scene, Claude Monet style, soft focus",
            "renaissance": "renaissance painting, classical art, oil on canvas, chiaroscuro, detailed realism, old master technique",
            "abstract": "abstract art, non-representational, bold shapes, expressive colors, modern art, artistic interpretation",
            "expressionist": "expressionist painting, emotional, distorted forms, intense colors, dramatic brushwork, raw emotion",
            "baroque": "baroque painting, dramatic lighting, rich colors, ornate details, dynamic composition, theatrical"
        }

        style_enhancement = style_prompts.get(painting_style, style_prompts["oil"])
        artistic_prompt = f"{prompt}, {style_enhancement}, hand-painted, fine art, masterpiece"

        if negative_prompt:
            enhanced_negative = f"{negative_prompt}, photograph, photo, realistic, 3d render, cgi, digital art, smooth, clean, pixelated"
        else:
            enhanced_negative = "photograph, photo, realistic, 3d render, cgi, digital art, smooth, clean, sharp edges, pixelated, low quality"

        print(f"Generating {painting_style} painting for prompt: '{prompt}'")

        generator = torch.Generator(self.device).manual_seed(int(datetime.now().timestamp()))

        images = self.pipe(
            prompt=artistic_prompt,
            negative_prompt=enhanced_negative,
            num_images_per_prompt=num_images,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        ).images

        return images

    def save_image(self, image, folder="output", filename=None):
        if not os.path.exists(folder):
            os.makedirs(folder)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_art_{timestamp}.png"

        save_path = os.path.join(folder, filename)
        image.save(save_path)
        return save_path

art_generator = GenerativeArtCreator()

def load_model_background():
    art_generator.load_model()

threading.Thread(target=load_model_background, daemon=True).start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'is_loaded': art_generator.is_loaded,
        'is_loading': art_generator.is_loading,
        'device': art_generator.device
    })


@app.route('/api/generate', methods=['POST'])
def generate_art():
    try:
        data = request.json
        
        if not art_generator.is_loaded:
            return jsonify({'error': 'Model is still loading. Please wait...'}), 503

        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        painting_style = data.get('painting_style', 'oil')
        negative_prompt = data.get('negative_prompt')
        num_images = min(int(data.get('num_images', 1)), 4)
        steps = max(20, min(100, int(data.get('steps', 50))))
        width = max(256, min(1024, int(data.get('width', 512))))
        height = max(256, min(1024, int(data.get('height', 512))))
        guidance_scale = float(data.get('guidance_scale', 7.5))

        images = art_generator.generate_image(
            prompt=prompt,
            painting_style=painting_style,
            negative_prompt=negative_prompt,
            num_images=num_images,
            steps=steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale
        )

        image_data = []
        for i, img in enumerate(images):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            save_path = art_generator.save_image(img)

            image_data.append({
                'data': img_str,
                'path': save_path
            })

        return jsonify({
            'success': True,
            'images': image_data,
            'prompt': prompt
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<path:filename>')
def download_image(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

PROMPT_DB = "prompt_database.json"

if not os.path.exists(PROMPT_DB):
    with open(PROMPT_DB, "w") as f:
        json.dump([], f)


def load_prompts():
    with open(PROMPT_DB, "r") as f:
        return json.load(f)


def save_prompt(prompt):
    prompts = load_prompts()
    prompts.append(prompt)
    with open(PROMPT_DB, "w") as f:
        json.dump(prompts, f)


def suggest_prompts(user_prompt):
    prompts = load_prompts()
    suggestions = [p for p in prompts if user_prompt.lower() in p.lower()]
    return suggestions[:3]


@app.route('/api/learn', methods=['POST'])
def learn_prompt():
    data = request.json
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Prompt required"}), 400

    save_prompt(prompt)
    suggestions = suggest_prompts(prompt)

    return jsonify({
        "message": "Prompt saved and learned.",
        "suggestions": suggestions
    })


_original_generate_art = generate_art

def auto_learn_generate_art():
    response = _original_generate_art()
    try:
        data = request.json
        if data:
            prompt = data.get("prompt", "")
            if prompt.strip() != "":
                save_prompt(prompt)
    except:
        pass
    return response

app.view_functions['generate_art'] = auto_learn_generate_art

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)