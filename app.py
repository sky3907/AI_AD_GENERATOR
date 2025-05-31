# 📦 Install all required packages

from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env

# Now you can access them like this:
api_key = os.getenv("API_KEY")

# 📥 Import all libraries
import os
import torch
import cv2
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO
from moviepy.editor import ImageSequenceClip, AudioFileClip
from diffusers import StableDiffusionPipeline
from rembg import remove
from io import BytesIO
import random
import gradio as gr

# 💾 Setup paths
OUTPUT_FOLDER = "output_ad"
IMAGE_FOLDER = os.path.join(OUTPUT_FOLDER, "images")
AUDIO_FILE = os.path.join(OUTPUT_FOLDER, "voiceover.mp3")
FINAL_VIDEO = os.path.join(OUTPUT_FOLDER, "final_ad.mp4")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# 🎯 Load YOLOv8
model = YOLO("yolov8n.pt")

# 🔍 Object Detection & Background Removal
def extract_object(image_path):
    with open(image_path, "rb") as f:
        input_bytes = f.read()
        output_bytes = remove(input_bytes)
        obj_image = Image.open(BytesIO(output_bytes)).convert("RGBA")

        obj_path = os.path.join(IMAGE_FOLDER, f"object_{os.path.basename(image_path)}")
        obj_image.save(obj_path)

    image_cv = cv2.imread(image_path)
    results = model(image_cv)
    label = "soft toy"
    for r in results:
        if len(r.boxes.cls) > 0:
            cls_id = int(r.boxes.cls[0])
            label = model.names[cls_id]
    return obj_path, None, label

# 🎨 Background Generation
def generate_ai_backgrounds(prompts=None):
    if prompts is None:
        prompts = [
            "a dining table", "a picnic blanket", "a vibrant farmer's market stand",
            "green background", "a clean white kitchen counter and a glass of apple juice"
        ]
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda")

    generated_backgrounds = []
    for prompt in prompts:
        image = pipe(prompt).images[0]
        bg_save_path = os.path.join(IMAGE_FOLDER, f"generated_bg_{random.randint(0, 9999)}.png")
        image.save(bg_save_path)
        generated_backgrounds.append(bg_save_path)
    return generated_backgrounds

# 📝 Script Generator
def generate_script_from_detection(detected_label, custom_script=None):
    # Use the custom script if provided, else use the default format
    if custom_script:
        return custom_script
    else:
        label_text = detected_label or "product"
        lines = [
            f"Introducing our fresh and juicy {label_text}, straight from the orchard.",
            f"Each bite is bursting with crisp flavor and natural sweetness.",
            f"Perfect for snacking, baking, or blending into your favorite recipes.",
            f"Add a healthy and delicious touch to your day with this premium {label_text}.",
            f"One bite of our {label_text}, and you'll taste the farm-fresh difference.",
        ]
        return " ".join(lines)

# 🖼 Merge with Backgrounds
def cut_and_merge_with_background(object_path, background_paths):
    obj_image = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)
    if obj_image.shape[2] == 3:
        obj_image = cv2.cvtColor(obj_image, cv2.COLOR_BGR2BGRA)

    first_bg = cv2.imread(background_paths[0])
    target_h, target_w, _ = first_bg.shape

    final_images = []
    for bg_path in background_paths:
        background = cv2.imread(bg_path)
        background = cv2.resize(background, (target_w, target_h))
        h, w, _ = background.shape
        obj_resized = cv2.resize(obj_image, (int(w * 0.3), int(h * 0.3)))
        mask = obj_resized[:, :, 3]
        obj_rgb = obj_resized[:, :, :3]

        x_offset = random.randint(0, max(0, w - obj_rgb.shape[1]))
        y_offset = h - obj_rgb.shape[0] - 50

        blended = background.copy()
        for c in range(3):
            blended[y_offset:y_offset + obj_rgb.shape[0], x_offset:x_offset + obj_rgb.shape[1], c] = \
                blended[y_offset:y_offset + obj_rgb.shape[0], x_offset:x_offset + obj_rgb.shape[1], c] * \
                (1 - mask / 255.0) + obj_rgb[:, :, c] * (mask / 255.0)

        output_path = os.path.join(IMAGE_FOLDER, f"final_{os.path.basename(bg_path)}")
        cv2.imwrite(output_path, blended)
        final_images.append(output_path)
    return final_images

# 🎤 Generate Voiceover
def generate_voiceover(script_text):
    print("Generating voiceover...")
     # Replace in production
    VOICE_ID = "pNInz6obpgDQGcFmaJgB"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    data = {"text": script_text, "model_id": "eleven_monolingual_v1"}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        with open(AUDIO_FILE, "wb") as f:
            f.write(response.content)
        return AUDIO_FILE
    else:
        print("Voiceover failed:", response.text)
        return None

# 🎬 Video Creator
def create_video(image_paths, audio_path, output_video, fps=24):
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    
    # Calculate frame duration based on total audio duration and number of images
    frame_duration = duration / len(image_paths)
    
    # Create video with specified fps
    video_clip = ImageSequenceClip(image_paths, fps=fps)
    
    # If the video is shorter than audio, loop the video
    if video_clip.duration < audio_clip.duration:
        video_clip = video_clip.loop(duration=audio_clip.duration)
    else:
        # If video is longer, trim it
        video_clip = video_clip.subclip(0, audio_clip.duration)
    
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(output_video, codec="libx264", fps=fps)
    return output_video

# 🚀 Main pipeline
def gradio_ad_generator(image1, image2, background_prompts, custom_script, fps=24, progress=gr.Progress()):
    progress(0, desc="Starting process...")
    
    # Process input images
    input_paths = []
    for i, img in enumerate([image1, image2]):
        if img is None:
            continue
        path = os.path.join(IMAGE_FOLDER, f"input_{i}.png")
        img.save(path)
        input_paths.append(path)
    
    if not input_paths:
        return "Please upload at least one product image"

    progress(0.1, desc="Processing images...")
    all_final_images = []
    script_parts = []
    
    # Use the custom background prompts if provided
    if background_prompts:
        background_prompts = background_prompts.split(",")  # Split by commas
        background_prompts = [p.strip() for p in background_prompts if p.strip()]  # Clean up
    else:
        background_prompts = [
            "a dining table", "a picnic blanket", "a vibrant farmer's market stand",
            "green background", "a clean white kitchen counter and a glass of apple juice"
        ]
    
    # Generate the script with custom or auto-generated content
    progress(0.2, desc="Extracting objects...")
    for img_path in input_paths:
        object_path, _, label = extract_object(img_path)
        progress(0.3, desc=f"Generating backgrounds...")
        bg_paths = generate_ai_backgrounds(background_prompts)
        progress(0.5, desc="Merging objects with backgrounds...")
        final_images = cut_and_merge_with_background(object_path, bg_paths)
        all_final_images.extend(final_images)
        
        # Generate the script either from the default or custom input
        script_parts.append(generate_script_from_detection(label, custom_script))

    progress(0.6, desc="Generating voiceover...")
    script_text = " ".join(script_parts)
    voiceover = generate_voiceover(script_text)
    
    if voiceover:
        progress(0.8, desc="Creating final video...")
        output = create_video(all_final_images, voiceover, FINAL_VIDEO, fps=int(fps))
        progress(1.0, desc="Complete!")
        return output
    else:
        return "Voiceover generation failed"
        
# Define a pastel theme with custom styling
theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="purple",
    neutral_hue="sky",
    radius_size=gr.themes.sizes.radius_lg,
    text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill="#F8F9FF",
    body_background_fill_dark="#F8F9FF",
    background_fill_primary="#FDF4F5",
    background_fill_primary_dark="#FDF4F5",
    background_fill_secondary="#F9F5FF",
    background_fill_secondary_dark="#F9F5FF",
    border_color_accent="#F472B6",
    border_color_accent_dark="#F472B6",
    color_accent="#EC4899",
    button_primary_background_fill="#EC4899",
    button_primary_background_fill_dark="#EC4899",
    button_primary_background_fill_hover="#DB2777",
    button_primary_background_fill_hover_dark="#DB2777",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    block_label_text_color="#6B7280",
    block_label_text_color_dark="#6B7280",
    block_title_text_color="#4B5563",
    block_title_text_color_dark="#4B5563",
    input_background_fill="#FDF4F5",
    input_background_fill_dark="#FDF4F5",
    input_border_color="#F9A8D4",
    input_border_color_dark="#F9A8D4",
    block_shadow="0px 4px 6px rgba(0, 0, 0, 0.05)",
)

# Additional custom CSS for styling refinements with pastel colors
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

.main-header {
    text-align: center !important;
    animation: pastel-glow 3s ease-in-out infinite alternate;
    background: linear-gradient(90deg, #EC4899, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -0.025em !important;
    line-height: 1.2 !important;
    padding-bottom: 10px !important;
    margin-bottom: 20px !important;
    border-bottom: 1px solid #F9A8D4 !important;
}

.main-subtitle {
    text-align: center !important;
    margin-bottom: 30px !important;
    color: #6B7280 !important;
    font-weight: 400 !important;
}

@keyframes pastel-glow {
    from {
        text-shadow: 0 0 5px rgba(236, 72, 153, 0.5);
    }
    to {
        text-shadow: 0 0 15px rgba(167, 139, 250, 0.7);
    }
}

.container {
    border-radius: 16px !important;
    background-color: #FDF4F5 !important;
    padding: 24px !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
    border: 1px solid #FBCFE8 !important;
    margin-bottom: 24px !important;
}

.tab-nav {
    border-bottom: 1px solid #FBCFE8 !important;
    padding-bottom: 10px !important;
    margin-bottom: 20px !important;
}

.section-header {
    color: #4B5563 !important;
    padding: 8px 0 !important;
    margin-bottom: 12px !important;
    border-bottom: 1px solid #FBCFE8 !important;
    font-weight: 600 !important;
}

.footer {
    text-align: center !important;
    margin-top: 40px !important;
    padding-top: 20px !important;
    border-top: 1px solid #FBCFE8 !important;
    color: #6B7280 !important;
    font-size: 0.9em !important;
}

.app-status-indicator {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 8px 12px !important;
    background-color: #FBCFE8 !important;
    border: 1px solid #F9A8D4 !important;
    border-radius: 12px !important;
    margin-bottom: 15px !important;
    color: #BE185D !important;
}

.status-dot {
    height: 10px !important;
    width: 10px !important;
    background-color: #BE185D !important;
    border-radius: 50% !important;
    display: inline-block !important;
}

.accordion-wrapper {
    border: 1px solid #FBCFE8 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    margin-top: 15px !important;
}

.accordion-header {
    background-color: #FBCFE8 !important;
    padding: 12px 18px !important;
    cursor: pointer !important;
    font-weight: 600 !important;
    color: #BE185D !important;
}

.accordion-content {
    padding: 18px !important;
    background-color: #FDF4F5 !important;
}

/* Component-specific styling */
.file-preview {
    border: 1px dashed #F9A8D4 !important;
    border-radius: 12px !important;
    padding: 12px !important;
    transition: all 0.3s !important;
    background-color: #FEF2F2 !important;
}

.file-preview:hover {
    border-color: #F472B6 !important;
    background-color: #FECDD3 !important;
}

.input-box {
    background-color: #FDF4F5 !important;
    border: 1px solid #F9A8D4 !important;
    border-radius: 12px !important;
    transition: all 0.3s !important;
}

.input-box:focus {
    border-color: #F472B6 !important;
    box-shadow: 0 0 0 2px rgba(236, 72, 153, 0.2) !important;
}

.generate-btn {
    background: linear-gradient(90deg, #EC4899, #A78BFA) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.025em !important;
    transition: all 0.3s !important;
    border-radius: 12px !important;
}

.generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 15px -3px rgba(236, 72, 153, 0.3) !important;
}

/* Image preview enhancements */
.image-preview-wrapper {
    position: relative !important;
    overflow: hidden !important;
    border-radius: 12px !important;
    background-color: #FEF2F2 !important;
    border: 1px solid #FBCFE8 !important;
    transition: all 0.3s !important;
}

.image-preview-wrapper:hover {
    border-color: #F472B6 !important;
}

/* Progress bar styling */
.progress-bar-bg {
    background-color: #FCE7F3 !important;
    height: 8px !important;
    border-radius: 4px !important;
    margin: 10px 0 !important;
}

.progress-bar-fill {
    background: linear-gradient(90deg, #EC4899, #A78BFA) !important;
    height: 100% !important;
    border-radius: 4px !important;
    transition: width 0.3s ease-in-out !important;
}

/* Feature cards */
.feature-card {
    background-color: #FDF4F5 !important;
    border: 1px solid #FBCFE8 !important;
    border-radius: 12px !important;
    padding: 18px !important;
    margin-bottom: 15px !important;
    transition: all 0.3s !important;
}

.feature-card:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 10px 15px -3px rgba(236, 72, 153, 0.2) !important;
    background-color: #FCE7F3 !important;
}

.feature-icon {
    font-size: 2em !important;
    color: #DB2777 !important;
    margin-bottom: 10px !important;
}

.feature-title {
    font-weight: 600 !important;
    color: #9D174D !important;
    margin-bottom: 5px !important;
}

.feature-description {
    color: #4B5563 !important;
    font-size: 0.9em !important;
}
"""

# Create a themed Gradio blocks interface
with gr.Blocks(theme=theme, css=custom_css) as demo:
    # Header section
    gr.HTML("""
        <h1 class="main-header">🎬 AI Advertisement Generator</h1>
        <p class="main-subtitle">Create professional product ads with AI-powered backgrounds, object extraction, and voiceover</p>
    """)
    
    # App status indicator
    gr.HTML("""
        <div class="app-status-indicator">
            <span class="status-dot"></span>
            <span>System ready</span>
        </div>
    """)
    
    # Main tabs
    with gr.Tabs() as tabs:
        # Creator tab
        with gr.Tab("🎨 Create Ad", id=0):
            with gr.Row():
                # Left column - Image inputs
                with gr.Column():
                    gr.HTML('<div class="section-header">📷 Product Images</div>')
                    
                    with gr.Group():
                        image1 = gr.Image(
                            type="pil", 
                            label="Primary Product Image", 
                            elem_classes="file-preview"
                        )
                        image2 = gr.Image(
                            type="pil", 
                            label="Secondary Product (Optional)", 
                            elem_classes="file-preview"
                        )
                
                # Right column - Settings
                with gr.Column():
                    gr.HTML('<div class="section-header">⚙ Ad Settings</div>')
                    
                    background_prompts = gr.Textbox(
                        lines=2,
                        label="Background Scene Prompts",
                        placeholder="Enter background prompts separated by commas (e.g. 'a sunny beach, a forest, a city skyline')",
                        elem_classes="input-box"
                    )
                    
                    custom_script = gr.Textbox(
                        lines=4,
                        label="Custom Script (Optional)",
                        placeholder="Enter a custom script for the ad voiceover. If left empty, a script will be auto-generated based on detected objects.",
                        elem_classes="input-box"
                    )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        fps_slider = gr.Slider(
                            minimum=0,
                            maximum=30,
                            value=24,
                            step=1,
                            label="Video FPS"
                        )
            
            # Generation button and output
            with gr.Row():
                generate_btn = gr.Button(
                    "🚀 Generate Advertisement", 
                    variant="primary",
                    elem_classes="generate-btn"
                )
                
            output_video = gr.Video(label="🎥 Generated Advertisement")
            
        # How It Works tab
        with gr.Tab("📖 How It Works", id=1):
            gr.HTML("""
                <div class="container">
                    <h2 style="color: #9D174D; margin-bottom: 20px;">The AI Ad Creation Process</h2>
                    
                    <div class="feature-card">
                        <div class="feature-icon">🔍</div>
                        <div class="feature-title">1. Object Detection & Extraction</div>
                        <div class="feature-description">Your product images are analyzed using YOLO object detection to identify what's in the image. The background is automatically removed to isolate the product.</div>
                    </div>
                    
                    <div class="feature-card">
                        <div class="feature-icon">🎨</div>
                        <div class="feature-title">2. Background Generation</div>
                        <div class="feature-description">Based on your prompts, AI generates beautiful, contextually appropriate backgrounds using Stable Diffusion technology.</div>
                    </div>
                    
                    <div class="feature-card">
                        <div class="feature-icon">🎤</div>
                        <div class="feature-title">3. Script & Voiceover Creation</div>
                        <div class="feature-description">A professional-quality script is written based on detected objects or your custom text, then converted to natural-sounding speech.</div>
                    </div>
                    
                    <div class="feature-card">
                        <div class="feature-icon">🎬</div>
                        <div class="feature-title">4. Final Video Assembly</div>
                        <div class="feature-description">All elements are combined into a polished advertisement with your product seamlessly composited onto the AI backgrounds, synchronized with the voiceover.</div>
                    </div>
                </div>
                
                <div class="container">
                    <h2 style="color: #9D174D; margin-bottom: 20px;">Tips for Best Results</h2>
                    <ul style="color: #4B5563; line-height: 1.6;">
                        <li><strong>Product Images:</strong> Use high-quality images with good lighting and clear focus on the product.</li>
                        <li><strong>Background Prompts:</strong> Be specific with descriptions (e.g., "modern kitchen with marble countertop" rather than just "kitchen").</li>
                        <li><strong>Custom Scripts:</strong> Keep your script between 3-5 sentences for optimal timing with the visuals.</li>
                        <li><strong>Multiple Products:</strong> For best results, use products that are thematically related for a cohesive ad.</li>
                    </ul>
                </div>
            """)
        
        # Examples tab
        with gr.Tab("💡 Examples", id=2):
            gr.HTML("""
                <div class="container">
                    <h2 style="color: #9D174D; margin-bottom: 20px;">Example Configurations</h2>
                    
                    <div style="border-left: 4px solid #EC4899; padding-left: 15px; margin-bottom: 25px; background-color: #FCE7F3; border-radius: 8px; padding: 15px;">
                        <h3 style="color: #9D174D; margin-bottom: 10px;">Food Product Advertisement</h3>
                        <p style="color: #4B5563; margin-bottom: 5px;"><strong>Background Prompts:</strong> elegant restaurant table setting, rustic wooden table with sunlight, clean kitchen counter with ingredients, farmer's market display</p>
                        <p style="color: #4B5563; margin-bottom: 5px;"><strong>Sample Script:</strong> "Introducing our artisanal cheese selection. Crafted with care from the finest ingredients. Elevate your dining experience with our premium flavors. Available now at select specialty stores."</p>
                    </div>
                    
                    <div style="border-left: 4px solid #8B5CF6; padding-left: 15px; margin-bottom: 25px; background-color: #F3E8FF; border-radius: 8px; padding: 15px;">
                        <h3 style="color: #6D28D9; margin-bottom: 10px;">Fashion Product Advertisement</h3>
                        <p style="color: #4B5563; margin-bottom: 5px;"><strong>Background Prompts:</strong> urban street scene with models, luxury boutique interior, sunset beach scene, fashion runway</p>
                        <p style="color: #4B5563; margin-bottom: 5px;"><strong>Sample Script:</strong> "Discover our exclusive summer collection. Designed for comfort and style in mind. Turn heads with our premium craftsmanship. Express yourself with [your brand] this season."</p>
                    </div>
                    
                    <div style="border-left: 4px solid #60A5FA; padding-left: 15px; margin-bottom: 25px; background-color: #EFF6FF; border-radius: 8px; padding: 15px;">
                        <h3 style="color: #1E40AF; margin-bottom: 10px;">Tech Product Advertisement</h3>
                        <p style="color: #4B5563; margin-bottom: 5px;"><strong>Background Prompts:</strong> modern office workspace, minimalist home desk setup, creative studio environment, futuristic tech lab</p>
                        <p style="color: #4B5563; margin-bottom: 5px;"><strong>Sample Script:</strong> "Introducing the next evolution in technology. Seamlessly integrate this device into your daily workflow. Experience unprecedented performance and reliability. Upgrade your life with innovation that matters."</p>
                    </div>
                </div>
            """)
    
    # Footer
    gr.HTML("""
        <div class="footer">
            <p>Built with AI-powered advertisement technology © 2025</p>
            <p style="font-size: 0.8em; margin-top: 5px;">Using YOLO for object detection, Stable Diffusion for background generation, and ElevenLabs for voiceover</p>
        </div>
    """)
    
    # Connect the button to the generation function
    generate_btn.click(
        fn=gradio_ad_generator,
        inputs=[image1, image2, background_prompts, custom_script, fps_slider],
        outputs=output_video
    )

# Launch the application
demo.launch(share=True)