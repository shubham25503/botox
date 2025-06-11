from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import io
from PIL import Image
import base64
from rembg import remove

app = FastAPI(title="Stable Diffusion API 2")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base prompts for Botox and Filler treatments
base_prompts = {
    # Botox areas
    "forehead_lines_botox": "Smooth horizontal forehead lines while maintaining natural skin texture, tone, and expressions. Subtle, realistic improvement without altering facial identity.",
    "frown_lines_glabella_botox": "Reduce vertical '11' lines between eyebrows, keeping a relaxed, natural look. Preserve muscle balance and skin realism.",
    "crows_feet_botox": "Softly diminish crow's feet around the eyes while preserving eye shape and natural expressions. Maintain fine skin texture.",
    "nasalis_lines_botox": "Soften nasal 'bunny' lines, retaining natural nose contours and realistic skin appearance.",
    "vertical_lip_lines_botox": "Subtly smooth vertical wrinkles above the lips while preserving natural lip texture, curves, and surrounding skin.",
    "lip_flip_botox": "Enhance the upper lip's fullness with a natural lift near Cupid's Bow. Preserve lip shape, texture, and volume.",
    "smile_lift_botox": "Lift corners of the mouth slightly, reducing downward smile lines naturally. Maintain smile structure and facial harmony.",
    "masseter_reduction_botox": "Slightly slim the jawline by softening the masseter muscles while preserving facial symmetry and jaw contours.",
    "dimpled_chin_botox": "Smooth dimpled chin texture while keeping natural chin definition and facial proportions.",
    "platysmal_bands_botox": "Reduce vertical neck bands, creating a smoother neckline while preserving skin texture and natural contours.",

    # Filler areas
    "cheek_filler": "Add gentle volume to the cheeks with natural, lifted facial contours. Preserve skin texture, symmetry, and balance.",
    "smile_line_filler": "Subtly fill nasolabial folds (smile lines) for a smoother, youthful look while keeping natural facial movement and expressions.",
    "lip_filler": "Plump and naturally shape the lips with soft, balanced volume enhancement. Maintain lip texture and proportions.",
    "temple_filler": "Restore lost volume in the temples for a refreshed, youthful contour while preserving natural facial lines and textures.",
    "nose_filler": "Smooth and refine the nasal bridge and tip with subtle, natural contour improvements. Maintain original nose shape."
}

# Max units for Botox areas
max_units = {
    "forehead_lines_botox": 30,
    "frown_lines_glabella_botox": 25,
    "crows_feet_botox": 30,
    "nasalis_lines_botox": 15,
    "vertical_lip_lines_botox": 8,
    "lip_flip_botox": 6,
    "smile_lift_botox": 12,
    "masseter_reduction_botox": 60,
    "dimpled_chin_botox": 8,
    "platysmal_bands_botox": 30
}

# Define areas
BOTOX_AREAS = set(max_units.keys())
FILLER_AREAS = set(area for area in base_prompts if area not in BOTOX_AREAS)

# Common negative prompt elements
common_negative_prompt = (
    "changed face, changed skin tone, mutated hands, blurry, deformed, bad anatomy, disfigured, mutation, "
    "fused fingers, too many fingers, long neck, cloned face, duplicate face, alien, plastic, waxy, cartoon, "
    "unnatural skin, glowing skin, anime, identity change, poorly drawn face"
)

# Initialize the model
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")

def build_prompt(area: str, injection_number: int) -> (str, float):
    """Builds the prompt and strength based on treatment type and units."""
    treatment_name = area.replace("_", " ").title()

    if area in BOTOX_AREAS:
        max_area_units = max_units.get(area, 30)
        normalized_units = min(injection_number / max_area_units, 1.0)
        effect_strength = 0.35 + (0.3 * (normalized_units ** 0.7))
        strength = min(effect_strength, 0.375)  # Slightly lower max cap for safety

        prompt = (
            f"High-quality medical photograph after {injection_number} units of Botox in the {treatment_name} area. "
            f"{base_prompts.get(area, '')} Eyes, facial features, and skin tone remain completely unchanged. "
            "No artistic changes. Strictly realistic and medically accurate."
        )

    elif area in FILLER_AREAS:
        strength = 0.375  # Filler generally uses fixed, subtle strength
        prompt = (
            f"High-quality medical photograph after filler treatment in the {treatment_name} area. "
            f"{base_prompts.get(area, '')} Eyes, facial features, and skin tone remain completely unchanged. "
            "No artistic changes. Strictly realistic and medically accurate."
        )

    else:
        raise ValueError(f"Unknown treatment area: {area}")

    return prompt, strength

@app.post("/generate")
async def generate_image(
    injection_number: int = Form(...),
    selected_area: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # Read and process the image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        
        # Build prompt and get strength
        prompt, strength = build_prompt(selected_area, injection_number)
        
        # Generate image
        output = pipe(
            prompt=prompt,
            negative_prompt=common_negative_prompt,
            image=image,
            strength=strength,
            guidance_scale=8.5
        ).images[0]

        # Convert to base64
        buffered = io.BytesIO()
        output.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "image": img_str
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove-background")
async def remove_background_endpoint(file: UploadFile = File(...)):
    try:
        # Read the input image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        
        # Remove the background
        output_image = remove(image)
        
        # Convert to base64
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "status": "success",
            "image": img_str
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": model_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8889)