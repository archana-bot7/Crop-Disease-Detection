import html
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import pipeline

MODEL_PATH = Path(__file__).resolve().parent / "model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_CROPS = ["Maize", "Paddy", "Potato", "Tomato"]

GUIDES = {'Maize___Common_rust': {'title': 'Common Rust', 'type': 'Fungal disease', 'symptoms': 'Rust-coloured to dark-brown raised pustules on leaves. Severe infection can cause yellowing and leaf drying.', 'actions': ['Use a locally recommended resistant/tolerant maize hybrid where available.', 'Scout regularly, especially during cool, humid and wet weather.', 'Avoid leaving heavily infected crop residue where it can maintain inoculum; follow local rotation and sanitation advice.'], 'medicine': 'Fungicide treatment is not always necessary. If disease is severe and early, ask an agriculture technician about a maize-labelled fungicide containing an active ingredient such as azoxystrobin or propiconazole. Use only if the current Nepal label lists maize and this disease.', 'note': 'The Nepal National Maize Research Program notes that sowing time can influence common-rust incidence and recommends locally adapted management.'}, 'Maize___Gray_leaf_spot': {'title': 'Gray Leaf Spot', 'type': 'Fungal disease', 'symptoms': 'Long, narrow, rectangular grey-to-tan lesions, often limited by leaf veins. Lesions can merge and dry large areas of the leaf.', 'actions': ['Grow tolerant varieties recommended for your altitude and region.', 'Rotate with a non-host crop and manage infected residue according to local extension advice.', 'Improve field aeration and avoid unnecessary prolonged leaf wetness.'], 'medicine': 'For serious disease, ask a local technician about a maize-labelled fungicide such as azoxystrobin or propiconazole. Product effectiveness depends on timing, disease pressure and local registration; do not use a product unless its Nepal label includes maize.', 'note': "Nepal's National Maize Research Program lists Manakamana and Ganesh varieties as tolerant options for specified hill zones."}, 'Maize___Northern_leaf_blight': {'title': 'Northern Leaf Blight', 'type': 'Fungal disease', 'symptoms': 'Large, elongated, cigar-shaped grey-green or tan lesions that develop along the leaf.', 'actions': ['Choose resistant/tolerant maize varieties suited to your area.', 'Rotate crops and manage infected maize residue to reduce carry-over.', 'Avoid excessive plant crowding and monitor the crop during humid weather.'], 'medicine': 'If disease is progressing and likely to reduce yield, consult an agriculture technician about a maize-labelled fungicide (for example, a product based on azoxystrobin or propiconazole where registered for this use). Follow the exact product label.', 'note': 'The Nepal National Maize Research Program identifies Manakamana-1 as a source population for northern-leaf-blight resistance/tolerance.'}, 'Maize___healthy': {'title': 'Healthy Maize', 'type': 'No disease detected by the model', 'symptoms': 'The model classified the uploaded leaf as healthy.', 'actions': ['Continue regular field scouting.', 'Use balanced fertilizer and suitable irrigation.', 'Keep weeds and crop residues managed, and use quality seed.'], 'medicine': 'No disease medicine is indicated by this prediction. Do not spray pesticides preventively without a diagnosed need.', 'note': 'A healthy prediction does not guarantee that the whole plant or field is disease-free.'}, 'Paddy___Brownspot': {'title': 'Brown Spot', 'type': 'Fungal disease', 'symptoms': 'Small, round-to-oval brown spots on leaves. Severe infections may affect seedlings and grain quality.', 'actions': ['Use clean, healthy seed and locally recommended tolerant varieties.', 'Maintain balanced crop nutrition, especially avoiding nutrient stress.', 'Remove or manage infected crop residues and weeds; maintain good field hygiene.'], 'medicine': 'For confirmed, significant disease, ask a rice specialist about a paddy-labelled fungicide. Products containing mancozeb or other rice-labelled fungicides may be considered only where the Nepal label specifically permits brown spot control. Do not copy rates from another country.', 'note': 'Brown spot can be associated with plant stress and poor crop nutrition. Confirm the diagnosis before spraying.'}, 'Paddy___Hispa': {'title': 'Rice Hispa', 'type': 'Insect pest (not a fungal/bacterial disease)', 'symptoms': 'Adult beetles scrape leaf surfaces, leaving white streaks; larvae mine inside leaves and cause whitish blotches.', 'actions': ['Inspect the field for adult beetles, damaged leaves and leaf mines.', 'Where practical, collect and destroy adults and badly infested leaves.', 'Keep field borders and weeds managed, and avoid excessive nitrogen that produces overly lush growth.', 'Seek local advice if infestation is spreading across the field.'], 'medicine': 'This is an insect pest, so fungicides will not work. If damage exceeds the local action threshold, ask an agriculture technician to identify a currently registered rice-hispa insecticide in Nepal. Do not choose a pesticide based only on this image result.', 'note': "The model's label is 'Hispa'; field confirmation is especially important before any insecticide application."}, 'Paddy___Leafblast': {'title': 'Rice Leaf Blast', 'type': 'Fungal disease', 'symptoms': 'Spindle- or diamond-shaped lesions with grey or whitish centres and darker brown margins. Neck or panicle infection can be especially damaging.', 'actions': ['Use resistant/tolerant rice varieties recommended locally and healthy seed.', 'Avoid excessive nitrogen; apply fertilizer in balanced, recommended splits.', 'Maintain suitable water management, spacing and field sanitation.', 'Monitor closely around tillering, booting and heading stages.'], 'medicine': 'For confirmed blast, rice guidance commonly includes tricyclazole-based products; propiconazole or other locally approved rice fungicides may also be options. Use only a product registered in Nepal for rice blast, at the label rate and growth stage. Rotate modes of action as advised.', 'note': 'ICAR and Sri Lanka Department of Agriculture guidance support integrated management and label-based fungicide use; local Nepal advice takes priority.'}, 'Paddy___healthy': {'title': 'Healthy Paddy', 'type': 'No disease detected by the model', 'symptoms': 'The model classified the uploaded leaf as healthy.', 'actions': ['Continue routine scouting for leaf spots, blast lesions and insect damage.', 'Use clean seed, balanced fertilizer and recommended water management.', 'Maintain field sanitation and monitor crop growth stages.'], 'medicine': 'No disease medicine is indicated by this prediction.', 'note': 'A healthy leaf image does not confirm that the entire field is pest- or disease-free.'}, 'Potato___Early_blight': {'title': 'Early Blight', 'type': 'Fungal disease', 'symptoms': 'Brown, often circular lesions with concentric rings (target-like pattern), usually beginning on older leaves.', 'actions': ['Remove badly affected lower leaves where practical and dispose of infected plant material appropriately.', 'Maintain balanced nutrition and irrigation to reduce plant stress.', 'Use crop rotation and healthy seed tubers; avoid prolonged leaf wetness.'], 'medicine': 'When disease is confirmed and likely to cause loss, a potato-labelled protectant/systemic fungicide may be needed. Active ingredients used in potato early-blight programmes include azoxystrobin, mancozeb or chlorothalonil in some jurisdictions; verify the product is currently registered in Nepal for potato early blight before use.', 'note': "Fungicides protect new growth; existing lesions will not disappear. Follow the label's spray interval, pre-harvest interval and resistance-management instructions."}, 'Potato___Late_blight': {'title': 'Late Blight', 'type': 'Oomycete disease (water mould)', 'symptoms': 'Irregular water-soaked dark lesions that expand rapidly in cool, humid conditions; white growth may appear on leaf undersides in wet weather.', 'actions': ['Treat suspected late blight as urgent because it can spread rapidly.', 'Remove and safely destroy severely infected plants and volunteer potatoes; do not leave infected tubers or cull piles exposed.', 'Use certified healthy seed tubers and resistant/tolerant varieties where available.', 'Avoid overhead irrigation and promote rapid drying of foliage.'], 'medicine': 'A locally registered potato late-blight fungicide may be needed promptly. Ask an agriculture technician for a Nepal-registered product and rotate fungicide groups; do not rely on one active ingredient repeatedly. Some products used internationally include protectant or oomycete-specific mixtures, but registration and resistance status vary.', 'note': 'Late blight is not the same as early blight. Get urgent local confirmation if lesions are spreading quickly or stems/tubers are affected.'}, 'Potato___healthy': {'title': 'Healthy Potato', 'type': 'No disease detected by the model', 'symptoms': 'The model classified the uploaded leaf as healthy.', 'actions': ['Continue scouting, especially during cool and wet weather.', 'Use certified seed tubers and remove volunteer potato plants.', 'Maintain good drainage and avoid prolonged leaf wetness.'], 'medicine': 'No disease medicine is indicated by this prediction.', 'note': 'Continue checking stems and tubers; this model analyses only the uploaded image.'}, 'Tomato___Bacterial_spot': {'title': 'Bacterial Spot', 'type': 'Bacterial disease', 'symptoms': 'Small dark, sometimes water-soaked spots on leaves; fruit may develop raised, scabby spots.', 'actions': ['Use disease-free seed and transplants.', 'Avoid overhead watering and handling plants when foliage is wet.', 'Remove infected crop debris and rotate away from tomato/pepper where possible.', 'Keep plants spaced to improve airflow.'], 'medicine': 'There is no curative spray for infected tissue. Copper-based bactericides can provide partial, preventive protection in some situations; copper resistance can occur. Use only a Nepal-registered product labelled for tomato bacterial spot, and ask a technician whether it is appropriate locally.', 'note': 'Copper products are protectants, not a cure for existing lesions.'}, 'Tomato___Early_blight': {'title': 'Tomato Early Blight', 'type': 'Fungal disease', 'symptoms': 'Dark brown spots, often with concentric rings, commonly starting on older lower leaves with surrounding yellowing.', 'actions': ['Remove infected lower leaves and crop debris.', 'Mulch to reduce soil splash; water at the base rather than wetting foliage.', 'Provide adequate spacing and rotate crops where possible.', 'Keep plants healthy with balanced irrigation and nutrition.'], 'medicine': 'For significant disease, a tomato-labelled fungicide may protect new foliage. Active ingredients used for fungal leaf-spot management include mancozeb, chlorothalonil or copper in some regions; verify Nepal registration and tomato/early-blight label directions before use.', 'note': 'Fungicides do not remove existing spots; they help protect healthy tissue.'}, 'Tomato___Late_blight': {'title': 'Tomato Late Blight', 'type': 'Oomycete disease (water mould)', 'symptoms': 'Irregular water-soaked lesions that turn brown or black and can spread quickly; white growth may appear underneath leaves in humid weather.', 'actions': ['Act quickly and seek local confirmation; the disease can spread rapidly.', 'Remove and safely dispose of infected plant material; do not compost diseased plants in a cool home compost pile.', 'Avoid overhead watering and improve airflow.', 'Keep volunteer potatoes and infected crop debris under control.'], 'medicine': 'A tomato-labelled late-blight fungicide may be required. Ask a local plant-protection technician for a Nepal-registered product suitable for oomycetes and rotate modes of action. Do not use an early-blight fungicide automatically for late blight.', 'note': 'Late blight can affect both tomato and potato and may threaten nearby crops.'}, 'Tomato___Leaf_mold': {'title': 'Leaf Mold', 'type': 'Fungal disease', 'symptoms': 'Pale green or yellow patches on the upper leaf surface, often with olive-green to brown velvety growth underneath.', 'actions': ['Reduce humidity and improve ventilation, especially in greenhouses.', 'Avoid wetting leaves; water at the soil level.', 'Remove affected leaves and plant debris; disinfect tools between plants.', 'Use resistant varieties where available.'], 'medicine': "If disease persists, ask a technician about a fungicide labelled for tomato leaf mold in Nepal. A product's active ingredient must be confirmed against the current local label; do not assume a general leaf-spot fungicide will control leaf mold.", 'note': 'Humidity management and resistant varieties are important parts of control.'}, 'Tomato___Mosaic_virus': {'title': 'Tomato Mosaic Virus', 'type': 'Viral disease', 'symptoms': 'Mottled light- and dark-green leaves, sometimes with distortion, narrow leaves or uneven fruit development.', 'actions': ['There is no medicine that cures a virus-infected plant.', 'Remove and safely dispose of confirmed infected plants to reduce spread.', 'Wash hands and clean tools after handling suspect plants.', 'Use certified virus-free seed/transplants and resistant varieties where available.', 'Avoid handling healthy plants immediately after diseased plants.'], 'medicine': 'No fungicide or antibiotic cures tomato mosaic virus. Focus on sanitation, clean planting material and preventing mechanical spread. Ask an extension worker to confirm the virus because nutrient issues and herbicide injury can look similar.', 'note': "Do not spend money on a 'virus cure' pesticide; it cannot eliminate virus already inside the plant."}, 'Tomato___Septoria_leaf_spot': {'title': 'Septoria Leaf Spot', 'type': 'Fungal disease', 'symptoms': 'Many small circular spots with pale grey centres and dark borders, often starting on lower leaves; tiny dark dots may be visible in spots.', 'actions': ['Remove infected lower leaves and crop debris.', 'Mulch to reduce soil splash and water at the base of plants.', 'Increase spacing and airflow; avoid working among wet plants.', 'Rotate crops and use healthy transplants.'], 'medicine': 'For severe or recurring disease, a tomato-labelled protectant fungicide containing fixed copper or another locally approved leaf-spot active ingredient may slow spread. Verify the current Nepal label; fungicides protect healthy foliage rather than curing old spots.', 'note': 'University extension guidance emphasizes sanitation, spacing, mulch and avoiding wet foliage.'}, 'Tomato___Spider_mites': {'title': 'Two-Spotted Spider Mites', 'type': 'Mite pest (not a fungal disease)', 'symptoms': 'Fine pale stippling or bronzing on leaves, sometimes with delicate webbing underneath. Leaves may dry in heavy infestations.', 'actions': ['Check leaf undersides with a hand lens for mites and eggs.', 'Wash small infestations from leaf undersides with a firm stream of water.', 'Remove heavily infested leaves and reduce dust and plant stress.', 'Avoid unnecessary broad-spectrum insecticides that can kill beneficial mite predators.'], 'medicine': 'If needed, use a miticide specifically labelled for spider mites on tomato in Nepal. Insecticidal soap or horticultural oil may help when applied thoroughly, but can injure plants in heat. Abamectin is an acaricide active ingredient in some registered products; use it only if the exact Nepal product label lists tomato and mites. Do not use ordinary insecticide without checking its mite activity.', 'note': 'Spider mites are arachnids, not insects; many general insecticides do not control them.'}, 'Tomato___Target_spot': {'title': 'Target Spot', 'type': 'Fungal disease', 'symptoms': 'Brown circular lesions with concentric rings; spots can occur on leaves, stems and fruit and may resemble early blight.', 'actions': ['Remove badly affected leaves and crop debris.', 'Improve spacing and airflow and avoid overhead irrigation.', 'Use crop rotation and clean transplants.', 'Confirm the diagnosis because target spot resembles other tomato leaf-spot diseases.'], 'medicine': 'If confirmed, ask a plant-protection technician for a fungicide labelled for tomato target spot in Nepal. Do not assume an early-blight treatment will work; fungicide sensitivity varies and resistance can develop.', 'note': 'A field diagnosis is recommended before chemical treatment because several tomato leaf spots look alike.'}, 'Tomato___Yellow_leaf_curl_virus': {'title': 'Tomato Yellow Leaf Curl Virus', 'type': 'Viral disease (spread by whiteflies)', 'symptoms': 'Leaves curl upward, become smaller and yellow between veins; plants may be stunted and set fewer fruits.', 'actions': ['There is no medicine that cures an infected plant.', 'Remove and safely dispose of confirmed infected plants early.', 'Use resistant varieties and healthy transplants where available.', 'Use insect-proof nursery netting and monitor whiteflies; remove nearby weed hosts.', 'Use reflective mulch or other locally recommended whitefly-management measures where practical.'], 'medicine': 'No fungicide cures this virus. Management focuses on preventing whitefly transmission. If whiteflies need control, ask for a Nepal-registered tomato/whitefly product and use an integrated approach that protects beneficial insects and follows label directions.', 'note': 'Killing whiteflies after a plant is infected will not cure that plant.'}, 'Tomato___healthy': {'title': 'Healthy Tomato', 'type': 'No disease detected by the model', 'symptoms': 'The model classified the uploaded leaf as healthy.', 'actions': ['Continue checking leaves, stems and fruit regularly.', 'Water at the base of the plant and maintain good spacing.', 'Use clean transplants and remove old crop debris.'], 'medicine': 'No disease medicine is indicated by this prediction. Avoid unnecessary pesticide spraying.', 'note': 'A healthy leaf prediction does not prove the whole plant or field is disease-free.'}}

# ----------------------- MODEL -----------------------

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.pt is missing. Keep model.pt in the same folder as app.py."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    classes = checkpoint["classes"]

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=len(classes),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()
    return model, classes


@lru_cache(maxsize=1)
def get_model():
    return load_model()


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict(image, model, classes, detected_crop):
    tensor = get_transform()(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)[0]

        crop_indices = [
            i for i, class_name in enumerate(classes)
            if class_name.startswith(f"{detected_crop}___")
        ]
        if not crop_indices:
            raise ValueError(
                f"The trained model has no classes for {detected_crop}."
            )

        crop_logits = logits[crop_indices]
        probabilities = torch.softmax(crop_logits, dim=0)
        top_values, local_indices = torch.topk(
            probabilities, k=min(3, len(crop_indices))
        )
        top_indices = [crop_indices[int(i)] for i in local_indices.cpu()]

    return [
        (classes[index], float(probability))
        for index, probability in zip(top_indices, top_values.cpu())
    ]


def readable_label(label):
    if "___" not in label:
        return label.replace("_", " ")
    crop, disease = label.split("___", 1)
    title = GUIDES.get(label, {}).get("title", disease.replace("_", " "))
    return f"{crop} — {title}"


# ---------------- IMAGE SCREENING -----------------------

@lru_cache(maxsize=1)
def load_image_validator():
    return pipeline(
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        device=0 if torch.cuda.is_available() else -1,
    )


def check_image_quality(image):
    width, height = image.size

    if width < 224 or height < 224:
        return False, "The photo is too small. Move closer to the leaf and take another photo."

    array = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

    brightness = float(np.mean(gray))
    if brightness < 25:
        return False, "The photo is too dark. Take another photo in brighter, even light."
    if brightness > 245:
        return False, "The photo is overexposed. Avoid direct glare and try again."

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < 25:
        return False, "The photo appears blurry. Hold your phone steady and photograph the leaf again."

    return True, "Photo size, brightness and basic sharpness checks passed."


def validate_crop_leaf(image):
    labels = [
        "a close-up field photograph of a maize (corn) leaf",
        "a close-up field photograph of a paddy (rice) leaf",
        "a close-up field photograph of a potato leaf",
        "a close-up field photograph of a tomato leaf",
        "a leaf from a different, unsupported plant",
        "a person, face, animal, building, vehicle, food or other non-leaf object",
        "soil or an empty field with no crop leaf",
        "a screenshot, drawing, diagram or text document",
    ]

    label_to_crop = {
        labels[0]: "Maize",
        labels[1]: "Paddy",
        labels[2]: "Potato",
        labels[3]: "Tomato",
    }

    results = load_image_validator()(image, candidate_labels=labels)
    best = results[0]
    best_label = best["label"]
    best_score = float(best["score"])

    # This is a preliminary screening threshold, not a calibrated probability.
    if best_label not in label_to_crop or best_score < 0.28:
        return (
            False,
            None,
            "This does not look like a supported crop leaf. Please photograph one "
            "maize, paddy, potato or tomato leaf clearly.",
        )

    crop = label_to_crop[best_label]
    return True, crop, f"Possible {crop} leaf detected (CLIP screening score: {best_score:.2f})."


# ---------------- RESULTS -----------------------

def guide_markdown(label, confidence):
    guide = GUIDES.get(label)
    if not guide:
        return (
            "## Guidance not available yet\n"
            "A detailed guide has not been added for this model class. "
            "Please ask a local agriculture technician before taking action."
        )

    crop = label.split("___", 1)[0]
    lines = [
        f"## {guide.get('title', readable_label(label))}",
        f"**Crop:** {crop}  ·  **Category:** {guide.get('type', 'Not listed')}",
        f"**Model confidence within this crop's classes:** {confidence * 100:.1f}%",
        "",
        "### 🔎 Common symptoms",
        guide.get("symptoms", "No symptom description is available."),
        "",
        "### 🌿 What you can do",
    ]
    lines.extend(f"- {item}" for item in guide.get("actions", []))
    lines.extend([
        "",
        "### 🧴 Treatment information",
        guide.get("medicine", "No treatment information is available."),
        "",
        "### Important note",
        guide.get("note", ""),
    ])
    return "\n".join(lines)


def analyze_leaf(image):
    empty = ("", "", "", "", "")

    if image is None:
        raise gr.Error("Please upload or capture a leaf photo first.")

    try:
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        raise gr.Error("The image could not be read. Please try a JPG or PNG photo.")

    quality_ok, quality_message = check_image_quality(image)
    if not quality_ok:
        return (
            image,
            f"## 1 · Photo quality\n❌ {quality_message}",
            "",
            "",
            "",
            "",
        )

    try:
        valid, crop, crop_message = validate_crop_leaf(image)
    except Exception as exc:
        return (
            image,
            "## 1 · Photo quality\n"
            f"✅ {quality_message}\n\n"
            "## 2 · Crop-leaf screening\n"
            "❌ The CLIP screening model could not run. Check your internet connection "
            "for its first download and try again.\n\n"
            f"`{html.escape(str(exc))}`",
            "",
            "",
            "",
            "",
        )

    if not valid:
        return (
            image,
            f"## 1 · Photo quality\n✅ {quality_message}\n\n"
            f"## 2 · Crop-leaf screening\n❌ {crop_message}",
            "",
            "",
            "",
            "",
        )

    try:
        model, classes = get_model()
        predictions = predict(image, model, classes, crop)
    except Exception as exc:
        return (
            image,
            f"## 1 · Photo quality\n✅ {quality_message}\n\n"
            f"## 2 · Crop-leaf screening\n✅ {crop_message}\n\n"
            "## 3 · Disease model\n❌ Prediction could not be completed.\n\n"
            f"`{html.escape(str(exc))}`",
            "",
            "",
            "",
            "",
        )

    if not predictions:
        return (
            image,
            f"## 1 · Photo quality\n✅ {quality_message}\n\n"
            f"## 2 · Crop-leaf screening\n✅ {crop_message}",
            "",
            "",
            "No trained disease classes were found for this crop.",
            "",
        )

    best_label, confidence = predictions[0]
    interpretation = (
        "Strong model match" if confidence >= 0.75
        else "Moderate model match" if confidence >= 0.55
        else "Weak model match"
    )

    status = (
        f"## 1 · Photo quality\n✅ {quality_message}\n\n"
        f"## 2 · Crop-leaf screening\n✅ {crop_message}\n\n"
        f"## 3 · Disease model\n✅ Analysis completed for **{crop}**."
    )

    if confidence < 0.55:
        status += (
            "\n\n⚠️ The model match is weak. Retake the photo if possible and "
            "confirm visible symptoms with an agriculture technician."
        )
    elif confidence < 0.75:
        status += (
            "\n\nℹ️ The model match is moderate. Confirm the symptoms before "
            "taking any treatment action."
        )
    else:
        status += (
            "\n\n✅ The model found a strong match among this crop's trained classes. "
            "This is still an AI screening result, not a confirmed diagnosis."
        )

    prediction_md = (
        f"# {readable_label(best_label)}\n\n"
        f"**{confidence * 100:.1f}%** · {interpretation}\n\n"
        f"<small>Confidence is calculated only across the detected crop's model classes.</small>"
    )

    other_md = "\n".join(
        f"- **{readable_label(label)}** — {score * 100:.1f}%"
        for label, score in predictions[1:]
    ) or "No other candidate classes."

    guide_md = guide_markdown(best_label, confidence)
    safety_md = (
        "⚠️ **Safety reminder:** Crop Doctor provides educational guidance, not a "
        "confirmed diagnosis or pesticide prescription. Before using any chemical, "
        "check that the exact product is currently registered in Nepal for this crop "
        "and target pest/disease. Follow its label for dose, protective equipment, "
        "re-entry interval and pre-harvest interval. Ask a local agriculture technician "
        "when uncertain. Never spray solely because of an AI prediction."
    )

    return image, status, prediction_md, other_md, guide_md, safety_md


# ---------------- VISUAL DESIGN -----------------------

CSS = """
:root {
    --leaf: #176b45;
    --leaf-dark: #104d35;
    --mint: #eaf6ed;
    --cream: #fbfaf4;
    --ink: #19352a;
}
.gradio-container {
    max-width: 1220px !important;
    margin: 0 auto !important;
    background: linear-gradient(145deg, #f2f8ef 0%, #fbfaf4 55%, #edf7ef 100%) !important;
}
body, .gradio-container { color: var(--ink) !important; }
.section-card, .section-card *,
#hero, #hero *,
#safety, #safety *,
.gradio-container .markdown, .gradio-container .markdown * {
    color: #000000 !important;
}
#hero {
    border-radius: 26px;
    padding: 34px 38px;
    color: white;
    background: linear-gradient(115deg, #104d35 0%, #176b45 55%, #54a66a 100%);
    box-shadow: 0 14px 34px rgba(16, 77, 53, .18);
    margin-bottom: 18px;
}
#hero, #hero * { color: white !important; }
#hero .crop-pill,
#hero .crop-pill * {
    color: #000000 !important;
}
#hero h1 { font-size: clamp(2rem, 4vw, 3.2rem); margin: 0 0 8px; }
#hero p { color: #e7f7e9 !important; font-size: 1.05rem; margin: 0; }
#tips-accordion,
#tips-accordion *,
#tips-accordion summary,
#tips-accordion summary *,
#tips-accordion .label,
#tips-accordion .label *,
#tips-accordion .gradio-accordion,
#tips-accordion .gradio-accordion * {
    color: white !important;
}
#tips-accordion summary svg,
#tips-accordion summary path,
#tips-accordion summary polyline,
#tips-accordion summary line,
#tips-accordion summary circle,
#tips-accordion summary .icon,
#tips-accordion .gradio-accordion svg,
#tips-accordion .gradio-accordion path,
#tips-accordion .gradio-accordion polyline,
#tips-accordion .gradio-accordion line,
#tips-accordion .gradio-accordion circle {
    stroke: white !important;
    fill: white !important;
    color: white !important;
}
.hero-tag {
    display: inline-block; padding: 6px 12px; border-radius: 99px;
    background: rgba(255,255,255,.15); color: #fff; font-size: .83rem;
    margin-bottom: 14px; border: 1px solid rgba(255,255,255,.24);
}
.section-card {
    border: 1px solid #dceade !important;
    border-radius: 20px !important;
    background: rgba(255,255,255,.92) !important;
    box-shadow: 0 7px 22px rgba(25, 74, 46, .055) !important;
    padding: 18px !important;
}
#analyze-btn {
    background: linear-gradient(100deg, #176b45, #2c945d) !important;
    color: white !important;
    border: 0 !important;
    border-radius: 13px !important;
    min-height: 52px;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
}
#analyze-btn:hover { filter: brightness(1.06); }
#result-title { color: #176b45; }
#safety {
    border-left: 5px solid #d69a2d;
    background: #fff8e8;
    border-radius: 12px;
    padding: 14px 18px;
}
.crop-pill {
    display: inline-block; background: #e5f3e8; color: #000000 !important;
    border-radius: 99px; padding: 7px 12px; margin: 4px;
    font-weight: 600; font-size: .88rem;
}
.crop-pill span,
.crop-pill * {
    color: #000000 !important;
}
[data-testid="image"],
[data-testid="image"] *,
.gradio-image, .gradio-image *,
.upload, .upload *,
.dropzone, .dropzone * {
    color: #ffffff !important;
}
[data-testid="image"] svg,
[data-testid="image"] path,
[data-testid="image"] polyline,
[data-testid="image"] circle,
[data-testid="image"] line,
[data-testid="image"] rect,
.gradio-image svg, .gradio-image path,
.gradio-image polyline, .gradio-image circle,
.gradio-image line, .gradio-image rect {
    stroke: #ffffff !important;
    fill: #ffffff !important;
    color: #ffffff !important;
}
footer { display: none !important; }
"""

with gr.Blocks(
    title="Crop Doctor | Crop Disease Screening",
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="lime",
        neutral_hue="slate",
        radius_size="lg",
    ),
    css=CSS,
) as demo:

    gr.HTML("""
    <div id="hero">
      <div class="hero-tag">🌿 AI-ASSISTED CROP HEALTH</div>
      <h1>Crop Doctor</h1>
      <p>A simple crop-leaf screening assistant for farmers.</p>
      <div style="margin-top:18px">
        <span class="crop-pill">🌽 Maize</span>
        <span class="crop-pill">🌾 Paddy</span>
        <span class="crop-pill">🥔 Potato</span>
        <span class="crop-pill">🍅 Tomato</span>
      </div>
    </div>
    """)

    gr.Markdown(
        "Take a clear photo of **one crop leaf**. Crop Doctor checks the photo first, "
        "then analyses it with your trained disease model."
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["section-card"]):
            gr.Markdown("## 📷 1. Add a leaf photo")
            gr.Markdown(
                "Choose an existing image or switch to the camera option on your phone."
            )
            image_input = gr.Image(
                type="pil",
                sources=["upload", "webcam"],
                label="Upload or capture leaf photo",
                height=360,
            )
            with gr.Accordion("📸 Tips for a useful photo", open=False, elem_id="tips-accordion"):
                gr.Markdown(
                    "- Photograph one leaf at a time, close enough to see the symptoms.\n"
                    "- Use natural, even light and keep the leaf in focus.\n"
                    "- Avoid screenshots, filters, collages and heavy glare.\n"
                    "- If possible, photograph the upper and lower sides separately.\n"
                    "- A leaf photo cannot show every problem affecting roots, stems or fruit."
                )
            analyze_btn = gr.Button(
                "🌱  Check leaf and detect disease",
                variant="primary",
                elem_id="analyze-btn",
            )

        with gr.Column(scale=5, elem_classes=["section-card"]):
            gr.Markdown("## 🧭 2. Screening result")
            preview_image = gr.Image(
                label="Your submitted photo",
                type="pil",
                height=260,
                interactive=False,
            )
            status_box = gr.Markdown(
                "Your photo checks and analysis steps will appear here."
            )

    gr.Markdown("---")
    gr.Markdown("## 🩺 3. Disease analysis and guidance")

    with gr.Row(equal_height=False):
        with gr.Column(scale=6, elem_classes=["section-card"]):
            prediction_box = gr.Markdown(
                "### Waiting for a photo\nYour main prediction will appear here."
            )
        with gr.Column(scale=4, elem_classes=["section-card"]):
            gr.Markdown("### Other possible results")
            other_classes_box = gr.Markdown("No analysis yet.")

    with gr.Column(elem_classes=["section-card"]):
        guide_box = gr.Markdown(
            "Disease symptoms and practical guidance will appear after a valid photo is analysed."
        )

    with gr.Column(elem_classes=["section-card"]):
        safety_box = gr.Markdown(
            "⚠️ Always confirm uncertain results with a local agriculture technician."
        )

    with gr.Accordion("📚 Information sources and project notes", open=False):
        gr.Markdown(
            "- [Nepal Plant Quarantine and Pesticide Management Center](https://timure.npponepal.gov.np/en-publication-pesticide-management-and-regulation)\n"
            "- [Nepal National Maize Research Program](https://nmrp.gov.np/technology/pathology)\n"
            "- [CABI PlantwisePlus Nepal](https://tool.cabi.org/plantwiseplus_nepal)\n\n"
            "The disease guidance is carried over from the supplied project. "
            "Chemical registration and permitted crop uses must be checked against "
            "current Nepal requirements and the exact product label."
        )

    gr.Markdown(
        "<div style='text-align:center; padding:18px; color:#648071'>"
        "Crop Doctor · AI decision-support prototype · Not a substitute for field diagnosis"
        "</div>"
    )

    analyze_btn.click(
        fn=analyze_leaf,
        inputs=[image_input],
        outputs=[
            preview_image,
            status_box,
            prediction_box,
            other_classes_box,
            guide_box,
            safety_box,
        ],
    )

if __name__ == "__main__":
    demo.launch()
