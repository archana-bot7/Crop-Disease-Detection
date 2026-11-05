# Crop Doctor — Gradio version

## Included
- Maize, paddy, potato and tomato classes from the existing EfficientNet-B0 checkpoint
- Upload photo and webcam capture
- Basic image-size, brightness and blur checks
- CLIP zero-shot image screening for supported crop leaves
- Existing disease model inference, restricted to the screened crop
- Disease prediction, alternative model classes, symptoms and guidance
- Updated green agricultural Gradio interface

## Run on Windows

1. Extract this folder.
2. Open Command Prompt or the VS Code terminal inside this folder.
3. Create and activate a virtual environment:

   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```bat
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If PyTorch installation fails, install a compatible PyTorch and torchvision build
   from https://pytorch.org/get-started/locally/ and then install the remaining requirements.

5. Run:

   ```bat
   python app.py
   ```

6. Open the local Gradio URL printed in the terminal.

## First run
The CLIP screening model `openai/clip-vit-base-patch32` downloads from Hugging Face
on first use. An internet connection is required for that download.

## Phone camera
The webcam/camera input is available in the Gradio image component. For a deployed
website, use HTTPS so mobile browsers can grant camera permission.

## Important limitations
- The supplied trained checkpoint is used as-is; this app does not retrain it.
- CLIP is a general image similarity model, not a dedicated leaf detector. Its score
  is not a calibrated probability. It may reject genuine crop leaves or accept unrelated images.
- The blur and brightness thresholds are preliminary heuristics and need testing on
  photos from different phones and lighting conditions.
- The disease model only predicts among its trained classes. Restricting prediction
  to the screened crop reduces cross-crop errors but does not guarantee a correct diagnosis.
- The disease guides were carried over from the supplied app and have not been
  independently verified as current Nepal pesticide recommendations. Always check
  current registration and the exact product label, and consult an agriculture technician.
- Test the system on labelled real field images and have results reviewed by an agriculture
  specialist before farmers rely on it.
