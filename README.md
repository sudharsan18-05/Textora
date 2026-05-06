🖼️ Textora
AI-Powered Image Text Editor
A full-stack web app that detects, edits, removes, and replaces text in any image using AI OCR, intelligent background reconstruction, and Google Fonts rendering — all in the browser.
---
🧠 Overview
Textora is an intelligent image editing platform that:
- Detects all text in any uploaded image automatically
- Lets you click any detected word or phrase to edit it
- Removes text and reconstructs the original background seamlessly
- Re-renders new text with Google Fonts, matched to the original color
- Saves all your projects with full user authentication
---
❗ Problem Statement
Traditional image editors require:
- Manual selection of every text area
- Design skills to reconstruct backgrounds
- Expensive software (Photoshop, Illustrator)
- Technical knowledge to match fonts and colors
👉 This creates a high barrier for non-designers who need quick text edits in images.
---
💡 Solution
Textora introduces an AI-assisted editing pipeline that combines:
- OCR text detection (automatic bounding boxes)
- Coons-patch background reconstruction (no blurs, no solid blocks)
- Pillow font rendering with auto-detected color
- One-click workflow — detect → click → type → export
👉 Result: Professional image text edits in seconds, no design skills needed.
---
⚙️ System Architecture

User Uploads Image
↓
Cloudinary CDN (Image Storage)
↓
OCR.space API (Text Detection)
↓
Flask Backend
├─ Bounding Box Parser
├─ Background Reconstruction Engine (Coons Patch)
└─ Pillow Text Renderer (Google Fonts)
↓
Browser Editor (Vanilla JS)
├─ SVG Overlay (colored bounding boxes)
├─ Click-to-Edit Panel
├─ Drag / Reposition
└─ Undo / Redo Stack
↓
Export — Full Resolution Download

---
🔍 How the Editor Works
**OCR Detection**
Scans the uploaded image for all text regions using OCR.space API:
- URL-based scan (1–3 seconds) — preferred
- Base64 fallback for unsupported formats
**Text Editing**
- Click any bounding box → type new text
- Choose font, size, bold
- Auto-detects the original text color
- Apply all edits at once on the original clean image
**Text Removal**
- Select a region → click Remove
- Backend reconstructs background using Coons-patch algorithm
- Multi-pass fill for large regions (4 iterative passes, each shrinking inward)
- Result: smooth gradient fill, no visible seam
---
🎨 Background Reconstruction Engine
The Coons-patch algorithm fills removed text areas with a natural-looking gradient:

Sample pixel strips → Top / Bottom / Left / Right of bounding box
↓
Exponential-decay weighting (innermost pixels weighted highest)
↓
Bilinear interpolation across the interior
↓
Cosine edge feathering (eliminates hard rectangular seam)
↓
Multi-pass for large regions (>50px tall or >150px wide)

Region Size   → Strategy
Small (≤50px) → Single-pass Coons patch
Large (>50px) → 4-pass iterative fill (outer ring → inner ring)
---
📦 Features
- ✔ AI text detection with colored bounding box overlays
- ✔ Click-to-edit any detected text region
- ✔ Smart background reconstruction (Coons-patch, no blur)
- ✔ Google Fonts rendering — 7 fonts bundled (Lato, Ubuntu, Arvo, Tinos, PT Sans, Anton, Bebas Neue)
- ✔ Auto color detection — matches original text color
- ✔ Drag-to-reposition text boxes
- ✔ Undo / Redo full edit history
- ✔ Full resolution export
- ✔ User signup, login, and dashboard
- ✔ Auto-save for logged-in users
- ✔ Guest mode (3 free exports, no signup required)
- ✔ Cloudinary CDN image storage
---
🛠️ Tech Stack
| Layer            | Technology                        |
|------------------|-----------------------------------|
| Backend          | Python 3.11, Flask                |
| Database         | SQLite (WAL mode)                 |
| Image Processing | Pillow, NumPy, OpenCV             |
| OCR Engine       | OCR.space API                     |
| Image Storage    | Cloudinary CDN                    |
| Fonts            | Google Fonts (bundled .ttf files) |
| Frontend         | Vanilla JavaScript, HTML5, CSS3   |
| Auth             | Werkzeug password hashing, Flask sessions |
---
✅ Prerequisites
- [Python 3.11+](https://www.python.org/downloads/)
- pip (comes with Python)
- Free [Cloudinary](https://cloudinary.com) account
- Free [OCR.space](https://ocr.space/ocrapi) API key
---
🚀 How to Run the Project
**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/textora.git
cd textora

2. Create a virtual environment

python -m venv venv
# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Create your .env file

Create a file named .env in the same folder as app.py:

CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
OCRSPACE_API_KEY=your_ocrspace_key
SECRET_KEY=any_long_random_string

5. Run the app

python app.py

6. Open in your browser

http://localhost:5000

🔐 Environment Variables

Variable	Where to Get It
CLOUDINARY_CLOUD_NAME	cloudinary.com → Dashboard
CLOUDINARY_API_KEY	cloudinary.com → Dashboard
CLOUDINARY_API_SECRET	cloudinary.com → Dashboard
OCRSPACE_API_KEY	ocr.space/ocrapi → Free API Key
SECRET_KEY	Run: python -c "import secrets; print(secrets.token_hex(32))"
🖥️ Usage

Open http://localhost:5000
Click Try as Guest or Sign Up for a free account
Upload any image (JPG, PNG, WEBP)
Click Scan — AI detects all text and draws colored boxes
Click any box to select it → type new text in the right panel
Pick a font, adjust size → click Apply All to render
To erase text → select a box → click Remove
Click Export to download your final image
📡 API Endpoints

Endpoint	Method	Description
/	GET	Landing page
/editor	GET	Main editor
/detect	POST	Upload image → OCR → return regions
/apply-all	POST	Apply all edits to original image
/export	POST	Download final edited image
/dashboard	GET	User's saved projects
/login	GET/POST	Login
/signup	GET/POST	Signup
/logout	GET	Logout
🏆 Final Outcome

The app delivers:

✔ Automatic AI text detection in any image
✔ Seamless background reconstruction (no artifacts, no blur)
✔ Professional font rendering with color matching
✔ Full project management with auth and dashboard

👉 Transforms a complex multi-step Photoshop workflow into a single-page browser experience anyone can use.

📄 License

MIT License — free to use, modify, and distribute.
