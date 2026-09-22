import os
import json
import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from google import genai
from google.genai import types


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Create a .env file inside the backend folder."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="GITAM Attendance OCR API",
    description="AI-powered GITAM attendance screenshot reader",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://maheedhar56.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HOME / HEALTH CHECK
# ============================================================

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "GITAM Attendance OCR API is working"
    }


# ============================================================
# ATTENDANCE SCREENSHOT EXTRACTION
# ============================================================

@app.post("/extract-attendance")
async def extract_attendance(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # CHECK FILE TYPE
    # --------------------------------------------------------

    allowed_types = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp"
    ]

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid file type. "
                "Please upload JPG, JPEG, PNG or WEBP."
            )
        )


    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )


    # --------------------------------------------------------
    # PROMPT FOR GEMINI
    # --------------------------------------------------------

    prompt = """
You are an advanced attendance table extraction system.

Analyze the uploaded screenshot carefully.

The screenshot contains a college attendance table.

The table may be displayed on:

- Mobile phone
- Android phone
- iPhone
- Tablet
- Laptop
- Desktop
- Different screen resolutions
- Different browser zoom levels

The table normally contains columns similar to:

Subject Name | Present | Total | Percentage

Your task is to find EVERY attendance subject row.

For every subject, extract ONLY:

1. Complete subject name
2. Present classes
3. Total classes

IMPORTANT RULES:

1. Ignore the percentage shown in the screenshot.
2. We will calculate the percentage ourselves.
3. The first attendance number is PRESENT.
4. The second attendance number is TOTAL.
5. Present must never be greater than Total.
6. Include normal subjects.
7. Include laboratory subjects.
8. Include subjects with long names.
9. Include CRT if present.
10. Preserve the complete subject name.
11. Do not merge two different subjects.
12. Do not skip rows.
13. Ignore table headers.
14. Ignore unrelated text outside the attendance table.
15. Do not invent subjects.
16. Do not invent numbers.
17. If a row is unclear, carefully inspect the image again.
18. Return every readable attendance row.

Example:

If the screenshot contains:

Computer Networks    19    29    65

Return:

{
    "name": "Computer Networks",
    "present": 19,
    "total": 29
}

If it contains:

Artificial Intelligence Lab    11    18    61

Return:

{
    "name": "Artificial Intelligence Lab",
    "present": 11,
    "total": 18
}

The percentage column MUST NOT be used as Total.

Return only the structured JSON requested by the schema.
"""


    # --------------------------------------------------------
    # STRUCTURED OUTPUT SCHEMA
    # --------------------------------------------------------

    response_schema = {
        "type": "object",
        "properties": {
            "subjects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string"
                        },
                        "present": {
                            "type": "integer"
                        },
                        "total": {
                            "type": "integer"
                        }
                    },
                    "required": [
                        "name",
                        "present",
                        "total"
                    ]
                }
            }
        },
        "required": [
            "subjects"
        ]
    }


    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models_to_try = [
        "gemini-3.6-flash",
        "gemini-3.5-flash"
    ]


    response = None
    last_error = None


    # --------------------------------------------------------
    # GEMINI REQUEST WITH RETRIES
    # --------------------------------------------------------

    for model_name in models_to_try:

        for attempt in range(3):

            try:

                print(
                    f"\nTrying {model_name} "
                    f"(attempt {attempt + 1}/3)..."
                )


                response = client.models.generate_content(

                    model=model_name,

                    contents=[

                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=file.content_type
                        ),

                        prompt
                    ],

                    config=types.GenerateContentConfig(

                        response_mime_type="application/json",

                        response_schema=response_schema
                    )
                )


                print(
                    f"SUCCESS: Gemini responded using "
                    f"{model_name}"
                )

                break


            except Exception as e:

                last_error = e

                error_text = str(e)

                print(
                    f"ERROR with {model_name}: "
                    f"{error_text}"
                )


                # ------------------------------------------------
                # RETRY TEMPORARY 503 ERRORS
                # ------------------------------------------------

                if (
                    "503" in error_text
                    or
                    "UNAVAILABLE" in error_text
                ):

                    wait_time = 2 ** attempt

                    print(
                        f"Gemini temporarily unavailable."
                    )

                    print(
                        f"Waiting {wait_time} seconds "
                        f"before retry..."
                    )

                    time.sleep(wait_time)


                else:

                    # For errors other than 503,
                    # don't retry the same model.

                    break


        # ----------------------------------------------------
        # IF SUCCESSFUL, STOP TRYING OTHER MODELS
        # ----------------------------------------------------

        if response is not None:
            break


    # --------------------------------------------------------
    # ALL MODELS FAILED
    # --------------------------------------------------------

    if response is None:

        raise HTTPException(

            status_code=503,

            detail=(
                "Gemini service is temporarily unavailable. "
                f"Last error: {str(last_error)}"
            )
        )


    # --------------------------------------------------------
    # PARSE GEMINI JSON
    # --------------------------------------------------------

    try:

        data = json.loads(response.text)

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                "Gemini returned invalid JSON. "
                f"Error: {str(e)}"
            )
        )


    # --------------------------------------------------------
    # GET SUBJECTS
    # --------------------------------------------------------

    subjects = data.get(
        "subjects",
        []
    )


    # --------------------------------------------------------
    # VALIDATE SUBJECTS
    # --------------------------------------------------------

    valid_subjects = []


    for subject in subjects:

        # Get subject name

        name = str(
            subject.get(
                "name",
                ""
            )
        ).strip()


        # Get numbers

        try:

            present = int(
                subject.get(
                    "present",
                    0
                )
            )

            total = int(
                subject.get(
                    "total",
                    0
                )
            )

        except (
            ValueError,
            TypeError
        ):

            continue


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not name:
            continue


        if total <= 0:
            continue


        if present < 0:
            continue


        if present > total:
            continue


        # ----------------------------------------------------
        # CALCULATE PERCENTAGE OURSELVES
        # ----------------------------------------------------

        percentage = round(
            (present / total) * 100,
            2
        )


        # ----------------------------------------------------
        # ADD SUBJECT
        # ----------------------------------------------------

        valid_subjects.append(

            {
                "name": name,

                "present": present,

                "total": total,

                "percentage": percentage
            }
        )


    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {

        "success": True,

        "count": len(valid_subjects),

        "subjects": valid_subjects
    }