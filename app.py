import base64
import io
import json
import mimetypes
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import face_recognition
import numpy as np
from PIL import Image
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DATA_DIR = Path(__file__).parent / "data"
BLACKLIST_DIR = DATA_DIR / "blacklist"
METADATA_PATH = DATA_DIR / "metadata.json"
USER_DATA_DIR = DATA_DIR / "users"
DEFAULT_USER_ID = "default"
TOLERANCE = 0.55
MAX_IMAGE_SIZE = 800  # resize longer side for faster/more reliable detection
MANUAL_BOXES = {
    # filename -> (top, right, bottom, left)
    "ulugulu.JPG": (20, 210, 260, 20),
    "zackary.JPG": (30, 230, 250, 30),
}

app = FastAPI(title="Blacklist Face Recognition POC")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

# CORS to allow mobile devices on the same network
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Frame(BaseModel):
    image_base64: str  # data URL (e.g., "data:image/jpeg;base64,...")


class MatchResult(BaseModel):
    status: str
    name: Optional[str] = None
    reason: Optional[str] = None
    file: Optional[str] = None
    distance: Optional[float] = None


class BlacklistEntry(BaseModel):
    id: str
    name: str
    reason: str
    file: str


class BlacklistUpload(BaseModel):
    name: str
    reason: str
    image_base64: str
    filename: Optional[str] = None


user_blacklists_cache: Dict[str, Dict[str, List]] = {}


def resolve_user_paths(user_id: str):
    if user_id == DEFAULT_USER_ID and METADATA_PATH.exists():
        return METADATA_PATH, BLACKLIST_DIR
    user_dir = USER_DATA_DIR / user_id
    return user_dir / "metadata.json", user_dir / "blacklist"


def load_blacklist_for_user(user_id: str):
    metadata_path, blacklist_dir = resolve_user_paths(user_id)
    if not metadata_path.exists():
        return [], []

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    entries: List[BlacklistEntry] = []
    vectors: List = []
    changed = False
    for item in metadata:
        if "id" not in item:
            item["id"] = uuid.uuid4().hex
            changed = True
        entry = BlacklistEntry(**item)
        image_path = blacklist_dir / entry.file
        if not image_path.exists():
            print(f"[WARN] Missing image file: {image_path}")
            continue
        image = face_recognition.load_image_file(image_path)
        max_dim = max(image.shape[0], image.shape[1])
        if max_dim > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / float(max_dim)
            new_h = int(image.shape[0] * scale)
            new_w = int(image.shape[1] * scale)
            image = np.array(Image.fromarray(image).resize((new_w, new_h)))
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            locations = face_recognition.face_locations(image, model="cnn")
            if locations:
                encodings = face_recognition.face_encodings(image, locations)
        if not encodings:
            manual_box = MANUAL_BOXES.get(entry.file)
            if manual_box:
                encodings = face_recognition.face_encodings(image, known_face_locations=[manual_box])
        if not encodings:
            print(f"[WARN] No face found in {image_path}")
            continue
        entries.append(entry)
        vectors.append(encodings[0])

    print(f"[INFO] Loaded {len(entries)} entries for user {user_id}")
    if changed:
        save_metadata_for_user(user_id, entries)
    return entries, vectors


def save_metadata_for_user(user_id: str, entries: List[BlacklistEntry]):
    metadata_path, _ = resolve_user_paths(user_id)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump([e.dict() for e in entries], f, ensure_ascii=False, indent=2)


def ensure_user_blacklist_loaded(user_id: str):
    cache = user_blacklists_cache.get(user_id)
    if cache:
        return cache
    entries, vectors = load_blacklist_for_user(user_id)
    cache = {"entries": entries, "vectors": vectors}
    user_blacklists_cache[user_id] = cache
    return cache


def get_user_id(x_user_id: Optional[str] = Header(None)):
    return x_user_id or DEFAULT_USER_ID


def parse_data_url(data_url: str):
    if "," not in data_url:
        raise ValueError("Invalid data URL")
    header, b64data = data_url.split(",", 1)
    try:
        raw = base64.b64decode(b64data)
    except base64.binascii.Error as exc:
        raise ValueError("Invalid base64 data") from exc
    mime = header.split(";")[0].removeprefix("data:") if header.startswith("data:") else None
    return raw, mime


def decode_image(data_url: str):
    raw, _ = parse_data_url(data_url)
    return face_recognition.load_image_file(io.BytesIO(raw))


def determine_extension(filename: Optional[str], mime_type: Optional[str]) -> str:
    if filename:
        ext = Path(filename).suffix.lower()
        if ext:
            return ext
    if mime_type:
        ext = mimetypes.guess_extension(mime_type)
        if ext:
            return ext
    return ".jpg"


@app.on_event("startup")
def startup_event():
    ensure_user_blacklist_loaded(DEFAULT_USER_ID)


@app.get("/")
def root():
    index_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(index_path)


@app.get("/health")
def health(user_id: str = Depends(get_user_id)):
    cache = ensure_user_blacklist_loaded(user_id)
    return {"status": "ok", "blacklist_size": len(cache["entries"])}


@app.get("/blacklist", response_model=List[BlacklistEntry])
def list_blacklist(user_id: str = Depends(get_user_id)):
    cache = ensure_user_blacklist_loaded(user_id)
    return cache["entries"]


@app.post("/blacklist", response_model=BlacklistEntry)
def add_blacklist(upload: BlacklistUpload, user_id: str = Depends(get_user_id)):
    try:
        data, mime = parse_data_url(upload.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    ext = determine_extension(upload.filename, mime)
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Unsupported format (use JPG/PNG)")
    img = face_recognition.load_image_file(io.BytesIO(data))
    encodings = face_recognition.face_encodings(img)
    if not encodings:
        raise HTTPException(status_code=400, detail="No face found in image")

    entry_id = uuid.uuid4().hex
    filename = f"{entry_id}{ext}"
    _, blacklist_dir = resolve_user_paths(user_id)
    blacklist_dir.mkdir(parents=True, exist_ok=True)
    out_path = blacklist_dir / filename
    with out_path.open("wb") as f:
        f.write(data)

    entry = BlacklistEntry(id=entry_id, name=upload.name, reason=upload.reason, file=filename)
    cache = ensure_user_blacklist_loaded(user_id)
    cache["entries"].append(entry)
    cache["vectors"].append(encodings[0])
    save_metadata_for_user(user_id, cache["entries"])
    return entry


@app.delete("/blacklist/{entry_id}")
def delete_blacklist(entry_id: str, user_id: str = Depends(get_user_id)):
    cache = ensure_user_blacklist_loaded(user_id)
    entries = cache["entries"]
    vectors = cache["vectors"]
    metadata_path, blacklist_dir = resolve_user_paths(user_id)
    for idx, entry in enumerate(entries):
        if entry.id == entry_id:
            entries.pop(idx)
            if idx < len(vectors):
                vectors.pop(idx)
            try:
                (blacklist_dir / entry.file).unlink(missing_ok=True)
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] Could not delete file {entry.file}: {exc}")
            save_metadata_for_user(user_id, entries)
            return {"status": "deleted", "id": entry_id}
    raise HTTPException(status_code=404, detail="Entry not found")


@app.post("/detect", response_model=List[MatchResult])
def detect(frame: Frame, user_id: str = Depends(get_user_id)):
    cache = ensure_user_blacklist_loaded(user_id)
    entries = cache["entries"]
    vectors = cache["vectors"]

    try:
        image = decode_image(frame.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    face_locations = face_recognition.face_locations(image)
    face_encs = face_recognition.face_encodings(image, face_locations)
    results: List[MatchResult] = []

    for enc in face_encs:
        distances = face_recognition.face_distance(vectors, enc)
        if len(distances) == 0:
            results.append(MatchResult(status="unknown"))
            continue
        best_index = distances.argmin()
        best_distance = distances[best_index]
        if best_distance <= TOLERANCE:
            entry = entries[best_index]
            results.append(
                MatchResult(
                    status="blacklisted",
                    name=entry.name,
                    reason=entry.reason,
                    file=entry.file,
                    distance=float(best_distance),
                )
            )
        else:
            results.append(MatchResult(status="unknown", distance=float(best_distance)))

    if not results:
        results.append(MatchResult(status="no_face"))
    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
