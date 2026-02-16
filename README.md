# ANPR Vehicle Attendance System

Automatic Number Plate Recognition for vehicle registration and attendance, built with FastAPI, SQLAlchemy, YOLO, and CRNN/Tesseract-based CRNN/OCR pipeline.

## Architecture Overview

This project follows **Clean Architecture** with practical API layering.

### Clean Architecture Mapping

- `core/`: Enterprise business rules (framework-independent)
  - `entities/`: Domain entities (`VehicleRegistrationEntity`, `VehicleAttendanceEntity`)
  - `ports/`: Interfaces (repository and detector contracts)
  - `usecases/`: Application business logic (`register_vehicle`, `mark_attendance`, normalization, fuzzy matching)
- `adapters/`: Infrastructure implementations of core ports
  - `repositories/sqlalchemy_adapter.py`: SQLAlchemy repository adapter
  - `detectors/yolo_tesseract_detector.py`: Detector adapter calling `detect.py`
- `api/`: Interface adapters for HTTP
  - `routes/vehicles.py`: API controllers
  - `viewmodels/vehicles.py`: Response models
  - `dependencies.py`: DI wiring, API-key auth, cookie issuing
  - `upload_service.py`: Upload validation/storage and cleanup scheduler
- `database/`: ORM models, engine/session setup, table initialization
- `ui/`: HTML views for home, register, and attendance

### MVC in This Project

- **Model**: `core/entities/`, `core/usecases/`, `database/models.py`
- **View**: `ui/home.html`, `ui/register.html`, `ui/attendance.html`
- **Controller**: `api/routes/vehicles.py` and route handlers in `main.py`

### MVVM in This Project (API layer)

- **View**: Browser/HTTP client and HTML pages
- **ViewModel**: `api/viewmodels/vehicles.py` (Pydantic response contracts)
- **Model**: Core entities/use-cases + persistence models

## Features

- Vehicle registration with image upload and manual vehicle number
- Attendance marking from image using detection + CRNN/OCR
- Vehicle number normalization and fuzzy matching (`rapidfuzz`, fallback to `SequenceMatcher`)
- Duplicate registration prevention
- API-key protection (`X-API-Key` header or secure cookie)
- Upload safety checks
  - File extensions: `.jpg`, `.jpeg`, `.png`
  - Max upload size via `MAX_UPLOAD_BYTES`
- Background cleanup for old uploads (`UPLOAD_TTL_SECONDS`)
- Transaction-safe SQLAlchemy writes with rollback on failure

## End-to-End Flow

1. User opens `/register`, uploads vehicle image + enters vehicle number.
2. Backend normalizes number and stores registration in DB.
3. User opens `/attendance`, uploads vehicle image.
4. Detector pipeline extracts plate text (`detect.py` via adapter).
5. Use-case fuzzy-matches detected text against registered vehicles.
6. If matched/registered, attendance is recorded; otherwise response is denied with detected details.

## Project Structure (Actual)

```text
ANPR_V1.1.1_AIP/
├── main.py
├── config.py
├── detect.py
├── requirements.txt
├── README.md
├── api/
│   ├── __init__.py
│   ├── dependencies.py
│   ├── upload_service.py
│   ├── routes/
│   │   ├── __init__.py
│   │   └── vehicles.py
│   └── viewmodels/
│       └── vehicles.py
├── core/
│   ├── entities/
│   │   └── __init__.py
│   ├── ports/
│   │   ├── plate_detector.py
│   │   └── repositories.py
│   └── usecases/
│       └── vehicle_service.py
├── adapters/
│   ├── detectors/
│   │   └── yolo_tesseract_detector.py
│   └── repositories/
│       └── sqlalchemy_adapter.py
├── database/
│   ├── __init__.py
│   ├── database.py
│   ├── init_db.py
│   └── models.py
├── ui/
│   ├── home.html
│   ├── register.html
│   └── attendance.html
├── scripts/
│   └── db_health.py
├── data/
│   └── uploads/
├── Temp/
│   └── unknown_plate.jpg
└── models/
    ├── yolo/
    └── crnn/
```

## API Routes

- `GET /` - Home page
- `GET /register` - Registration page
- `GET /attendance` - Attendance page
- `POST /api/register` - Register a vehicle
- `POST /api/mark-attendance` - Detect plate and mark attendance

Note:
- `/api/*` routes require API key.
- API key is accepted from `X-API-Key` or cookie `anpr_api_key`.
- Visiting UI routes issues cookie automatically.

## Request Formats

### Register Vehicle

`POST /api/register` (multipart/form-data)

- `image`: JPG/PNG file
- `vehicle_number`: string (required)
- `owner_name`: string (optional, defaults to vehicle number)

### Mark Attendance

`POST /api/mark-attendance` (multipart/form-data)

- `image`: JPG/PNG file

## Database

Configured for **MySQL** in current code.

Priority for DB config:
1. `DATABASE_URL`
2. `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DB`

Tables (from `database/models.py`):
- `vehicle_registrations`
- `vehicle_attendance`

## Setup

### 1. Clone and Install

```bash
git clone <your-repo-url>
cd ANPR_V1.1.1_AIP
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env.local` (or `.env.app`) and set:

```env
DATABASE_URL=mysql+pymysql://USER:PASSWORD@HOST:3306/DB_NAME

# Optional alternatives if DATABASE_URL is not set:
# MYSQL_USER=root
# MYSQL_PASSWORD=your_password
# MYSQL_HOST=localhost
# MYSQL_PORT=3306
# MYSQL_DB=anpr_db

# Security
ANPR_API_KEY=change-me-strong-random-key
COOKIE_SECURE=false

# Upload settings
MAX_UPLOAD_BYTES=5242880
UPLOAD_TTL_SECONDS=86400
UPLOAD_CLEANUP_INTERVAL_SECONDS=3600

# Model paths (optional)
YOLO_MODEL_PATH=models/yolo/best.pt
CRNN_MODEL_PATH=models/crnn/34e_crnn_simple_best.pth
```

### 3. Initialize Tables

```bash
python3 -m database.init_db
```

### 4. Run Server

```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

Open:
- `http://localhost:5000/`

## Development Utilities

### DB Health Check

```bash
python3 scripts/db_health.py
```

### Useful Notes

- Upload files are stored in `data/uploads/`.
- Expired upload files are deleted automatically by scheduler on startup.
- Detection pipeline entry point is `detect.detect_vehicle_number()`.

## Tech Stack

- FastAPI + Uvicorn
- SQLAlchemy + PyMySQL
- YOLO (Ultralytics)
- PyTorch + CRNN inference
- OpenCV + PIL
- Jinja2 templates

## License

MIT (add/update `LICENSE` file if needed).
