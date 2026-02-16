import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from api.dependencies import issue_api_cookie
from api.routes.vehicles import router as vehicles_router
from api.upload_service import start_upload_cleanup_scheduler, stop_upload_cleanup_scheduler
from database.init_db import init_db

os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data", exist_ok=True)

app = FastAPI(title="ANPR API", version="1.0.0")
templates = Jinja2Templates(directory="ui")


@app.exception_handler(HTTPException)
def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": str(exc.detail)},
    )
@app.exception_handler(RequestValidationError)
def validation_exception_handler(_: Request, exc: RequestValidationError):
    first_error = exc.errors()[0]["msg"] if exc.errors() else "Validation error"
    return JSONResponse(
        status_code=422,
        content={"success": False, "message": first_error},
    )


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    start_upload_cleanup_scheduler()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await stop_upload_cleanup_scheduler()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    response = templates.TemplateResponse("home.html", {"request": request})
    issue_api_cookie(response)
    return response


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    response = templates.TemplateResponse("register.html", {"request": request})
    issue_api_cookie(response)
    return response


@app.get("/attendance", response_class=HTMLResponse)
def attendance_page(request: Request):
    response = templates.TemplateResponse("attendance.html", {"request": request})
    issue_api_cookie(response)
    return response


app.include_router(vehicles_router, prefix="/api", tags=["vehicles"])


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
