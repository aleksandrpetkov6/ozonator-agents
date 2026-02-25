from fastapi import FastAPI

app = FastAPI(
    title="Ozonator AA",
    description="Агент-администратор (АА) для контура АЗ-АС-АК",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "service": "AA",
        "status": "ok",
        "message": "Агент-администратор запущен"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }
