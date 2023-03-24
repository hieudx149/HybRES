"""wsgi service."""
from src.service.service import app

if __name__ == "__main__":
    print("start")
    app.run(debug=False)
