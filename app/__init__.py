"""Flask application factory."""

from flask import Flask

from app.config import BASE_DIR, Config


def create_app(config_class=Config) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    app.config.from_object(config_class)
    config_class.ensure_directories()

    from app.routes import bp

    app.register_blueprint(bp)
    return app
