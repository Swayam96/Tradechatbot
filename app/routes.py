"""Flask routes and API endpoints."""

from flask import Blueprint, jsonify, render_template, request

from app.config import Config
from app.rag.pipeline import answer_query, rebuild_index
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    """Render the main chat interface."""
    return render_template(
        "index.html",
        app_name="TradeScope-AI",
        knowledge_base_url=Config.TARGET_WEBSITE_BASE_URL,
    )


@bp.route("/about")
def about():
    """Render the about page."""
    return render_template(
        "about.html",
        app_name="TradeScope-AI",
        knowledge_base_url=Config.TARGET_WEBSITE_BASE_URL,
        vector_store=Config.VECTOR_STORE_TYPE,
        embedding_model=Config.EMBEDDING_MODEL_NAME,
        llm_provider=Config.LLM_PROVIDER,
        llm_model=Config.LLM_MODEL_NAME,
    )


@bp.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat messages via the RAG pipeline."""
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Message cannot be empty.", "answer": "", "sources": []}), 400

    logger.info("Chat request: %s", message[:100])
    result = answer_query(message)
    return jsonify(result)


@bp.route("/api/rebuild-index", methods=["POST"])
def rebuild_index_endpoint():
    """Trigger website re-ingestion and vector index rebuild."""
    auth_header = request.headers.get("X-API-Key", "")
    expected_key = Config.REBUILD_API_KEY

    if not expected_key:
        return jsonify(
            {"error": "REBUILD_API_KEY is not configured on the server."}
        ), 503

    if auth_header != expected_key:
        return jsonify({"error": "Unauthorized."}), 401

    try:
        data = request.get_json(silent=True) or {}
        summary = rebuild_index(
            base_url=data.get("base_url"),
            max_pages=data.get("max_pages"),
            max_depth=data.get("max_depth"),
        )
        return jsonify({"status": "success", "summary": summary})
    except Exception as exc:
        logger.exception("Index rebuild failed")
        return jsonify({"error": str(exc)}), 500


@bp.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "app": "TradeScope-AI"})
