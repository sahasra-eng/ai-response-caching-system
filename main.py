import os
import json
import math
import uuid
from datetime import datetime, timedelta

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

client = genai.Client(api_key=API_KEY) if API_KEY else None

CACHE_FILE = "cache.json"
CHAT_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
SIMILARITY_THRESHOLD = 0.82
CACHE_MAX_AGE_DAYS = 30


def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {
            "entries": [],
            "stats": {"hits": 0, "misses": 0}
        }

    try:
        with open(CACHE_FILE, "r") as file:
            data = json.load(file)

        if isinstance(data, list):
            data = {"entries": data, "stats": {"hits": 0, "misses": 0}}

        data.setdefault("entries", [])
        data.setdefault("stats", {"hits": 0, "misses": 0})
        return data

    except (json.JSONDecodeError, OSError):
        return {
            "entries": [],
            "stats": {"hits": 0, "misses": 0}
        }


cache_data = load_cache()


def save_cache():
    with open(CACHE_FILE, "w") as file:
        json.dump(cache_data, file, indent=2)


def now_text():
    return datetime.now().isoformat()


def get_embedding(text):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text
    )
    return list(result.embeddings[0].values)


def cosine_similarity(vector_a, vector_b):
    if not vector_a or not vector_b:
        return 0.0

    if len(vector_a) != len(vector_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def automatic_forgetting():
    now = datetime.now()
    remaining = []

    for entry in cache_data["entries"]:
        try:
            created = datetime.fromisoformat(entry["created_at"])
            age_days = (now - created).days

            last_decay = datetime.fromisoformat(
                entry.get("last_decay_at", entry["created_at"])
            )

            if (now - last_decay).days >= 7:
                entry["importance"] = max(
                    1, int(entry.get("importance", 1)) - 1
                )
                entry["last_decay_at"] = now_text()

            expired = (
                age_days >= CACHE_MAX_AGE_DAYS
                and int(entry.get("importance", 1)) <= 1
            )

            if not expired:
                remaining.append(entry)

        except (ValueError, KeyError, TypeError):
            remaining.append(entry)

    if len(remaining) != len(cache_data["entries"]):
        cache_data["entries"] = remaining

    save_cache()


def rank_entries(query_vector):
    ranked = []
    now = datetime.now()

    for entry in cache_data["entries"]:
        embedding = entry.get("embedding", [])

        similarity = cosine_similarity(query_vector, embedding)

        try:
            last_used = datetime.fromisoformat(
                entry.get("last_used", entry["created_at"])
            )
            age_hours = max(0, (now - last_used).total_seconds() / 3600)
            recency = 1 / (1 + age_hours / 24)
        except (ValueError, KeyError):
            recency = 0

        importance = int(entry.get("importance", 1)) / 3

        score = (
            similarity * 0.70
            + importance * 0.20
            + recency * 0.10
        )

        ranked.append({
            "entry": entry,
            "similarity": similarity,
            "score": score
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    if client is None:
        return jsonify({
            "error": "Gemini API key not found. Check your .env file."
        }), 500

    data = request.get_json(silent=True) or {}
    query = str(data.get("message", "")).strip()

    if not query:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        automatic_forgetting()
        query_vector = get_embedding(query)
        ranked = rank_entries(query_vector)

        best = ranked[0] if ranked else None

        if best and best["similarity"] >= SIMILARITY_THRESHOLD:
            entry = best["entry"]

            entry["hits"] = int(entry.get("hits", 0)) + 1
            entry["importance"] = min(
                3, int(entry.get("importance", 1)) + 1
            )
            entry["last_used"] = now_text()

            cache_data["stats"]["hits"] += 1
            save_cache()

            return jsonify({
                "answer": entry["response"],
                "cache_status": "HIT",
                "similarity": round(best["similarity"], 3),
                "cached_at": entry["created_at"],
                "cache_id": entry["id"]
            })

        cache_data["stats"]["misses"] += 1

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=(
                "You are a helpful AI assistant. Give clear, accurate "
                "answers. Use readable paragraphs and examples when useful.\n\n"
                "User question: " + query
            )
        )

        answer = (response.text or "").strip()

        if not answer:
            answer = "The AI returned an empty response. Please try again."

        current_time = now_text()

        # Refine an existing entry when the new question is related,
        # but not similar enough to safely reuse its answer.
        if best and best["similarity"] >= 0.65:
            related = best["entry"]
            related["importance"] = min(
                3, int(related.get("importance", 1)) + 1
            )
            related["last_used"] = current_time

        new_entry = {
            "id": str(uuid.uuid4()),
            "query": query,
            "response": answer,
            "embedding": query_vector,
            "importance": 1,
            "hits": 0,
            "created_at": current_time,
            "last_used": current_time,
            "last_decay_at": current_time
        }

        cache_data["entries"].append(new_entry)
        save_cache()

        return jsonify({
            "answer": answer,
            "cache_status": "MISS",
            "similarity": round(best["similarity"], 3) if best else 0,
            "cached_at": current_time,
            "cache_id": new_entry["id"]
        })

    except Exception as error:
        print("Chat error:", repr(error))
        return jsonify({
            "error": (
                "The AI request failed. Check your API key, internet "
                "connection and Gemini API limits, then try again."
            )
        }), 500


@app.route("/api/cache")
def get_cache():
    automatic_forgetting()

    entries = []

    for entry in cache_data["entries"]:
        entries.append({
            "id": entry["id"],
            "query": entry["query"],
            "response": entry["response"],
            "importance": entry.get("importance", 1),
            "hits": entry.get("hits", 0),
            "created_at": entry["created_at"],
            "last_used": entry.get("last_used", entry["created_at"])
        })

    entries.reverse()
    return jsonify(entries)


@app.route("/api/stats")
def get_stats():
    stats = cache_data["stats"]
    hits = int(stats.get("hits", 0))
    misses = int(stats.get("misses", 0))
    total = hits + misses

    return jsonify({
        "total_cached": len(cache_data["entries"]),
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total * 100, 1) if total else 0
    })


@app.route("/api/cache/delete", methods=["POST"])
def delete_cache_entry():
    data = request.get_json(silent=True) or {}
    entry_id = data.get("id")

    before = len(cache_data["entries"])
    cache_data["entries"] = [
        entry for entry in cache_data["entries"]
        if entry.get("id") != entry_id
    ]

    if len(cache_data["entries"]) == before:
        return jsonify({"message": "Cache entry not found."}), 404

    save_cache()
    return jsonify({"message": "Cached response deleted."})


@app.route("/api/cache/update", methods=["POST"])
def update_cache_entry():
    data = request.get_json(silent=True) or {}
    entry_id = data.get("id")
    new_response = str(data.get("response", "")).strip()

    if not new_response:
        return jsonify({"message": "Response cannot be empty."}), 400

    for entry in cache_data["entries"]:
        if entry.get("id") == entry_id:
            entry["response"] = new_response
            entry["last_used"] = now_text()
            save_cache()
            return jsonify({"message": "Cached response updated."})

    return jsonify({"message": "Cache entry not found."}), 404


if __name__ == "__main__":
    app.run(debug=True)